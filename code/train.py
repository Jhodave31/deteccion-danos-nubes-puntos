"""Entrenamiento y evaluación de PointNet++ sobre 3DAeroRelief.

Protocolo (división oficial cruzada por áreas):
  - entrenamiento: Areas 1,3,4,5,6,7 (y 8 salvo que se use como validación)
  - validación:    Area_8 (selección del mejor checkpoint por mIoU)
  - prueba:        Area_2 (8 nubes), evaluada una sola vez con el mejor checkpoint

Uso:
  python train.py --data data/processed --out runs/pn2 --block 20 --epochs 60
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from pointnet2 import PointNet2Seg

CLASS_NAMES = ["Background", "Building-Damage", "Building-No-Damage", "Road", "Tree"]
NC = len(CLASS_NAMES)


def load_clouds(data_dir, areas):
    clouds = []
    for f in sorted(Path(data_dir).glob("*.npz")):
        area = f.stem.rsplit("_", 1)[0]
        if area in areas:
            d = np.load(f)
            xyz = d["xyz"].astype(np.float32)
            clouds.append(dict(name=f.stem, xyz=xyz, rgb=d["rgb"], label=d["label"].astype(np.int64),
                               zmin=xyz[:, 2].min(), zmax=xyz[:, 2].max(), lo=xyz.min(0), hi=xyz.max(0)))
    return clouds


def make_features(xyz_block, rgb, cloud, center, block):
    """Coordenadas locales normalizadas + rasgos de entrada (7 canales):
    RGB/255, altura relativa en la nube, posición XYZ normalizada en la nube."""
    local = (xyz_block - center) / block                           # coords. de la red (radios relativos al bloque)
    z_rel = (xyz_block[:, 2:3] - cloud["zmin"]) / max(cloud["zmax"] - cloud["zmin"], 1e-6)
    pos = (xyz_block - cloud["lo"]) / np.maximum(cloud["hi"] - cloud["lo"], 1e-6)
    feats = np.concatenate([rgb.astype(np.float32) / 255.0, z_rel, pos], 1)
    return local.astype(np.float32), feats.astype(np.float32)


class BlockDataset(Dataset):
    """Muestrea bloques columnares (block x block en XY) con `npts` puntos."""

    def __init__(self, clouds, block, npts, iters, augment):
        self.clouds, self.block, self.npts, self.iters, self.augment = clouds, block, npts, iters, augment
        sizes = np.array([len(c["xyz"]) for c in clouds], np.float64)
        self.p = sizes / sizes.sum()
        self.trees = None  # árboles k-d en XY, construidos de forma perezosa en cada worker

    def _tree(self, k):
        if self.trees is None:
            self.trees = {}
        if k not in self.trees:
            self.trees[k] = cKDTree(self.clouds[k]["xyz"][:, :2])
        return self.trees[k]

    def __len__(self):
        return self.iters

    def __getitem__(self, i):
        rng = np.random.default_rng()
        k = rng.choice(len(self.clouds), p=self.p)
        c = self.clouds[k]
        xyz = c["xyz"]
        tree = self._tree(k)
        for _ in range(10):
            center = xyz[rng.integers(len(xyz))].copy()
            # bloque cuadrado: bola en norma infinito de radio block/2 en XY
            idx = np.asarray(tree.query_ball_point(center[:2], self.block / 2, p=np.inf), dtype=np.int64)
            if len(idx) >= self.npts // 4:
                break
        idx = rng.choice(idx, self.npts, replace=len(idx) < self.npts)
        center[2] = xyz[idx, 2].min()
        p, rgb, lab = xyz[idx].copy(), c["rgb"][idx].copy(), c["label"][idx]
        local, feats = make_features(p, rgb, c, center, self.block)
        if self.augment:
            th = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
            local = local @ R.T * rng.uniform(0.9, 1.1)
            if rng.random() < 0.5:
                local[:, 0] = -local[:, 0]
            local += np.clip(rng.normal(0, 0.002, local.shape), -0.01, 0.01).astype(np.float32)
            if rng.random() < 0.2:                       # color drop (Qian et al., PointNeXt)
                feats[:, :3] = 0
            else:                                        # jitter de brillo/contraste
                feats[:, :3] = np.clip(feats[:, :3] * rng.uniform(0.8, 1.2) + rng.normal(0, 0.03), 0, 1)
        return torch.from_numpy(local), torch.from_numpy(feats), torch.from_numpy(lab)


def confusion(pred, gt):
    return np.bincount(gt * NC + pred, minlength=NC * NC).reshape(NC, NC)


def metrics(cm):
    tp = np.diag(cm).astype(np.float64)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        acc = tp / (tp + fn)
        prec = tp / (tp + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return dict(OA=float(tp.sum() / cm.sum()), mIoU=float(np.nanmean(iou)), mAcc=float(np.nanmean(acc)),
                mF1=float(np.nanmean(f1)), IoU=iou.tolist(), Acc=acc.tolist(), Prec=prec.tolist(), F1=f1.tolist())


@torch.no_grad()
def predict_cloud(model, c, block, npts, device, stride_ratio=0.5, bs=16, rot=0.0, flip=False):
    """Inferencia por ventana deslizante en XY con votación de probabilidades.
    `rot` (radianes) y `flip` rotan/reflejan las coordenadas locales (test-time augmentation)."""
    R = np.array([[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1]], np.float32)
    xyz = c["xyz"]
    probs = np.zeros((len(xyz), NC), np.float32)
    stride = block * stride_ratio
    lo, hi = c["lo"], c["hi"]
    xs = np.arange(lo[0], max(hi[0] - block, lo[0]) + stride, stride)
    ys = np.arange(lo[1], max(hi[1] - block, lo[1]) + stride, stride)
    batch = []

    def flush():
        if not batch:
            return
        L = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        Fe = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(L, Fe)
        out = F.softmax(out.float(), 1).permute(0, 2, 1).cpu().numpy()
        for (_, _, idx), o in zip(batch, out):
            np.add.at(probs, idx, o)
        batch.clear()

    rng = np.random.default_rng(0)
    for x0 in xs:
        mx = (xyz[:, 0] >= x0) & (xyz[:, 0] <= x0 + block)
        for y0 in ys:
            idx_all = np.flatnonzero(mx & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y0 + block))
            if len(idx_all) < 32:
                continue
            idx_all = rng.permutation(idx_all)
            center = np.array([x0 + block / 2, y0 + block / 2, xyz[idx_all, 2].min()], np.float32)
            for s in range(0, len(idx_all), npts):  # cubre todos los puntos del bloque
                idx = idx_all[s:s + npts]
                if len(idx) < npts:
                    idx = np.concatenate([idx, rng.choice(idx_all, npts - len(idx))])
                local, feats = make_features(xyz[idx], c["rgb"][idx], c, center, block)
                local = local @ R.T
                if flip:
                    local[:, 0] = -local[:, 0]
                batch.append((local, feats, idx))
                if len(batch) == bs:
                    flush()
    flush()
    return probs


def evaluate(model, clouds, args, device, keep_preds=False):
    model.eval()
    cm = np.zeros((NC, NC), np.int64)
    preds = {}
    for c in clouds:
        pr = predict_cloud(model, c, args.block, args.npts, device).argmax(1)
        cm += confusion(pr, c["label"])
        if keep_preds:
            preds[c["name"]] = pr
    return metrics(cm), cm, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=float, default=20.0, help="lado del bloque XY (unidades de la nube)")
    ap.add_argument("--npts", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--iters", type=int, default=500, help="bloques por época")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_c = load_clouds(args.data, {f"Area_{i}" for i in (1, 3, 4, 5, 6, 7)})
    val_c = load_clouds(args.data, {"Area_8"})
    test_c = load_clouds(args.data, {"Area_2"})
    print(f"nubes: train={len(train_c)} val={len(val_c)} test={len(test_c)}")

    counts = sum(np.bincount(c["label"], minlength=NC) for c in train_c).astype(np.float64)
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq + 1e-6)          # pesos por frecuencia inversa (raíz)
    weights = weights / weights.mean()
    print("frecuencias:", dict(zip(CLASS_NAMES, freq.round(4))), "pesos:", weights.round(3))

    loader = DataLoader(BlockDataset(train_c, args.block, args.npts, args.iters * args.bs, True),
                        batch_size=args.bs, num_workers=args.workers, persistent_workers=True,
                        pin_memory=True, drop_last=True)
    model = PointNet2Seg(in_feats=7, num_classes=NC).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    wt = torch.tensor(weights, dtype=torch.float32, device=device)

    log, best = [], -1.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for local, feats, lab in loader:
            local, feats, lab = local.to(device), feats.to(device), lab.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(local, feats)
                loss = F.cross_entropy(logits.float(), lab, weight=wt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * lab.numel()
            tc += (logits.argmax(1) == lab).sum().item()
            tn += lab.numel()
        rec = dict(epoch=ep, loss=tl / tn, train_acc=tc / tn, lr=sched.get_last_lr()[0], time_min=(time.time() - t0) / 60)
        if ep % args.val_every == 0 or ep == args.epochs:
            vm, _, _ = evaluate(model, val_c, args, device)
            rec.update(val_mIoU=vm["mIoU"], val_OA=vm["OA"], val_IoU=vm["IoU"])
            if vm["mIoU"] > best:
                best = vm["mIoU"]
                torch.save(model.state_dict(), out / "best.pt")
        log.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "val_IoU"}), flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))

    torch.save(model.state_dict(), out / "last.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    tm, cm, preds = evaluate(model, test_c, args, device, keep_preds=True)
    np.savez_compressed(out / "test_preds.npz", **preds)
    res = dict(args=vars(args), params=n_params, class_weights=weights.tolist(), train_freq=freq.tolist(),
               best_val_mIoU=best, test=tm, confusion=cm.tolist(), train_time_min=(time.time() - t0) / 60)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print("TEST:", json.dumps({k: tm[k] for k in ("OA", "mIoU", "mAcc", "mF1")}))
    print("IoU por clase:", dict(zip(CLASS_NAMES, np.round(tm["IoU"], 4))))


if __name__ == "__main__":
    main()
