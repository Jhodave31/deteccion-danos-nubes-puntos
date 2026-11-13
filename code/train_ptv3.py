"""Entrenamiento y evaluación de Point Transformer V3 sobre 3DAeroRelief.

Mismo protocolo que train.py (PointNet++): bloques de 12 m x 12 m, Áreas 1,3-7 para entrenar,
Área 8 para validar, Área 2 para la prueba final, entropía cruzada ponderada y mismas métricas.

Modelos (--model):
  ptv3          PTv3 desde cero (codificador + decodificador, inicialización aleatoria)
  sonata_small  codificador PTv3 preentrenado con Sonata (CVPR 2025) + fine-tuning
  utonia_small  codificador PTv3 preentrenado con Utonia (ICML 2026) + fine-tuning
  (cualquier preset de point_transformer_v3 sin decodificador sirve igual)

Rasgos de entrada (9 canales, como Sonata/Utonia): coordenadas escaladas, RGB en [0,1] y normales.

Uso:
  python train_ptv3.py --data ../data/processed --out ../runs/ptv3 --model ptv3
  python train_ptv3.py --data ../data/processed --out ../runs/utonia --model utonia_small --scale 0.1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import point_transformer_v3 as ptv3
import torch
import torch.nn as nn
import torch.nn.functional as F
from point_transformer_v3 import upcast
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from train import CLASS_NAMES, NC, confusion, metrics

# PyTorch para Windows no incluye FlashAttention: se usa la atención estándar (SDPA) de PyTorch.
ptv3.set_backend(attention="sdpa")

# escala y celda de grilla por modelo: celda/escala = 0,1 m = vóxel del preprocesamiento
DEFAULTS = {"ptv3": (1.0, 0.1), "sonata": (0.2, 0.02), "concerto": (0.2, 0.02), "utonia": (0.1, 0.01)}


def model_defaults(name):
    for k, v in DEFAULTS.items():
        if name.startswith(k):
            return v
    return (1.0, 0.1)


def load_clouds(data_dir, areas):
    clouds = []
    for f in sorted(Path(data_dir).glob("*.npz")):
        area = f.stem.rsplit("_", 1)[0]
        if area in areas:
            d = np.load(f)
            xyz = d["xyz"].astype(np.float32)
            clouds.append(dict(name=f.stem, xyz=xyz, rgb=d["rgb"], label=d["label"].astype(np.int64),
                               normal=d["normal"].astype(np.float32), lo=xyz.min(0), hi=xyz.max(0)))
    return clouds


def block_arrays(c, idx, center):
    local = c["xyz"][idx] - center
    return local.astype(np.float32), c["rgb"][idx].astype(np.float32) / 255.0, c["normal"][idx].copy()


class BlockDataset(Dataset):
    def __init__(self, clouds, block, max_pts, iters):
        self.clouds, self.block, self.max_pts, self.iters = clouds, block, max_pts, iters
        sizes = np.array([len(c["xyz"]) for c in clouds], np.float64)
        self.p = sizes / sizes.sum()
        self.trees = None

    def __len__(self):
        return self.iters

    def _tree(self, k):
        if self.trees is None:
            self.trees = {}
        if k not in self.trees:
            self.trees[k] = cKDTree(self.clouds[k]["xyz"][:, :2])
        return self.trees[k]

    def __getitem__(self, i):
        rng = np.random.default_rng()
        k = rng.choice(len(self.clouds), p=self.p)
        c = self.clouds[k]
        for _ in range(10):
            center = c["xyz"][rng.integers(len(c["xyz"]))].copy()
            idx = np.asarray(self._tree(k).query_ball_point(center[:2], self.block / 2, p=np.inf), np.int64)
            if len(idx) >= 1024:
                break
        if len(idx) > self.max_pts:
            idx = rng.choice(idx, self.max_pts, replace=False)
        center[2] = c["xyz"][idx, 2].min()
        local, rgb, nrm = block_arrays(c, idx, center)
        # aumento de datos (igual que PointNet++): rotación en z, escala, reflexión, ruido, color
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
        s = rng.uniform(0.9, 1.1)
        local, nrm = local @ R.T * s, nrm @ R.T
        if rng.random() < 0.5:
            local[:, 0], nrm[:, 0] = -local[:, 0], -nrm[:, 0]
        local += np.clip(rng.normal(0, 0.01, local.shape), -0.05, 0.05).astype(np.float32)
        if rng.random() < 0.2:
            rgb[:] = 0
        else:
            rgb = np.clip(rgb * rng.uniform(0.8, 1.2) + rng.normal(0, 0.03), 0, 1).astype(np.float32)
        return local, rgb, nrm, c["label"][idx]


def collate_blocks(items):
    return items  # listas de arrays; el empaquetado (grilla, offset) se hace en GPU


def make_batch(blocks, scale, grid, device):
    """Empaqueta bloques en un ``Point`` de PTv3: una fila por celda de grilla (la primera)."""
    coords, feats, keep_all, offsets, n = [], [], [], [], 0
    for local, rgb, nrm in blocks:
        coord = torch.from_numpy(local).to(device) * scale
        g = torch.div(coord - coord.min(0)[0], grid, rounding_mode="floor").long()
        key = (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]
        uniq, inv = torch.unique(key, return_inverse=True)
        first = torch.full((len(uniq),), len(key), device=device, dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(len(key), device=device), reduce="amin")
        coord = coord[first]
        feat = torch.cat([coord, torch.from_numpy(rgb).to(device)[first], torch.from_numpy(nrm).to(device)[first]], 1)
        coords.append(coord)
        feats.append(feat)
        keep_all.append((first, inv))
        n += len(first)
        offsets.append(n)
    point = dict(coord=torch.cat(coords), feat=torch.cat(feats), grid_size=grid,
                 offset=torch.tensor(offsets, device=device))
    return point, keep_all


class Segmentor(nn.Module):
    def __init__(self, name, num_classes):
        super().__init__()
        if name == "ptv3":
            cfg = dict(ptv3.get_preset("ptv3").config)
            cfg["in_channels"] = 9
            self.backbone = ptv3.PointTransformerV3(config=ptv3.Config.from_dict(cfg))
            self.backbone._init_weights()
        else:
            self.backbone = ptv3.load(name, device="cpu", head=False, warmup=False)
        self.backbone.train()
        ch = self.backbone.feature_channels
        self.head = nn.Sequential(nn.Linear(ch, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, point):
        out = upcast(self.backbone(point))
        return self.head(out["feat"].float())


@torch.no_grad()
def predict_cloud(model, c, args, device, rot=0.0, flip=False):
    xyz = c["xyz"]
    probs = np.zeros((len(xyz), NC), np.float32)
    block, stride = args.block, args.block / 2
    lo, hi = c["lo"], c["hi"]
    xs = np.arange(lo[0], max(hi[0] - block, lo[0]) + stride, stride)
    ys = np.arange(lo[1], max(hi[1] - block, lo[1]) + stride, stride)
    R = np.array([[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1]], np.float32)
    rng = np.random.default_rng(0)
    batch, idxs = [], []

    def flush():
        if not batch:
            return
        point, keep = make_batch(batch, args.scale, args.grid, device)
        # BF16 también en inferencia: el FP16 por defecto de PTv3 en eval se desborda con BatchNorm
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(point)
        p = F.softmax(logits.float(), 1)
        start = 0
        for (first, inv), idx in zip(keep, idxs):
            pb = p[start:start + len(first)][inv].cpu().numpy()
            np.add.at(probs, idx, pb)
            start += len(first)
        batch.clear()
        idxs.clear()

    for x0 in xs:
        mx = (xyz[:, 0] >= x0) & (xyz[:, 0] <= x0 + block)
        for y0 in ys:
            idx_all = np.flatnonzero(mx & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y0 + block))
            if len(idx_all) < 32:
                continue
            center = np.array([x0 + block / 2, y0 + block / 2, xyz[idx_all, 2].min()], np.float32)
            parts = [idx_all] if len(idx_all) <= args.max_pts_eval else np.array_split(rng.permutation(idx_all), int(np.ceil(len(idx_all) / args.max_pts_eval)))
            for idx in parts:
                local, rgb, nrm = block_arrays(c, idx, center)
                local, nrm = local @ R.T, nrm @ R.T
                if flip:
                    local[:, 0], nrm[:, 0] = -local[:, 0], -nrm[:, 0]
                batch.append((local, rgb, nrm))
                idxs.append(idx)
                if len(batch) == args.eval_bs:
                    flush()
    flush()
    return probs


def evaluate(model, clouds, args, device, keep_preds=False, tta=False):
    model.eval()
    views = [(k * np.pi / 2, f) for f in (False, True) for k in range(4)] if tta else [(0.0, False)]
    cm = np.zeros((NC, NC), np.int64)
    preds = {}
    for c in clouds:
        probs = sum(predict_cloud(model, c, args, device, r, f) for r, f in views)
        pr = probs.argmax(1)
        cm += confusion(pr, c["label"])
        if keep_preds:
            preds[c["name"]] = pr
    model.train()
    return metrics(cm), cm, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ptv3")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--grid", type=float, default=None)
    ap.add_argument("--block", type=float, default=12.0)
    ap.add_argument("--max_pts", type=int, default=16000)
    ap.add_argument("--max_pts_eval", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--eval_bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_backbone", type=float, default=None, help="por defecto lr (desde cero) o lr/10 (preentrenado)")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--val_every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    s, g = model_defaults(args.model)
    args.scale = args.scale if args.scale is not None else s
    args.grid = args.grid if args.grid is not None else g
    if args.lr_backbone is None:
        args.lr_backbone = args.lr if args.model == "ptv3" else args.lr / 10

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_c = load_clouds(args.data, {f"Area_{i}" for i in (1, 3, 4, 5, 6, 7)})
    val_c = load_clouds(args.data, {"Area_8"})
    test_c = load_clouds(args.data, {"Area_2"})
    counts = sum(np.bincount(c["label"], minlength=NC) for c in train_c).astype(np.float64)
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq + 1e-6)
    weights = weights / weights.mean()
    wt = torch.tensor(weights, dtype=torch.float32, device=device)

    model = Segmentor(args.model, NC).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"modelo={args.model} parámetros={n_params / 1e6:.1f}M escala={args.scale} grilla={args.grid}", flush=True)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr},
    ], weight_decay=args.wd)
    loader = DataLoader(BlockDataset(train_c, args.block, args.max_pts, args.iters * args.bs), batch_size=args.bs,
                        num_workers=args.workers, persistent_workers=True, collate_fn=collate_blocks, drop_last=True)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_backbone, args.lr], total_steps=total, pct_start=0.05)

    log, best = [], -1.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for items in loader:
            point, keep = make_batch([(a, b, c) for a, b, c, _ in items], args.scale, args.grid, device)
            lab = torch.cat([torch.from_numpy(l).to(device)[first] for (_, _, _, l), (first, _) in zip(items, keep)])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(point)
            loss = F.cross_entropy(logits.float(), lab, weight=wt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            tl += loss.item() * lab.numel()
            tc += (logits.argmax(1) == lab).sum().item()
            tn += lab.numel()
        rec = dict(epoch=ep, loss=tl / tn, train_acc=tc / tn, time_min=(time.time() - t0) / 60,
                   mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        if ep % args.val_every == 0 or ep == args.epochs:
            vm, _, _ = evaluate(model, val_c, args, device)
            rec.update(val_mIoU=vm["mIoU"], val_OA=vm["OA"])
            if vm["mIoU"] > best:
                best = vm["mIoU"]
                torch.save(model.state_dict(), out / "best.pt")
        log.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    res = dict(args=vars(args), params=n_params, class_weights=weights.tolist(), best_val_mIoU=best,
               train_time_min=(time.time() - t0) / 60)
    for tta in (False, True):
        tm, cm, preds = evaluate(model, test_c, args, device, keep_preds=True, tta=tta)
        key = "test_tta" if tta else "test"
        res[key], res["confusion" + ("_tta" if tta else "")] = tm, cm.tolist()
        np.savez_compressed(out / f"{key}_preds.npz", **preds)
        print(key.upper() + ":", json.dumps({k: round(tm[k], 4) for k in ("OA", "mIoU", "mAcc", "mF1")}),
              "IoU:", dict(zip(CLASS_NAMES, np.round(tm["IoU"], 4))), flush=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
