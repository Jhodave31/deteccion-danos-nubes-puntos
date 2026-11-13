"""Evaluación de un modelo entrenado sobre el Área 2 (prueba).

  - Predice sobre los puntos submuestreados (0,1 m) con ventana deslizante.
  - Opcional: test-time augmentation (TTA) promediando 8 vistas (4 rotaciones x reflexión).
  - Opcional: proyecta las probabilidades a la resolución ORIGINAL de cada nube por vecino
    más cercano (misma alineación vertical que preprocess.py) y evalúa sobre todos los puntos.

Uso:
  python evaluate.py --data ../data/processed --raw ../data/3DAeroRelief --run ../runs/pn2 --tta --fullres
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from pointnet2 import PointNet2Seg
from preprocess import align_vertical, colors_to_labels, read_normals, read_ply
from train import CLASS_NAMES, NC, confusion, load_clouds, metrics, predict_cloud

TTA_VIEWS = [(k * np.pi / 2, f) for f in (False, True) for k in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", default=None, help="carpeta 3DAeroRelief original (para --fullres)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--fullres", action="store_true")
    ap.add_argument("--area", default="Area_2")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    run = Path(a.run)
    args = json.loads((run / "results.json").read_text())["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointNet2Seg(in_feats=7, num_classes=NC).to(device)
    model.load_state_dict(torch.load(run / a.ckpt, map_location=device))
    model.eval()

    views = TTA_VIEWS if a.tta else [(0.0, False)]
    clouds = load_clouds(a.data, {a.area})
    cm_vox = np.zeros((NC, NC), np.int64)
    cm_full = np.zeros((NC, NC), np.int64)
    per_cloud, all_preds = {}, {}
    for c in clouds:
        probs = sum(predict_cloud(model, c, args["block"], args["npts"], device, rot=r, flip=f) for r, f in views)
        pred = probs.argmax(1)
        cm_vox += confusion(pred, c["label"])
        all_preds[c["name"]] = pred
        rec = dict(n_vox=len(pred), acc_vox=float((pred == c["label"]).mean()))
        if a.fullres:
            cloud = c["name"].rsplit("_", 1)[1]
            pp = Path(a.raw) / a.area / f"{cloud}.ply"
            xyz, _ = read_ply(pp)
            xyz_l, rgb_l = read_ply(pp.parent / f"segment{cloud}.ply")
            lab = colors_to_labels(rgb_l)
            if not (len(xyz_l) == len(xyz) and np.allclose(xyz_l[::997], xyz[::997])):
                lab = lab[cKDTree(xyz_l).query(xyz, k=1, workers=-1)[1]]
            xyz, _ = align_vertical(xyz, read_normals(pp))
            _, nn = cKDTree(c["xyz"]).query(xyz.astype(np.float32), k=1, workers=-1)
            pf = pred[nn]
            cm_full += confusion(pf, lab.astype(np.int64))
            rec.update(n_full=len(pf), acc_full=float((pf == lab).mean()))
        per_cloud[c["name"]] = rec
        print(c["name"], {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)

    out = dict(ckpt=a.ckpt, tta=a.tta, voxel=metrics(cm_vox), confusion_voxel=cm_vox.tolist(), per_cloud=per_cloud)
    if a.fullres:
        out.update(full=metrics(cm_full), confusion_full=cm_full.tolist())
    tag = a.tag or ("eval" + ("_tta" if a.tta else "") + ("_full" if a.fullres else ""))
    (run / f"{tag}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(run / f"{tag}_preds.npz", **all_preds)
    for k in ("voxel", "full"):
        if k in out:
            m = out[k]
            print(f"[{k}] mIoU={m['mIoU']:.4f} OA={m['OA']:.4f} mAcc={m['mAcc']:.4f} mF1={m['mF1']:.4f} IoU=",
                  dict(zip(CLASS_NAMES, np.round(m["IoU"], 4))))


if __name__ == "__main__":
    main()
