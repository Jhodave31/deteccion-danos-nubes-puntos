"""Carga un modelo entrenado (PointNet++ o PTv3) y segmenta una nube .ply completa.

Aplica exactamente el mismo preprocesamiento del experimento (code/preprocess.py y
code/add_normals.py): alineación vertical, submuestreo por vóxel de 0,1 m y normales por PCA.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from add_normals import estimate_normals  # noqa: E402
from preprocess import align_vertical, colors_to_labels, read_normals, read_ply, voxel_subsample  # noqa: E402

CLASS_NAMES = ["Fondo", "Edificación dañada", "Edificación sin daño", "Vía", "Árbol"]
CLASS_COLORS = [(135, 135, 135),(230, 25, 75), (70, 240, 240), (255, 225, 25), (0, 128, 0)]


class Predictor:
    def __init__(self, run_dir, device=None):
        self.run = Path(run_dir)
        res = json.loads((self.run / "results.json").read_text())
        self.args = Namespace(**res["args"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kind = "ptv3" if hasattr(self.args, "model") else "pointnet2"
        if self.kind == "ptv3":
            from train_ptv3 import Segmentor
            self.model = Segmentor(self.args.model, 5)
            self.args.eval_bs = 2
        else:
            from pointnet2 import PointNet2Seg
            self.model = PointNet2Seg(in_feats=7, num_classes=5)
        self.model.load_state_dict(torch.load(self.run / "best.pt", map_location="cpu"))
        self.model.to(self.device).eval()
        self.name = f"PTv3 ({self.args.model})" if self.kind == "ptv3" else "PointNet++"

    def load_cloud(self, ply_path, scale=1.0, with_labels=True):
        """Lee y preprocesa un .ply de 3DAeroRelief (o cualquier nube con xyz + RGB)."""
        ply_path = Path(ply_path)
        xyz, rgb = read_ply(ply_path)
        n_raw = len(xyz)
        label = None
        seg = ply_path.parent / f"segment{ply_path.name}"
        if with_labels and seg.exists():
            xyz_l, rgb_l = read_ply(seg)
            label = colors_to_labels(rgb_l)
            if not (len(xyz_l) == len(xyz) and np.allclose(xyz_l[::997], xyz[::997])):
                from scipy.spatial import cKDTree
                label = label[cKDTree(xyz_l).query(xyz, k=1, workers=-1)[1]]
        xyz, _ = align_vertical(xyz, read_normals(ply_path))
        xyz = xyz * scale
        keep = voxel_subsample(xyz, 0.1)
        xyz, rgb = xyz[keep].astype(np.float32), rgb[keep]
        label = label[keep].astype(np.int64) if label is not None else None
        normal = estimate_normals(xyz.astype(np.float64))
        c = dict(name=ply_path.stem, xyz=xyz, rgb=rgb, normal=normal, lo=xyz.min(0), hi=xyz.max(0),
                 zmin=xyz[:, 2].min(), zmax=xyz[:, 2].max(), n_raw=n_raw)
        if label is not None:
            c["label"] = label
        return c

    @torch.no_grad()
    def predict(self, c, tta=False):
        views = [(k * np.pi / 2, f) for f in (False, True) for k in range(4)] if tta else [(0.0, False)]
        if self.kind == "ptv3":
            from train_ptv3 import predict_cloud
            probs = sum(predict_cloud(self.model, c, self.args, self.device, r, f) for r, f in views)
        else:
            from train import predict_cloud
            probs = sum(predict_cloud(self.model, c, self.args.block, self.args.npts, self.device, rot=r, flip=f)
                        for r, f in views)
        return probs.argmax(1)


def summarize(pred, label=None):
    n = len(pred)
    counts = np.bincount(pred, minlength=5)
    out = dict(points=int(n), classes=[dict(name=CLASS_NAMES[k], color=CLASS_COLORS[k], points=int(counts[k]),
                                            percent=float(100 * counts[k] / n)) for k in range(5)])
    b = counts[1] + counts[2]
    out["damage_ratio"] = float(100 * counts[1] / b) if b else 0.0
    if label is not None:
        out["accuracy"] = float((pred == label).mean())
        inter = [int(((pred == k) & (label == k)).sum()) for k in range(5)]
        union = [int(((pred == k) | (label == k)).sum()) for k in range(5)]
        out["iou"] = [i / u if u else None for i, u in zip(inter, union)]
    return out
