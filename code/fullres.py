"""Evalúa predicciones de cualquier modelo a la resolución ORIGINAL del Área 2.

Toma las predicciones sobre los puntos submuestreados (``*_preds.npz``, guardadas por train.py,
train_ptv3.py o evaluate.py), aplica a las nubes originales la misma alineación vertical de
preprocess.py y asigna a cada punto original la predicción de su vecino submuestreado más cercano.

Uso: python fullres.py --data ../data/processed --raw ../data/3DAeroRelief --preds ../runs/ptv3/test_tta_preds.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from preprocess import align_vertical, colors_to_labels, read_normals, read_ply
from train import CLASS_NAMES, NC, confusion, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--preds", required=True)
    a = ap.parse_args()
    preds = np.load(a.preds)
    cm = np.zeros((NC, NC), np.int64)
    for name in sorted(preds.files):
        area, cloud = name.rsplit("_", 1)
        vox = np.load(Path(a.data) / f"{name}.npz")["xyz"]
        cache = Path(a.data).parent / "fullres_cache" / f"{name}.npz"
        if cache.exists():  # nube original ya alineada + etiquetas (se calcula una sola vez)
            d = np.load(cache)
            xyz, lab = d["xyz"], d["label"]
        else:
            pp = Path(a.raw) / area / f"{cloud}.ply"
            xyz, _ = read_ply(pp)
            xyz_l, rgb_l = read_ply(pp.parent / f"segment{cloud}.ply")
            lab = colors_to_labels(rgb_l)
            if not (len(xyz_l) == len(xyz) and np.allclose(xyz_l[::997], xyz[::997])):
                lab = lab[cKDTree(xyz_l).query(xyz, k=1, workers=-1)[1]]
            xyz, _ = align_vertical(xyz, read_normals(pp))
            xyz = xyz.astype(np.float32)
            cache.parent.mkdir(exist_ok=True)
            tmp = cache.with_name(cache.stem + "_tmp.npz")  # escribir y renombrar: nunca queda a medias
            np.savez(tmp, xyz=xyz, label=lab.astype(np.uint8))
            tmp.replace(cache)
        _, nn = cKDTree(vox).query(xyz, k=1, workers=-1)
        cm += confusion(preds[name][nn], lab.astype(np.int64))
        print(name, len(xyz), flush=True)
    m = metrics(cm)
    out = Path(a.preds).with_name(Path(a.preds).stem.replace("_preds", "") + "_fullres.json")
    out.write_text(json.dumps(dict(full=m, confusion_full=cm.tolist()), indent=1))
    print(f"[fullres] mIoU={m['mIoU']:.4f} OA={m['OA']:.4f} mAcc={m['mAcc']:.4f} mF1={m['mF1']:.4f}",
          dict(zip(CLASS_NAMES, np.round(m["IoU"], 4))))


if __name__ == "__main__":
    main()
