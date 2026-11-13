"""Agrega normales de superficie a las nubes preprocesadas (.npz).

Las normales originales solo existen en 54 de las 64 nubes, así que se recalculan para todas de
forma uniforme sobre los puntos ya alineados y submuestreados (0,1 m): para cada punto, el vector
propio de menor valor propio de la covarianza de sus k vecinos más cercanos (PCA local), orientado
hacia +z (hacia arriba, donde estaba el dron). No usa etiquetas.

Uso: python add_normals.py ../data/processed [--k 16]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def estimate_normals(xyz, k=16, chunk=200_000):
    tree = cKDTree(xyz)
    normals = np.empty_like(xyz, dtype=np.float32)
    for s in range(0, len(xyz), chunk):
        p = xyz[s:s + chunk]
        _, idx = tree.query(p, k=k, workers=-1)
        nb = xyz[idx] - xyz[idx].mean(1, keepdims=True)          # [n,k,3]
        cov = np.einsum("nki,nkj->nij", nb, nb) / k
        _, vecs = np.linalg.eigh(cov)                            # ascendente
        n = vecs[:, :, 0]
        n[n[:, 2] < 0] *= -1                                     # orientar hacia arriba
        normals[s:s + chunk] = n
    return normals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    for f in sorted(Path(a.data).glob("*.npz")):
        d = dict(np.load(f))
        d["normal"] = estimate_normals(d["xyz"].astype(np.float64), a.k)
        np.savez_compressed(f, **d)
        print(f.stem, len(d["xyz"]), "normal_z medio:", round(float(d["normal"][:, 2].mean()), 3), flush=True)


if __name__ == "__main__":
    main()
