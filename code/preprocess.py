"""Preprocesamiento de 3DAeroRelief.

Para cada nube pp<i>.ply:
  1. Lee geometría + RGB (pp<i>.ply) y la nube de etiquetas (segmentpp<i>.ply).
  2. Convierte el color de la nube de etiquetas a un ID de clase (color más cercano,
     igual que el script oficial add_labels_and_viz.py).
  3. Transfiere la etiqueta a cada punto de la geometría por vecino más cercano (KD-tree).
  4. Alinea la nube con la vertical: las reconstrucciones SfM (COLMAP) tienen orientación
     arbitraria, así que se toma como "arriba" el eje de menor varianza (PCA, plano del suelo)
     y su sentido se fija con las normales: COLMAP las orienta hacia la cámara (el dron), por lo
     que su componente vertical media debe ser positiva. Si la nube no trae normales, se usa la
     asimetría de la distribución de alturas (suelo abajo, techos y copas como cola superior).
     No usa etiquetas.
  4b. (Opcional) aplica un factor de escala por área (--scale, ver estimate_scale.py).
  5. Submuestrea por vóxel (grid subsampling) para homogeneizar la densidad.
  6. Guarda un .npz comprimido con xyz (float32), rgb (uint8) y label (uint8).

Uso: python preprocess.py --input data/3DAeroRelief --output data/processed --voxel 0.05
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

CLASS_NAMES = ["Background", "Building-Damage", "Building-No-Damage", "Road", "Tree"]
CLASS_COLORS = np.array([(0, 0, 0), (230, 25, 75), (70, 240, 240), (255, 255, 25), (0, 128, 0)], dtype=np.float32)


def read_normals(path):
    v = PlyData.read(str(path))["vertex"].data
    if "nx" not in v.dtype.names:
        return None
    return np.stack([v["nx"], v["ny"], v["nz"]], 1)


def read_ply(path):
    v = PlyData.read(str(path))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    names = v.dtype.names
    if "red" in names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255 if rgb.max() <= 1.0 else rgb).clip(0, 255)
        rgb = rgb.astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 255, np.uint8)
    return xyz, rgb


def colors_to_labels(rgb):
    out = np.empty(len(rgb), np.uint8)
    for s in range(0, len(rgb), 2_000_000):
        c = rgb[s:s + 2_000_000].astype(np.float32)
        d = ((c[:, None, :] - CLASS_COLORS[None]) ** 2).sum(-1)
        out[s:s + 2_000_000] = d.argmin(1)
    return out


def align_vertical(xyz, normals=None):
    c = xyz.mean(0)
    sub = xyz[:: max(1, len(xyz) // 500_000)] - c
    w, V = np.linalg.eigh(np.cov(sub.T))           # autovalores ascendentes
    R = V[:, [2, 1, 0]]                            # x,y = ejes principales del suelo; z = normal del suelo
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    step = max(1, len(xyz) // 500_000)
    if normals is not None:                        # normales orientadas a la cámara -> deben apuntar arriba
        flip = np.mean(normals[::step] @ R[:, 2]) < 0
    else:                                          # sin normales: asimetría negativa -> invertida
        h = sub @ R[:, 2]
        flip = ((h - h.mean()) ** 3).mean() < 0
    if flip:
        R[:, 2], R[:, 0] = -R[:, 2], -R[:, 0]
    out = (xyz - c) @ R
    return out - out.min(0), np.sqrt(w)


def voxel_subsample(xyz, voxel):
    """Conserva un punto aleatorio por vóxel."""
    g = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    dims = g.max(0) + 1
    keys = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
    perm = np.random.default_rng(0).permutation(len(xyz))
    _, first = np.unique(keys[perm], return_index=True)
    return np.sort(perm[first])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--voxel", type=float, default=0.0, help="tamaño de vóxel (0 = sin submuestreo)")
    ap.add_argument("--scale", default=None, help="JSON de estimate_scale.py con factores por área")
    ap.add_argument("--areas", nargs="*", default=None, help="procesar solo estas áreas")
    args = ap.parse_args()
    scales = json.loads(Path(args.scale).read_text())["scale"] if args.scale else {}

    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else []
    for pp in sorted(inp.rglob("pp*.ply")):
        area = pp.parent.name
        if args.areas and area not in args.areas:
            continue
        stats = [s for s in stats if (s["area"], s["cloud"]) != (area, pp.stem)]
        seg = pp.parent / f"segment{pp.name}"
        if not seg.exists():
            print("sin etiquetas:", pp)
            continue
        xyz, rgb = read_ply(pp)
        xyz_l, rgb_l = read_ply(seg)
        lab_src = colors_to_labels(rgb_l)
        if len(xyz_l) == len(xyz) and np.allclose(xyz_l[::997], xyz[::997]):
            label = lab_src                        # misma geometría: correspondencia directa
        else:
            _, idx = cKDTree(xyz_l).query(xyz, k=1, workers=-1)
            label = lab_src[idx]
        n_raw = len(xyz)
        xyz, sig = align_vertical(xyz, read_normals(pp))
        factor = float(scales.get(area, 1.0))
        xyz = xyz * factor
        if args.voxel > 0:
            keep = voxel_subsample(xyz, args.voxel)
            xyz, rgb, label = xyz[keep], rgb[keep], label[keep]
        ext = xyz.max(0) - xyz.min(0)
        np.savez_compressed(out / f"{area}_{pp.stem}.npz", xyz=xyz.astype(np.float32), rgb=rgb, label=label)
        counts = np.bincount(label, minlength=5).tolist()
        # verificación a posteriori (no se usa para alinear): altura media por clase
        zc = [float(xyz[label == k, 2].mean()) if counts[k] else None for k in range(5)]
        stats.append(dict(area=area, cloud=pp.stem, n_raw=n_raw, n=len(xyz), extent=ext.round(2).tolist(),
                          pca_sigma=sig.round(3).tolist(), scale=factor, counts=counts, z_mean_per_class=zc))
        print(f"{area}/{pp.name}: {n_raw:,} -> {len(xyz):,} pts, extent={ext.round(1)}, counts={counts}, "
              f"z_por_clase={[round(z, 1) if z is not None else None for z in zc]}", flush=True)
    stats.sort(key=lambda s: (int(s["area"].split("_")[1]), s["cloud"]))
    stats_path.write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
