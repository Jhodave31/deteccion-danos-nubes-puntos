"""Diagnóstico del sentido de la vertical tras align_vertical().

Para cada nube compara dos reglas sin etiquetas:
  - asimetría de alturas (la usada en preprocess.py)
  - normales: COLMAP orienta las normales hacia la cámara (el dron), así que su
    componente vertical media debería ser positiva si la vertical es correcta
y las contrasta con una verificación con etiquetas (edificaciones/árboles más
altos que vías/fondo), usada solo como referencia.
"""
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import align_vertical  # noqa: E402


def main(raw_dir, stats_path):
    stats = {(s["area"], s["cloud"]): s for s in json.loads(Path(stats_path).read_text())}
    rows = []
    for pp in sorted(Path(raw_dir).rglob("pp*.ply")):
        key = (pp.parent.name, pp.stem)
        if key not in stats:
            continue
        v = PlyData.read(str(pp))["vertex"].data
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        c = xyz.mean(0)
        sub = xyz[:: max(1, len(xyz) // 500_000)] - c
        # reconstruir la vertical final usada en preprocess (misma función)
        w, V = np.linalg.eigh(np.cov(sub.T))
        up = V[:, 0]
        h = sub @ up
        skew_sign = 1 if ((h - h.mean()) ** 3).mean() >= 0 else -1
        up_final = up * skew_sign  # vertical elegida por preprocess.py
        normal_vote = None
        if "nx" in v.dtype.names:
            n = np.stack([v["nx"], v["ny"], v["nz"]], 1)[:: max(1, len(xyz) // 500_000)]
            normal_vote = float(np.mean(n @ up_final))
        z = stats[key]["z_mean_per_class"]
        hi = [z[k] for k in (1, 2, 4) if z[k] is not None]
        lo = [z[k] for k in (0, 3) if z[k] is not None]
        label_ok = bool(np.mean(hi) > np.mean(lo)) if hi and lo else None
        rows.append(dict(area=key[0], cloud=key[1], normal_vote=normal_vote, label_ok=label_ok))
        print(key, "normal_vote=", None if normal_vote is None else round(normal_vote, 3), "label_ok=", label_ok, flush=True)
    Path(stats_path).with_name("orientation.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
