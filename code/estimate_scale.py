"""Armonización de escala entre áreas de entrenamiento.

El artículo de 3DAeroRelief indica que las nubes se reescalaron tomando como referencia una casa
de un piso de ~3 m. Sin embargo, en las Áreas 4 y 6 las edificaciones aparecen varias veces más
altas que en el resto. Este script estima, para cada área de ENTRENAMIENTO/VALIDACIÓN, la altura
típica de las edificaciones sobre el suelo:

    h_area = mediana sobre nubes de [ mediana(z | edificación) - percentil 5 de z ]

y propone el factor  s_area = mediana(h de las demás áreas) / h_area  cuando el área se desvía más
de 1,5 veces. El Área 2 (prueba) se excluye: no se usan sus etiquetas para nada.

Uso: python estimate_scale.py data/processed scale.json
"""
import json
import sys
from pathlib import Path

import numpy as np

TEST_AREA = "Area_2"


def main(data_dir, out_json):
    per_area = {}
    for f in sorted(Path(data_dir).glob("*.npz")):
        area = f.stem.rsplit("_", 1)[0]
        if area == TEST_AREA:
            continue
        d = np.load(f)
        z, lab = d["xyz"][:, 2], d["label"]
        b = np.isin(lab, (1, 2))
        if b.sum() < 1000:
            continue
        per_area.setdefault(area, []).append(float(np.median(z[b]) - np.percentile(z, 5)))
    h = {a: float(np.median(v)) for a, v in per_area.items()}
    scales = {}
    for a, ha in sorted(h.items()):
        ref = float(np.median([v for k, v in h.items() if k != a]))
        r = ref / ha
        scales[a] = round(r, 3) if (r < 1 / 1.5 or r > 1.5) else 1.0
        print(f"{a}: altura típica edificación {ha:.2f} (ref. {ref:.2f}) -> factor {scales[a]}")
    Path(out_json).write_text(json.dumps(dict(height=h, scale=scales), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
