"""Precalcula los análisis de la demo para las nubes del Área 2 (y opcionalmente otras), de modo que
en la sustentación se abran al instante. Usa el modelo indicado en demo/config.json.

Uso (desde demo/):  python precache.py [Area_2 Area_8 ...]
"""
import sys

import app
from predictor import Predictor

areas = sys.argv[1:] or ["Area_2"]
cfg = app.json.loads((app.HERE / "config.json").read_text())
app.predictor = Predictor(app.ROOT / cfg["run"])
print("Modelo:", app.predictor.name)
for f in sorted(app.RAW.glob("Area_*/pp*.ply")):
    if f.parent.name not in areas:
        continue
    key = f"{f.parent.name}/{f.stem}|0"
    res = app.analyze(f, key, False, persist=True)
    acc = res["summary"].get("accuracy")
    print(f"{key}: {res['n_vox']:,} puntos, acierto {acc:.3f}" if acc is not None else key, flush=True)
