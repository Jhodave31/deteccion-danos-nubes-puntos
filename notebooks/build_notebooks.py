"""Genera los tres notebooks del proyecto (luego se ejecutan con nbconvert para guardar sus salidas).

Uso (desde notebooks/):  python build_notebooks.py
"""
import nbformat as nbf

KERNEL = {"display_name": "Python (electiva)", "language": "python", "name": "electiva"}


def nb(cells, path):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = KERNEL
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("MD:") else nbf.v4.new_code_cell(c) for c in cells]
    nbf.write(n, path)
    print("escrito", path)


SETUP = r'''import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd().parent            # carpeta raíz del proyecto
sys.path.insert(0, str(ROOT / "code"))
DATA, RAW, RUNS = ROOT / "data" / "processed", ROOT / "data" / "3DAeroRelief", ROOT / "runs"
CLASES = ["Fondo", "Edif. dañada", "Edif. sin daño", "Vía", "Árbol"]
pd.set_option("display.precision", 3)
print("Proyecto:", ROOT.name)'''

# --------------------------------------------------------------------------------------------
nb([
"MD:# 1. Datos y preprocesamiento — 3DAeroRelief\n"
"Detección de daños estructurales en nubes de puntos de dron (Electiva de Ingeniería de Software, USB Cali).\n\n"
"Este notebook documenta el dataset y el preprocesamiento descritos en la sección *Metodología* del informe:\n"
"fusión de etiquetas, **alineación vertical**, **armonización de escala**, submuestreo por vóxel y normales.\n\n"
"> El dataset original (≈21 GB) se descarga de <https://github.com/BinaLab/3DAeroRelief> y se descomprime en `data/3DAeroRelief/`.",
SETUP,
"MD:## 1.1 Comandos de preprocesamiento\n"
"El preprocesamiento completo tarda ~30 min, así que se ejecuta con los scripts de `code/`. Cambie `EJECUTAR` a `True` para repetirlo.",
r'''EJECUTAR = False
comandos = [
    "python preprocess.py --input ../data/3DAeroRelief --output ../data/processed --voxel 0.1",
    "python estimate_scale.py ../data/processed ../data/scale.json",
    "python preprocess.py --input ../data/3DAeroRelief --output ../data/processed --voxel 0.1 --scale ../data/scale.json --areas Area_4 Area_6",
    "python add_normals.py ../data/processed",
]
for c in comandos:
    print(c)
if EJECUTAR:
    import subprocess
    for c in comandos:
        subprocess.run([sys.executable] + c.split()[1:], cwd=ROOT / "code", check=True)''',
"MD:## 1.2 Estadísticas del dataset (Tabla de estadísticas del informe)",
r'''stats = json.loads((DATA / "stats.json").read_text())
df = pd.DataFrame(stats)
uso = {"Area_2": "Prueba", "Area_8": "Validación"}
filas = []
for area, g in df.groupby("area", sort=False):
    c = np.sum(np.stack(g["counts"].values), 0)
    filas.append(dict(Área=area, Uso=uso.get(area, "Entrenamiento"), Nubes=len(g),
                      **{"Puntos orig. (M)": g["n_raw"].sum() / 1e6, "Tras vóxel (M)": g["n"].sum() / 1e6},
                      **{f"{k} %": 100 * v / c.sum() for k, v in zip(CLASES, c)}))
tabla = pd.DataFrame(filas).sort_values("Área", key=lambda s: s.str[5:].astype(int))
tot = np.sum(np.stack(df["counts"].values), 0)
tabla.loc[len(tabla)] = dict(Área="Total", Uso="—", Nubes=len(df), **{"Puntos orig. (M)": df["n_raw"].sum() / 1e6,
                             "Tras vóxel (M)": df["n"].sum() / 1e6}, **{f"{k} %": 100 * v / tot.sum() for k, v in zip(CLASES, tot)})
tabla.round(1)''',
"MD:## 1.3 Alineación vertical (verificación con etiquetas)\n"
"La vertical se fija sin etiquetas (PCA + normales). Aquí se **verifica a posteriori** que edificación y árbol queden por encima de vía y fondo.",
r'''ok = []
for s in stats:
    z = s["z_mean_per_class"]
    alto = [z[k] for k in (1, 2, 4) if z[k] is not None]
    bajo = [z[k] for k in (0, 3) if z[k] is not None]
    ok.append(np.mean(alto) > np.mean(bajo))
print(f"Nubes con orientación consistente: {sum(ok)} de {len(ok)}")
print("Excepciones:", [f"{s['area']}/{s['cloud']}" for s, o in zip(stats, ok) if not o])''',
"MD:## 1.4 Armonización de escala\nFactores estimados solo con áreas de entrenamiento/validación (el Área 2 de prueba no se usa).",
r'''escala = json.loads((ROOT / "data" / "scale.json").read_text())
pd.DataFrame({"altura típica edificación (m)": escala["height"], "factor aplicado": escala["scale"]})''',
"MD:## 1.5 Distribución de clases por área y normales",
r'''from make_figures import distribution
distribution(str(DATA), ROOT / "entrega_final" / "figs")
from IPython.display import Image, display
display(Image(filename=str(ROOT / "entrega_final" / "figs" / "distribucion.png")))
d = np.load(DATA / "Area_2_pp6.npz")
print("Campos de cada nube procesada:", list(d.keys()), "| puntos:", len(d["xyz"]),
      "| componente vertical media de la normal:", round(float(d["normal"][:, 2].mean()), 3))''',
], "01_datos_y_preprocesamiento.ipynb")

# --------------------------------------------------------------------------------------------
nb([
"MD:# 2. Modelos y entrenamiento\n"
"Se comparan cuatro modelos con **el mismo protocolo** (bloques de 12 m × 12 m, entrenamiento: Áreas 1 y 3–7, validación: Área 8, prueba: Área 2):\n\n"
"| Modelo | Script |\n|---|---|\n| PointNet++ (SSG) | `code/train.py` |\n| PTv3 desde cero | `code/train_ptv3.py --model ptv3` |\n"
"| PTv3 + Sonata (preentrenado) | `code/train_ptv3.py --model sonata_small` |\n| PTv3 + Utonia (preentrenado) | `code/train_ptv3.py --model utonia_tiny` |\n\n"
"Cada entrenamiento tarda entre 45 min y 2 h en una RTX 4060 Ti; por eso se ejecutaron con `code/run_all.sh` y aquí se cargan sus registros. "
"Para reentrenar, cambie `ENTRENAR = True`.",
SETUP,
"MD:## 2.1 Arquitecturas y número de parámetros",
r'''import torch
from pointnet2 import PointNet2Seg
from train_ptv3 import Segmentor
filas = [dict(Modelo="PointNet++", Parámetros=sum(p.numel() for p in PointNet2Seg(7, 5).parameters()))]
for nombre, preset in [("PTv3 desde cero", "ptv3"), ("PTv3 + Sonata", "sonata_small"), ("PTv3 + Utonia", "utonia_tiny")]:
    m = Segmentor(preset, 5)
    filas.append(dict(Modelo=nombre, Parámetros=sum(p.numel() for p in m.parameters())))
t = pd.DataFrame(filas); t["Millones"] = (t["Parámetros"] / 1e6).round(1); t''',
"MD:## 2.2 Comandos de entrenamiento (3 semillas por modelo, según presupuesto)",
r'''ENTRENAR = False
cmd_pn2 = "python train.py --data ../data/processed --out ../runs/{run} --block 12 --epochs 40 --iters 400 --bs 16 --val_every 4 --seed {seed}"
cmd_ptv3 = "python train_ptv3.py --data ../data/processed --out ../runs/{run} --model {model} --seed {seed} --epochs 30 --iters 400 --bs 4 --val_every 3"
plan = [("pn2", None, 0), ("pn2_s1", None, 1), ("pn2_s2", None, 2), ("ptv3_s0", "ptv3", 0)] + \
       [(f"sonata_s{s}", "sonata_small", s) for s in range(3)] + [(f"utonia_tiny_s{s}", "utonia_tiny", s) for s in range(3)]
for run, model, seed in plan:
    c = cmd_pn2.format(run=run, seed=seed) if model is None else cmd_ptv3.format(run=run, model=model, seed=seed)
    print(c)
    if ENTRENAR:
        import subprocess
        subprocess.run([sys.executable] + c.split()[1:], cwd=ROOT / "code", check=True)''',
"MD:## 2.3 Registros de entrenamiento: mejor mIoU de validación y tiempo",
r'''NOMBRE = {"pn2": "PointNet++", "ptv3": "PTv3 desde cero", "sonata": "PTv3 + Sonata", "utonia": "PTv3 + Utonia"}
filas = []
for run, _, seed in plan:
    r = json.loads((RUNS / run / "results.json").read_text())
    log = json.loads((RUNS / run / "log.json").read_text())
    mejor = max((e for e in log if "val_mIoU" in e), key=lambda e: e["val_mIoU"])
    filas.append(dict(Corrida=run, Modelo=NOMBRE[run.split("_")[0]], Semilla=seed, Épocas=len(log),
                      **{"Mejor mIoU val.": r["best_val_mIoU"], "Época del mejor": mejor["epoch"],
                         "Tiempo (min)": round(r["train_time_min"])}))
pd.DataFrame(filas)''',
"MD:## 2.4 Curvas de entrenamiento (mIoU de validación, Área 8)",
r'''fig, ax = plt.subplots(1, 4, figsize=(16, 3.2), sharey=True)
for i, pref in enumerate(["pn2", "ptv3", "sonata", "utonia"]):
    for run, _, seed in plan:
        if run.split("_")[0] != pref:
            continue
        log = json.loads((RUNS / run / "log.json").read_text())
        v = [(e["epoch"], e["val_mIoU"]) for e in log if "val_mIoU" in e]
        ax[i].plot(*zip(*v), "o-", ms=3, label=f"semilla {seed}")
    ax[i].set_title(NOMBRE[pref]); ax[i].set_xlabel("época"); ax[i].legend(fontsize=8)
ax[0].set_ylabel("mIoU validación"); plt.tight_layout(); plt.show()''',
"MD:Los modelos preentrenados alcanzan su mejor validación en pocas épocas y luego se sobreajustan; "
"PTv3 desde cero aprende más despacio. Por eso se guarda el checkpoint con mejor mIoU de validación (`best.pt`).",
], "02_modelos_y_entrenamiento.ipynb")

# --------------------------------------------------------------------------------------------
nb([
"MD:# 3. Evaluación y resultados\n"
"Este notebook (1) **vuelve a ejecutar un modelo entrenado** sobre el área de prueba para comprobar que reproduce las métricas guardadas, y "
"(2) genera **todas las tablas de resultados del informe** a partir de las predicciones y métricas de cada corrida en `runs/`.\n\n"
"Métricas: IoU por clase, mIoU, exactitud global (OA), exactitud media por clase (mAcc) y F1, sobre el Área 2 "
"(a la resolución submuestreada de 0,1 m y a la resolución original de 30,9 millones de puntos).",
SETUP,
"MD:## 3.1 Ejecución del modelo entrenado (PTv3 + Utonia, semilla 0) sobre el Área 2\n"
"Se carga `runs/utonia_tiny_s0/best.pt` y se segmentan las 8 nubes de prueba (sin TTA). El resultado debe coincidir con el guardado al final del entrenamiento "
"(diferencias en la 4.ª cifra decimal pueden deberse al no determinismo de la GPU).",
r'''import torch
from argparse import Namespace
from train_ptv3 import Segmentor, load_clouds, evaluate
from train import metrics
run = RUNS / "utonia_tiny_s0"
args = Namespace(**json.loads((run / "results.json").read_text())["args"]); args.eval_bs = 2
modelo = Segmentor(args.model, 5).cuda()
modelo.load_state_dict(torch.load(run / "best.pt", map_location="cuda"))
prueba = load_clouds(str(DATA), {"Area_2"})
m, cm, preds = evaluate(modelo, prueba, args, "cuda", keep_preds=True)
guardado = json.loads((run / "results.json").read_text())["test"]
pd.DataFrame({"ahora": {k: m[k] for k in ("mIoU", "OA", "mAcc", "mF1")},
              "guardado": {k: guardado[k] for k in ("mIoU", "OA", "mAcc", "mF1")}})''',
r'''pd.DataFrame({"IoU ahora": m["IoU"], "IoU guardado": guardado["IoU"]}, index=CLASES)''',
"MD:## 3.2 Resultados de todas las corridas\n"
"`aggregate.py` reúne las métricas de cada corrida (`results.json`, `eval_tta_full.json`, `test_tta_fullres.json`) y calcula media ± desviación entre semillas. "
"Es exactamente el procedimiento que genera las tablas y figuras del informe.",
r'''import aggregate
aggregate.main()
S = json.loads((RUNS / "summary.json").read_text())
MODELOS = ["PointNet++", "PTv3 desde cero", "PTv3 + Sonata", "PTv3 + Utonia"]
ms = lambda o: f"{o['mean']:.3f} ± {o['std']:.3f}" if o["n"] > 1 else f"{o['mean']:.3f}"
tabla = pd.DataFrame([{"Modelo": m, "Semillas": len(S[m]["runs"]),
    "mIoU vóxel": ms(S[m]["vox"]["mIoU"]), "mIoU vóxel TTA": ms(S[m]["vox_tta"]["mIoU"]),
    "mIoU original": ms(S[m]["full_tta"]["mIoU"]), "OA original": ms(S[m]["full_tta"]["OA"]),
    "mAcc original": ms(S[m]["full_tta"]["mAcc"]), "IoU dañada": ms(S[m]["full_tta"]["IoU"][1])} for m in MODELOS])
print("Tabla principal del informe (Área 2):"); tabla''',
"MD:### IoU por clase (resolución original, con TTA, media de semillas)",
r'''pd.DataFrame({m: [S[m]["full_tta"]["IoU"][c]["mean"] for c in range(5)] + [S[m]["full_tta"]["mIoU"]["mean"]]
              for m in MODELOS}, index=CLASES + ["mIoU"]).T.round(3)''',
"MD:### Variabilidad entre semillas (mIoU de validación y de prueba)",
r'''pd.DataFrame([{"Modelo": m, "mIoU validación (s0/s1/s2)": " / ".join(f"{v:.3f}" for v in S[m]["best_val"]["values"]),
               "mIoU prueba (s0/s1/s2)": " / ".join(f"{v:.3f}" for v in S[m]["full_tta"]["mIoU"]["values"])}
              for m in MODELOS if len(S[m]["runs"]) > 1])''',
"MD:## 3.3 Comparación con el benchmark publicado (Le et al., 2026)",
r'''bench = {"PTv1": ([0.385, 0.511, 0.154, 0.300, 0.191], 0.308, 0.461), "PTv2": ([0.501, 0.002, 0.253, 0.486, 0.037], 0.256, 0.394),
         "PTv3": ([0.773, 0.761, 0.311, 0.047, 0.400], 0.458, 0.551), "FPT": ([0.41, 0.14, 0.03, 0.00, 0.00], 0.11, 0.22),
         "OA-CNNs": ([0.228, 0.582, 0.196, 0.000, 0.296], 0.260, 0.440)}
filas = [dict(Método=k + " (artículo)", **dict(zip(CLASES, iou)), mIoU=mi, mAcc=ma) for k, (iou, mi, ma) in bench.items()]
filas += [dict(Método=m + " (este trabajo)", **{c: S[m]["full_tta"]["IoU"][i]["mean"] for i, c in enumerate(CLASES)},
               mIoU=S[m]["full_tta"]["mIoU"]["mean"], mAcc=S[m]["full_tta"]["mAcc"]["mean"]) for m in MODELOS]
pd.DataFrame(filas).round(3)''',
"MD:## 3.4 Exactitud por nube (semilla 0): PointNet++ vs. PTv3 + Utonia",
r'''P1 = np.load(RUNS / "pn2" / "eval_tta_full_preds.npz"); P2 = np.load(RUNS / "utonia_tiny_s0" / "test_tta_preds.npz")
filas = []
for k in sorted(P1.files):
    lab = np.load(DATA / f"{k}.npz")["label"]
    a, b = (P1[k] == lab).mean(), (P2[k] == lab).mean()
    filas.append(dict(Nube=k.replace("Area_2_", ""), **{"PointNet++": a, "PTv3 + Utonia": b, "Diferencia": b - a}))
pd.DataFrame(filas).round(2)''',
"MD:## 3.5 Figuras del informe",
r'''from IPython.display import Image, display
for f in ["iou_modelos.png", "curvas_val.png", "cualitativo_final.png", "confusion_mejor.png"]:
    display(Image(filename=str(ROOT / "entrega_final" / "figs" / f)))''',
], "03_evaluacion_y_resultados.ipynb")
