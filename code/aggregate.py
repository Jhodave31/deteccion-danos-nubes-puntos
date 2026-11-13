"""Reúne los resultados de todos los experimentos de runs/ en tablas y figuras.

Salida: runs/summary.json, entrega_final/figs/{iou_modelos,curvas_val,confusion_mejor,cualitativo_final}.pdf/.png

Uso: python aggregate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "processed"
FIGS = ROOT / "entrega_final" / "figs"
SHORT = ["Fondo", "Edif. dañada", "Edif. sin daño", "Vía", "Árbol"]
COLORS = np.array([(135, 135, 135), (230, 25, 75), (70, 240, 240), (255, 225, 25), (0, 128, 0)]) / 255.0
MODELS = {  # nombre en el documento -> prefijo de carpeta en runs/
    "PointNet++": "pn2",
    "PTv3 desde cero": "ptv3",
    "PTv3 + Sonata": "sonata",
    "PTv3 + Utonia": "utonia_tiny",
}
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def run_dirs(prefix):
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        if prefix == "pn2" and (d.name == "pn2" or d.name.startswith("pn2_s")):
            out.append(d)
        elif prefix != "pn2" and d.name.startswith(prefix + "_s"):
            out.append(d)
    return [d for d in out if (d / "results.json").exists()]


def collect_run(d, kind):
    """Métricas de un run: sin TTA (vóxel), con TTA (vóxel) y con TTA a resolución original."""
    res = load(d / "results.json")
    r = dict(run=d.name, params=res.get("params"), best_val=res.get("best_val_mIoU"), time_min=res.get("train_time_min"))
    if kind == "pn2":
        ev = load(d / "eval_tta_full.json")
        r["vox"] = res.get("test")
        r["vox_tta"] = ev["voxel"] if ev else None
        r["full_tta"] = ev.get("full") if ev else None
        r["conf_full"] = ev.get("confusion_full") if ev else None
        r["preds"] = d / "eval_tta_full_preds.npz"
    else:
        full = load(d / "test_tta_fullres.json")
        r["vox"] = res.get("test")
        r["vox_tta"] = res.get("test_tta")
        r["full_tta"] = full["full"] if full else None
        r["conf_full"] = full.get("confusion_full") if full else None
        r["preds"] = d / "test_tta_preds.npz"
    log = load(d / "log.json") or []
    r["curve"] = [(e["epoch"], e["val_mIoU"]) for e in log if "val_mIoU" in e]
    return r


def agg(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)), values=v.tolist())


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, prefix in MODELS.items():
        runs = [collect_run(d, "pn2" if prefix == "pn2" else "ptv3") for d in run_dirs(prefix)]
        if not runs:
            continue
        m = dict(runs=[r["run"] for r in runs], params=runs[0]["params"],
                 time_min=agg([r["time_min"] for r in runs]), best_val=agg([r["best_val"] for r in runs]))
        for key in ("vox", "vox_tta", "full_tta"):
            ok = [r[key] for r in runs if r[key]]
            if not ok:
                continue
            m[key] = {k: agg([o[k] for o in ok]) for k in ("mIoU", "OA", "mAcc", "mF1")}
            m[key]["IoU"] = [agg([o["IoU"][c] for o in ok]) for c in range(5)]
            m[key]["F1"] = [agg([o["F1"][c] for o in ok]) for c in range(5)]
        m["_runs"] = runs
        summary[name] = m

    # mejor modelo: mayor mIoU medio de VALIDACIÓN (no de prueba)
    best_val = max(summary, key=lambda k: summary[k]["best_val"]["mean"])
    # mejor en prueba (descriptivo, para figuras y texto): mayor mIoU medio a resolución original con TTA
    best = max([k for k in summary if "full_tta" in summary[k]], key=lambda k: summary[k]["full_tta"]["mIoU"]["mean"])
    out = {k: {kk: vv for kk, vv in v.items() if kk != "_runs"} for k, v in summary.items()}
    out["_best_by_val"] = best_val
    out["_best_by_test"] = best
    (RUNS / "summary.json").write_text(json.dumps(out, indent=1, default=str))

    # Figura 1: IoU por clase de cada modelo (TTA, resolución original), media +- desv.
    key = "full_tta"
    names = [n for n in summary if key in summary[n]]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    w = 0.8 / max(len(names), 1)
    pal = ["#7f8c8d", "#2e86c1", "#c0392b", "#8e44ad"]
    for i, n in enumerate(names):
        mu = [summary[n][key]["IoU"][c]["mean"] for c in range(5)] + [summary[n][key]["mIoU"]["mean"]]
        sd = [summary[n][key]["IoU"][c]["std"] for c in range(5)] + [summary[n][key]["mIoU"]["std"]]
        ax.bar(np.arange(6) + (i - (len(names) - 1) / 2) * w, mu, w, yerr=sd, capsize=2, label=n, color=pal[i % 4])
    ax.set_xticks(range(6)); ax.set_xticklabels(SHORT + ["mIoU"])
    ax.set_ylabel("IoU (Área 2)"); ax.set_ylim(0, 1); ax.legend(frameon=False, ncol=len(names), fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / "iou_modelos.pdf"); fig.savefig(FIGS / "iou_modelos.png", dpi=130); plt.close(fig)

    # Figura 2: mIoU de validación por época (semilla 0 de cada modelo)
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for i, n in enumerate(summary):
        r0 = summary[n]["_runs"][0]
        if r0["curve"]:
            e, v = zip(*r0["curve"])
            ep_max = max(e)
            ax.plot(np.array(e) / ep_max, v, "o-", ms=3, label=n, color=pal[i % 4])
    ax.set_xlabel("Fracción del entrenamiento"); ax.set_ylabel("mIoU validación (Área 8)"); ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, ncol=4, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "curvas_val.pdf"); fig.savefig(FIGS / "curvas_val.png", dpi=130); plt.close(fig)

    # Figura 3: matriz de confusión del mejor modelo en prueba (semilla 0, TTA, resolución original)
    rb = summary[best]["_runs"][0]
    if rb["conf_full"]:
        cm = np.array(rb["conf_full"], float)
        cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks(range(5)); ax.set_xticklabels(SHORT, rotation=40, ha="right")
        ax.set_yticks(range(5)); ax.set_yticklabels(SHORT)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(FIGS / "confusion_mejor.pdf"); fig.savefig(FIGS / "confusion_mejor.png", dpi=130); plt.close(fig)

    # Figura 4: cualitativo: RGB / real / PointNet++ / mejor modelo
    pn = summary["PointNet++"]["_runs"][0]["preds"]
    bp = rb["preds"]
    if pn.exists() and bp.exists():
        P1, P2 = np.load(pn), np.load(bp)
        clouds = ["Area_2_pp4", "Area_2_pp6"]
        fig, axes = plt.subplots(len(clouds), 4, figsize=(7.2, 1.3 * len(clouds) + 0.4))
        for r, c in enumerate(clouds):
            d = np.load(DATA / f"{c}.npz")
            xyz, gt = d["xyz"], d["label"]
            idx = np.random.default_rng(0).permutation(len(xyz))[:250_000]
            idx = idx[np.argsort(xyz[idx, 2])]
            panels = [(d["rgb"][idx] / 255.0, f"{c.replace('Area_', 'Área ').replace('_', ' ')}: RGB"), (COLORS[gt[idx]], "Etiqueta real"),
                      (COLORS[P1[c][idx]], f"PointNet++ (acierto {(P1[c] == gt).mean():.2f})"),
                      (COLORS[P2[c][idx]], f"{best} (acierto {(P2[c] == gt).mean():.2f})")]
            for k, (col, t) in enumerate(panels):
                ax = axes[r, k]
                ax.scatter(xyz[idx, 0], xyz[idx, 1], c=col, s=0.15, linewidths=0, rasterized=True)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=7.5)
                for sp in ax.spines.values():
                    sp.set_visible(False)
        handles = [plt.Line2D([], [], marker="s", ls="", color=COLORS[i], mec="gray", label=SHORT[i]) for i in range(5)]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(FIGS / "cualitativo_final.pdf", dpi=220)
        fig.savefig(FIGS / "cualitativo_final.png", dpi=130); plt.close(fig)

    # Resumen legible
    for n, m in summary.items():
        line = [f"{n:16s} semillas={len(m['runs'])} val={m['best_val']['mean']:.3f}"]
        for k in ("vox", "vox_tta", "full_tta"):
            if k in m:
                line.append(f"{k}: mIoU {m[k]['mIoU']['mean']:.3f}±{m[k]['mIoU']['std']:.3f}")
        print(" | ".join(line))
    print("mejor por validación:", best_val, "| mejor en prueba:", best)


if __name__ == "__main__":
    main()
