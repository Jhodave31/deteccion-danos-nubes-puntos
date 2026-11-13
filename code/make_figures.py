"""Genera las figuras del documento a partir de los resultados del entrenamiento.

Uso: python make_figures.py --data data/processed --run runs/pn2 --out entrega2/figs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["Background", "Building-Damage", "Building-No-Damage", "Road", "Tree"]
SHORT = ["Fondo", "Edif. dañada", "Edif. sin daño", "Vía", "Árbol"]
COLORS = np.array([(135, 135, 135), (230, 25, 75), (70, 240, 240), (255, 225, 25), (0, 128, 0)]) / 255.0
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})


def curves(run, out):
    log = json.loads((run / "log.json").read_text())
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    ax[0].plot(ep, [r["loss"] for r in log], color="#1f4e79")
    ax[0].set_xlabel("Época"); ax[0].set_ylabel("Pérdida de entrenamiento")
    v = [r for r in log if "val_mIoU" in r]
    ax[1].plot(ep, [r["train_acc"] for r in log], label="Exactitud entrenamiento (bloques)", color="#1f4e79")
    ax[1].plot([r["epoch"] for r in v], [r["val_mIoU"] for r in v], "o-", label="mIoU validación (Área 8)", color="#c0392b", ms=3)
    ax[1].plot([r["epoch"] for r in v], [r["val_OA"] for r in v], "s--", label="OA validación (Área 8)", color="#7f8c8d", ms=3)
    ax[1].set_xlabel("Época"); ax[1].set_ylim(0, 1); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "curvas.pdf"); fig.savefig(out / "curvas.png", dpi=110); plt.close(fig)


def confusion_fig(res, out):
    cm = np.array(res["confusion"], dtype=np.float64)
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
    fig.tight_layout(); fig.savefig(out / "confusion.pdf"); fig.savefig(out / "confusion.png", dpi=110); plt.close(fig)


def topdown(ax, xyz, colors, title, maxpts=250_000):
    idx = np.random.default_rng(0).permutation(len(xyz))[:maxpts]
    idx = idx[np.argsort(xyz[idx, 2])]  # pintar de abajo hacia arriba
    ax.scatter(xyz[idx, 0], xyz[idx, 1], c=colors[idx], s=0.15, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def qualitative(data, run, out, names):
    preds = np.load(run / "test_preds.npz")
    fig, axes = plt.subplots(len(names), 4, figsize=(7.2, 1.25 * len(names) + 0.4))
    axes = np.atleast_2d(axes)
    for r, n in enumerate(names):
        d = np.load(Path(data) / f"{n}.npz")
        xyz, rgb, gt = d["xyz"], d["rgb"] / 255.0, d["label"]
        pr = preds[n]
        err = np.where((pr == gt)[:, None], [[0.85, 0.85, 0.85]], [[0.85, 0.1, 0.1]])
        acc = (pr == gt).mean()
        topdown(axes[r, 0], xyz, rgb, f"{n.replace('_', ' ')}: RGB")
        topdown(axes[r, 1], xyz, COLORS[gt], "Etiqueta real")
        topdown(axes[r, 2], xyz, COLORS[pr], "Predicción PointNet++")
        topdown(axes[r, 3], xyz, err, f"Errores (exactitud {acc:.2f})")
    handles = [plt.Line2D([], [], marker="s", ls="", color=COLORS[i], mec="gray", label=SHORT[i]) for i in range(5)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "cualitativo.pdf", dpi=220); fig.savefig(out / "cualitativo.png", dpi=110); plt.close(fig)


def distribution(data, out):
    stats = json.loads((Path(data) / "stats.json").read_text())
    areas = sorted({s["area"] for s in stats}, key=lambda a: int(a.split("_")[1]))
    M = np.array([np.sum([s["counts"] for s in stats if s["area"] == a], 0) for a in areas], dtype=np.float64)
    P = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 2.4))
    left = np.zeros(len(areas))
    for c in range(5):
        ax.barh(range(len(areas)), P[:, c], left=left, color=COLORS[c], edgecolor="gray", lw=0.3, label=SHORT[c])
        left += P[:, c]
    ax.set_yticks(range(len(areas))); ax.set_yticklabels([a.replace("Area_", "Área ") + (" (prueba)" if a == "Area_2" else " (val.)" if a == "Area_8" else "") for a in areas])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Proporción de puntos")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "distribucion.pdf"); fig.savefig(out / "distribucion.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clouds", nargs="*", default=None)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads((run / "results.json").read_text())
    curves(run, out)
    confusion_fig(res, out)
    distribution(a.data, out)
    names = a.clouds or sorted(np.load(run / "test_preds.npz").files)[:2]
    qualitative(a.data, run, out, names)
    print("figuras en", out)


if __name__ == "__main__":
    main()
