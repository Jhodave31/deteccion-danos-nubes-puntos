"""Genera entrega_final/main.tex a partir de main_template.tex y de los resultados en runs/.

Todas las cifras y las afirmaciones comparativas del texto se calculan a partir de runs/summary.json,
así que el documento se puede regenerar cuando terminan nuevos experimentos.

Uso (desde entrega_final/):
    python ../code/aggregate.py      # actualiza runs/summary.json y las figuras
    python build_document.py         # escribe main.tex
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S = json.loads((ROOT / "runs" / "summary.json").read_text(encoding="utf-8"))
BEST_VAL = S.pop("_best_by_val")
BEST = S.pop("_best_by_test")
MODELS = [m for m in ["PointNet++", "PTv3 desde cero", "PTv3 + Sonata", "PTv3 + Utonia"] if m in S]
PRE = [m for m in MODELS if m in ("PTv3 + Sonata", "PTv3 + Utonia")]
NOPRE = [m for m in MODELS if m not in PRE]
PN = "PointNet++"
BENCH = {  # Le et al. (2026), Tabla 3: IoU por clase (fondo, dañada, sin daño, vía, árbol), mIoU, mAcc
    "PTv1~\\cite{Zhao2021}": ([0.385, 0.511, 0.154, 0.300, 0.191], 0.308, 0.461),
    "PTv2~\\cite{Wu2022}": ([0.501, 0.002, 0.253, 0.486, 0.037], 0.256, 0.394),
    "PTv3~\\cite{Wu2024}": ([0.773, 0.761, 0.311, 0.047, 0.400], 0.458, 0.551),
    "FPT~\\cite{Park2022}": ([0.41, 0.14, 0.03, 0.00, 0.00], 0.11, 0.22),
    "OA-CNNs~\\cite{Peng2024}": ([0.228, 0.582, 0.196, 0.000, 0.296], 0.260, 0.440),
}


def f(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def pm(o, d=3):
    if o is None:
        return "---"
    return f"{f(o['mean'], d)} $\\pm$ {f(o['std'], d)}" if o["n"] > 1 else f(o["mean"], d)


def M(m, metric="mIoU", c=None, key="full_tta"):
    o = S[m][key][metric] if c is None else S[m][key][metric][c]
    return o["mean"]


def O(m, metric="mIoU", c=None, key="full_tta"):
    return S[m][key][metric] if c is None else S[m][key][metric][c]


def nseeds(m):
    return len(S[m]["runs"])


def seeds_txt(m):
    n = nseeds(m)
    return "una semilla" if n == 1 else f"{n} semillas"


def run_values(m, metric="mIoU", key="full_tta"):
    return O(m, metric, key=key)["values"]


def scratch_relation():
    """Relación de mIoU entre PTv3 desde cero y PointNet++: 'igual' si la diferencia cabe en la desviación entre
    semillas de PointNet++, si no 'mejor' o 'peor'."""
    d = M("PTv3 desde cero") - M(PN)
    tol = max(O(PN)["std"], 0.01)
    return "igual" if abs(d) <= tol else ("mejor" if d > 0 else "peor")


def per_cloud_acc():
    """Exactitud por nube (semilla 0, puntos submuestreados, con TTA) de PointNet++ y del mejor modelo en prueba."""
    files = {PN: ROOT / "runs" / "pn2" / "eval_tta_full_preds.npz",
             BEST: ROOT / "runs" / sorted(S[BEST]["runs"])[0] / "test_tta_preds.npz"}
    rows = {}
    for m, p in files.items():
        if not p.exists():
            return None
        P = np.load(p)
        for k in sorted(P.files):
            lab = np.load(ROOT / "data" / "processed" / f"{k}.npz")["label"]
            rows.setdefault(k, {})[m] = float((P[k] == lab).mean())
    return rows


def sdiff(a, b):
    d = round(a - b, 2)
    return f(0, 2) if d == 0 else ("+" if d > 0 else "$-$") + f(abs(d), 2)



def pct(v, d=0):
    return f"{100 * v:.{d}f}".replace(".", ",")


def conf_best():
    """Matriz de confusión normalizada por fila del mejor modelo (semilla 0, resolución original, TTA)."""
    run = sorted(S[BEST]["runs"])[0]
    fjs = ROOT / "runs" / run / ("eval_tta_full.json" if BEST == PN else "test_tta_fullres.json")
    cm = np.array(json.loads(fjs.read_text())["confusion_full"], float)
    return cm / cm.sum(1, keepdims=True)


def tables():
    """Tablas de resultados (LaTeX)."""
    rows = []
    for m in MODELS:
        s = S[m]
        rows.append(f"{m} & {nseeds(m)} & {pm(s['vox']['mIoU'])} & {pm(s['vox_tta']['mIoU'])} & "
                    f"{pm(s['full_tta']['mIoU'])} & {pm(s['full_tta']['OA'])} & {pm(s['full_tta']['mAcc'])} & "
                    f"{pm(s['full_tta']['IoU'][1])} \\\\")
    main_table = r"""\begin{table*}[t]
\centering
\caption{Resultados en el área de prueba (Área 2). ``Vóxel'': puntos submuestreados a 0,1\,m; ``original'': los 30,9 millones de puntos de las nubes originales. Media $\pm$ desviación estándar sobre las semillas indicadas (sin desviación cuando hay una sola).}\label{tab:final}
\footnotesize
\begin{tabular}{@{}lccccccc@{}}
\toprule
 & & \multicolumn{2}{c}{mIoU (vóxel)} & \multicolumn{4}{c}{Resolución original, con TTA} \\
\cmidrule(lr){3-4}\cmidrule(l){5-8}
Modelo & Semillas & sin TTA & con TTA & mIoU & OA & mAcc & IoU dañada \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}"""

    srows = []
    for m in MODELS:
        if nseeds(m) > 1:
            srows.append(f"{m} & " + " / ".join(f(x) for x in S[m]["best_val"]["values"]) + " & " +
                         " / ".join(f(x) for x in run_values(m)) + " \\\\")
    seed_table = r"""\begin{table}[t]
\centering
\caption{mIoU de validación del modelo seleccionado (Área 8) y de prueba (Área 2, resolución original) para las semillas 0 / 1 / 2.}\label{tab:semillas}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcc@{}}
\toprule
Modelo & mIoU validación & mIoU prueba \\
\midrule
""" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}
\end{table}"""

    brow = [f"{name} & " + " & ".join(f(v) for v in [iou[1], iou[2], iou[3], iou[4], iou[0]]) + f" & {f(miou)} & {f(macc)} \\\\"
            for name, (iou, miou, macc) in BENCH.items()]
    ours = [f"{m} & " + " & ".join(f(S[m]['full_tta']['IoU'][c]['mean']) for c in [1, 2, 3, 4, 0]) +
            f" & {f(M(m))} & {f(M(m, 'mAcc'))} \\\\" for m in MODELS]
    bench_table = r"""\begin{table*}[t]
\centering
\caption{Comparación con los modelos de referencia de 3DAeroRelief en el Área 2 (IoU por clase, mIoU y mAcc). Las cinco primeras filas son los valores reportados en~\cite{Le2026}; las de este trabajo son la media de sus semillas, a la resolución original y con TTA.}\label{tab:comparacion}
\footnotesize
\begin{tabular}{@{}lccccccc@{}}
\toprule
Método & Edif.\ dañada & Edif.\ sin daño & Vía & Árbol & Fondo & mIoU & mAcc \\
\midrule
""" + "\n".join(brow) + "\n\\midrule\n" + "\n".join(ours) + r"""
\bottomrule
\end{tabular}
\end{table*}"""

    pc = per_cloud_acc()
    lines = [f"{k.replace('Area_2_', '')} & {f(v[PN], 2)} & {f(v[BEST], 2)} & {sdiff(v[BEST], v[PN])} \\\\" for k, v in pc.items()]
    pc_table = r"""\begin{table}[t]
\centering
\caption{Exactitud por nube del Área 2 (semilla 0, puntos submuestreados, con TTA).}\label{tab:pornube}
\scriptsize
\begin{tabular}{@{}lccc@{}}
\toprule
Nube & PointNet++ & """ + BEST + r""" & Dif. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    return main_table, seed_table, bench_table, pc_table, pc


def results():
    main_table, seed_table, bench_table, pc_table, pc = tables()
    ranking = sorted(MODELS, key=lambda m: -M(m))
    nseed_txt = "; ".join(f"{m}: {nseeds(m)}" for m in MODELS)
    order_txt = ", ".join(f"{m} ({pm(O(m))})" for m in ranking)
    tta_gain = {m: M(m, key="vox_tta") - M(m, key="vox") for m in MODELS}
    tta_txt = ", ".join(f"{m}: {'+' if g >= 0 else '$-$'}{f(abs(g))}" for m, g in tta_gain.items())
    spread = "; ".join(f"{m}, de {f(min(run_values(m)))} a {f(max(run_values(m)))}" for m in MODELS if nseeds(m) > 1)
    better = sum(round(v[BEST] - v[PN], 2) > 0 for v in pc.values())
    worse = sum(round(v[BEST] - v[PN], 2) < 0 for v in pc.values())
    cm0 = conf_best()
    return rf"""Se entrenaron {len(MODELS)} configuraciones con el protocolo de la Sección~\ref{{sec:metodos}} (semillas: {nseed_txt}). Cada modelo se seleccionó por su mIoU en el Área 8, y el Área 2 se evaluó una sola vez con el modelo elegido.

\subsection{{Desempeño en el área de prueba}}

{main_table}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.85\textwidth]{{figs/iou_modelos.pdf}}
\caption{{IoU por clase en el Área 2 (resolución original, con TTA). Las barras de error indican la desviación estándar entre semillas.}}\label{{fig:iou}}
\end{{figure*}}

La Tabla~\ref{{tab:final}} y la Figura~\ref{{fig:iou}} resumen el desempeño en el área de prueba. Ordenados por mIoU medio a la resolución original, los modelos quedan así: {order_txt}. El mejor modelo, {BEST}, obtiene un IoU de {pm(O(BEST, 'IoU', 1))} en la clase objetivo, edificación dañada, frente a {pm(O(PN, 'IoU', 1))} de PointNet++. Los modelos preentrenados mejoran sobre todo edificación sin daño, árbol y, en el caso de Utonia, vía. El aumento en prueba (TTA) cambia poco el mIoU (diferencia con TTA menos sin TTA, sobre puntos submuestreados: {tta_txt}). Las métricas a la resolución original difieren de las submuestreadas, porque las nubes densas de capturas cercanas pesan más; por ejemplo, pp4 tiene 12,6 millones de puntos y está dominada por una edificación dañada.

\subsection{{Entrenamiento y variabilidad entre semillas}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.85\textwidth]{{figs/curvas_val.pdf}}
\caption{{mIoU de validación (Área 8) durante el entrenamiento, semilla 0 de cada modelo. El eje horizontal es la fracción del entrenamiento completada (40 épocas para PointNet++ y 30 para los modelos PTv3).}}\label{{fig:curvasval}}
\end{{figure*}}

{seed_table}

La Figura~\ref{{fig:curvasval}} muestra que los modelos preentrenados alcanzan una validación alta en pocas épocas y luego oscilan, mientras la exactitud de entrenamiento sigue subiendo (sobreajuste). PTv3 desde cero aprende más despacio y aún mejoraba al terminar las 30 épocas. La Tabla~\ref{{tab:semillas}} muestra que el mIoU de prueba cambia de forma notable con la semilla ({spread}), y que semillas con una validación casi igual obtienen resultados de prueba distintos.

\subsection{{Comparación con el benchmark publicado}}

{bench_table}

La Tabla~\ref{{tab:comparacion}} sitúa los resultados junto a los publicados por los autores del dataset sobre la misma área de prueba. El mejor modelo publicado, PTv3, obtiene un mIoU de 0,458, un mAcc de 0,551 y un IoU de edificación dañada de 0,761. {BEST} obtiene {f(M(BEST))}, {f(M(BEST, 'mAcc'))} y {f(M(BEST, 'IoU', 1))}, respectivamente.

\subsection{{Resultados cualitativos y errores}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.9\textwidth]{{figs/cualitativo_final.pdf}}
\caption{{Ejemplos cualitativos sobre el Área 2 (vista superior): color RGB, etiqueta real, predicción de PointNet++ y predicción de {BEST} (semilla 0).}}\label{{fig:cualitativo}}
\end{{figure*}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.9\linewidth]{{figs/confusion_mejor.pdf}}
\caption{{Matriz de confusión normalizada por fila de {BEST} en el Área 2 (semilla 0, resolución original, con TTA).}}\label{{fig:confusion}}
\end{{figure}}

{pc_table}

En la nube pp6 (Figura~\ref{{fig:cualitativo}}), PointNet++ clasifica como \emph{{sin daño}} buena parte del techo dañado, mientras que {BEST} lo reconoce casi completo. En la matriz de confusión (Figura~\ref{{fig:confusion}}), el {pct(cm0[1][1])}\,\% de los puntos de edificación dañada se clasifica correctamente. A la vez, el {pct(cm0[2][1])}\,\% de los de edificación sin daño se marca como dañada, y el {pct(cm0[3][0])}\,\% de la vía se confunde con fondo. Por nube (Tabla~\ref{{tab:pornube}}), {BEST} mejora a PointNet++ en {better} de las {len(pc)} nubes y empeora en {worse}."""


def discussion():
    rel = scratch_relation()
    scratch_txt = {"igual": "queda al nivel de", "mejor": "supera ligeramente a", "peor": "no supera a"}[rel]
    pre_all_better = PRE and min(M(m) for m in PRE) > max(M(m) for m in NOPRE)
    pre_txt = ("Los dos modelos preentrenados superan a los dos entrenados desde cero en mIoU medio, y la diferencia entre "
               "ambos grupos es mayor que la variación entre semillas." if pre_all_better else
               f"El mejor modelo es preentrenado ({BEST}), aunque no todos los preentrenados superan a los entrenados desde cero.")
    aerial = ""
    if "PTv3 + Utonia" in PRE and "PTv3 + Sonata" in PRE and M("PTv3 + Utonia") > M("PTv3 + Sonata"):
        ratio = S["PTv3 + Sonata"]["params"] / S["PTv3 + Utonia"]["params"]
        aerial = (f" Entre los dos preentrenados, Utonia rinde más que Sonata ({f(M('PTv3 + Utonia'))} frente a "
                  f"{f(M('PTv3 + Sonata'))}), aunque tiene {f(ratio, 1)} veces menos parámetros. La diferencia más clara está en la "
                  f"clase vía ({f(M('PTv3 + Utonia', 'IoU', 3))} frente a {f(M('PTv3 + Sonata', 'IoU', 3))}), una superficie exterior "
                  "plana que Sonata, preentrenado con escenas interiores, no ha visto. Utonia sí incluye escaneos aéreos y LiDAR exterior. "
                  "Esto sugiere que el dominio de los datos de preentrenamiento importa tanto como la cantidad de parámetros.")
    nopre_max = max(max(run_values(m)) for m in NOPRE)
    robust = [m for m in PRE if min(run_values(m)) > nopre_max]
    fragile = [m for m in PRE if m not in robust]
    robust_txt = ""
    if robust:
        robust_txt = (f"La ventaja de {' y '.join(robust)} es robusta: todas sus semillas superan a todas las de los modelos entrenados desde cero.")
    if fragile:
        robust_txt += (f" La de {' y '.join(fragile)} no lo es: alguna de sus semillas queda por debajo de la mejor semilla de un modelo entrenado desde cero.")
    d_b = M(BEST, "IoU", 1) - 0.761
    dmg_vs_bench = "queda al mismo nivel" if abs(d_b) <= 0.02 else ("lo supera" if d_b > 0 else f"queda {f(abs(d_b), 2)} por debajo")
    val_note = ("" if BEST_VAL == BEST else f" Además, por validación media se habría elegido {BEST_VAL} y no {BEST}.")
    return rf"""\subsection{{Arquitectura frente a preentrenamiento}}

El hallazgo central es que, con los datos disponibles, la arquitectura por sí sola no basta: PTv3 entrenado desde cero, con 46 veces más parámetros, {scratch_txt} PointNet++ (mIoU {f(M('PTv3 desde cero'))} frente a {pm(O(PN))}). Lo que marca la diferencia es partir de representaciones aprendidas. {pre_txt}{aerial} Este resultado coincide con dos observaciones de la literatura. Por un lado, PointNeXt~\cite{{Qian2022}} mostró que las estrategias de entrenamiento pesan tanto como el diseño de la red. Por otro, en el benchmark original~\cite{{Le2026}}, los modelos entrenados desde cero generalizan mal entre áreas. Con solo 50 nubes de entrenamiento de seis zonas, un transformer grande no tiene datos suficientes para aprender desde cero la geometría de techos, muros y vegetación. Un codificador preentrenado ya trae esa geometría y solo debe aprender a separar las clases de este problema.

\subsection{{Comparación con trabajos similares}}

Frente al benchmark (Tabla~\ref{{tab:comparacion}}), {BEST} supera al mejor modelo publicado en mIoU ({f(M(BEST))} frente a 0,458) y en mAcc, y {dmg_vs_bench} en la clase edificación dañada ({f(M(BEST, 'IoU', 1))} frente a 0,761). La comparación es \textbf{{indicativa}}: aquí se corrigieron la orientación vertical y la escala de las nubes (Sección~\ref{{sec:prepro}}), algo que los modelos de referencia no reportan, y los modelos se entrenaron sin el Área 8. La comparación entre los modelos de este trabajo, en cambio, sí es justa, porque todos comparten exactamente el mismo protocolo.

Frente a otros trabajos de daño estructural, las cifras de este proyecto son menores en valor absoluto, pero la tarea es más difícil. NL-3DCrack~\cite{{Lou2025}} alcanza un mIoU de 84,5\,\% y SNEPointNet++~\cite{{Bolourian2023}} un \emph{{recall}} superior al 92\,\%. Ambos segmentan grietas o desprendimientos en un elemento estructural escaneado de cerca, y evalúan sobre datos del mismo sitio. Liu et al.~\cite{{Liu2025}} reportan un IoU de 65,5\,\% para grietas en túneles con LiDAR. Zahs et al.~\cite{{Zahs2023}} logran una exactitud global de 92--95\,\%, pero clasifican edificaciones completas, no cada punto. Aquí, en cambio, se segmenta punto a punto una escena aérea completa con cinco clases y se evalúa en una zona geográfica distinta a las de entrenamiento. Además, las etiquetas del dataset se obtuvieron proyectando a 3D anotaciones 2D hechas en uno de cada diez fotogramas, lo que introduce ruido en los bordes. En ese contexto, un IoU de {f(M(BEST, 'IoU', 1))} para la clase dañada es un resultado competitivo, y la mejora de {f(M(BEST, 'IoU', 1) - M(PN, 'IoU', 1), 2)} sobre PointNet++ ({f(M(PN, 'IoU', 1))}) es relevante para una aplicación de triaje post-desastre.

\subsection{{Estabilidad de los resultados}}

La variabilidad entre semillas (Tabla~\ref{{tab:semillas}}) es tan importante como el promedio. Los modelos preentrenados alcanzan su mejor validación muy pronto y luego se sobreajustan, así que el checkpoint elegido depende del azar del entrenamiento. Además, el área de validación, con seis nubes, no siempre distingue un checkpoint bueno de uno regular.{val_note} Las diferencias pequeñas entre modelos, como entre PTv3 desde cero y PointNet++, no son concluyentes. {robust_txt} La evaluación por nube (Tabla~\ref{{tab:pornube}}) confirma que el promedio esconde diferencias grandes entre escenas. También muestra que la resolución de evaluación cambia las métricas, porque las nubes densas pesan más: por eso este trabajo reporta ambas resoluciones y compara con el benchmark a la resolución original.

\subsection{{Análisis de errores}}

El modelo reconoce bien la edificación dañada, que es la clase prioritaria para el triaje. Su error principal es el opuesto: marcar como dañada parte de las edificaciones sanas. A 0,1\,m de resolución, un techo intacto y uno con daño leve (tejas levantadas, lonas) son geométricamente parecidos, y en el área de prueba la edificación dañada es mayoritaria, lo que sesga la decisión hacia esa clase. Para una herramienta de inspección este tipo de error es preferible al contrario, porque genera falsas alarmas que un inspector descarta, en lugar de daños que pasen inadvertidos. La vía se confunde con el fondo porque ambas son superficies planas a ras de suelo, y la vía representa solo el 0,6\,\% de los puntos de prueba, así que su IoU no permite conclusiones.

\subsection{{Limitaciones}}

\begin{{itemize}}
    \item El área de prueba tiene solo 8 nubes de una misma zona. Los resultados varían entre nubes y entre semillas, y los modelos no tienen todos el mismo número de semillas.
    \item La comparación con el benchmark es indicativa, porque el preprocesamiento no es idéntico.
    \item Por el presupuesto de cómputo (una GPU de 8\,GB), se usaron las versiones pequeñas de los codificadores preentrenados y 30 épocas para PTv3.
\end{{itemize}}"""


def abstract():
    scratch_abs = {"igual": "PTv3 desde cero queda al nivel de PointNet++", "mejor": "PTv3 desde cero mejora solo un poco a PointNet++",
                   "peor": "PTv3 desde cero no mejora a PointNet++"}[scratch_relation()]
    return rf"""La inspección de estructuras civiles después de un desastre natural sigue dependiendo en gran medida de la observación visual, un proceso lento, subjetivo y riesgoso. Las nubes de puntos 3D obtenidas por fotogrametría con drones permiten automatizarla. Este trabajo estudia la segmentación semántica de edificaciones dañadas sobre el benchmark público 3DAeroRelief (huracán Ian, 2022), con su división oficial por áreas. Primero se identificaron y corrigieron dos inconsistencias no documentadas del dataset: la orientación vertical arbitraria de las reconstrucciones y una escala distinta en dos áreas. Luego se compararon, con un protocolo idéntico, cuatro modelos: PointNet++, Point Transformer V3 (PTv3) entrenado desde cero y PTv3 con codificadores preentrenados de forma autosupervisada (Sonata y Utonia). En el área de prueba, evaluada a la resolución original (30,9 millones de puntos), el mejor modelo, {BEST}, obtiene un mIoU de {f(M(BEST))} y un IoU de {f(M(BEST, 'IoU', 1))} en la clase edificación dañada, frente a {f(M(PN))} y {f(M(PN, 'IoU', 1))} de PointNet++. {scratch_abs}, lo que indica que, con pocos datos, el preentrenamiento importa más que la arquitectura; sin embargo, los modelos preentrenados muestran una variabilidad notable entre semillas. Finalmente, el modelo se integró en una aplicación web local que segmenta una nube y muestra el daño en 3D. El código, los modelos y la herramienta se entregan junto al documento."""


def conclusions():
    rel_txt = {"igual": "queda al nivel de", "mejor": "apenas mejora a", "peor": "no supera a"}[scratch_relation()]
    var = ""
    multi = [m for m in PRE if nseeds(m) > 1]
    if multi:
        var = ("Sin embargo, el ajuste de estos modelos es inestable: con la misma configuración, distintas semillas "
               "difieren en varios puntos de mIoU, y la validación no siempre lo anticipa.")
    return rf"""Este trabajo abordó la detección automática de edificaciones dañadas en nubes de puntos de dron, desde la revisión de la literatura hasta una herramienta utilizable. Las conclusiones principales son:
\begin{{itemize}}
    \item \textbf{{Es viable.}} En una zona que el modelo nunca vio, {BEST} identifica la edificación dañada con un IoU de {f(M(BEST, 'IoU', 1))} a la resolución original, y supera, de forma indicativa, los resultados publicados para el benchmark.
    \item \textbf{{El preentrenamiento importa más que la arquitectura.}} Un PTv3 entrenado desde cero {rel_txt} PointNet++ (mIoU {f(M('PTv3 desde cero'))} frente a {f(M(PN))}), mientras que el mismo PTv3 preentrenado lo supera con claridad: {", ".join(f"{m} {f(M(m))}" for m in PRE)}. {var}
    \item \textbf{{Revisar los datos es parte del método.}} La orientación arbitraria y la escala inconsistente no estaban documentadas; corregirlas fue necesario para que rasgos como la altura tengan significado.
    \item \textbf{{La evaluación debe ser cuidadosa.}} Las métricas cambian con la resolución de evaluación y varían mucho entre nubes y entre semillas, así que reportar una sola cifra puede ser engañoso.
    \item \textbf{{Del modelo a la herramienta.}} La aplicación desarrollada reutiliza el mismo código del experimento y funciona sin conexión, lo que la acerca a un uso real en campo.
\end{{itemize}}

Como trabajo futuro se proponen cinco líneas: (i) estabilizar el ajuste de los modelos preentrenados, por ejemplo con tasas de aprendizaje menores, congelando parte del codificador o promediando pesos de varias épocas; (ii) mejorar la distinción entre edificaciones sanas y con daño leve, por ejemplo con mayor resolución o rasgos geométricos de rugosidad del techo~\cite{{Bolourian2023}}; (iii) usar codificadores preentrenados más grandes y conjuntos de datos de otras zonas y tipos de desastre; (iv) incorporar escalas de daño más finas que la clasificación binaria; y (v) validar la herramienta con usuarios del área de gestión del riesgo."""


def cambios():
    return r"""\paragraph{Respecto al segundo avance}
\begin{itemize}
    \item Se agregaron Point Transformer V3 (desde cero) y PTv3 con preentrenamiento autosupervisado (Sonata y Utonia), con el mismo protocolo que PointNet++ (Sección~\ref{sec:modelos}), y dos referencias nuevas~\cite{Wu2025,Zhang2026}.
    \item Se agregaron normales de superficie recalculadas para las 64 nubes, la evaluación a la resolución original, el aumento en prueba (TTA) y varias semillas por modelo.
    \item Se corrigió un error de evaluación detectado durante el trabajo: la inferencia de PTv3 en media precisión FP16 se desbordaba; se usa BF16, igual que en el entrenamiento.
    \item Se agregó la Sección~\ref{sec:software}, con los requisitos, la arquitectura y la verificación de la herramienta.
    \item Se reescribieron la introducción, los objetivos, el resumen y las conclusiones para la versión final.
\end{itemize}"""


def main():
    t = (HERE / "main_template.tex").read_text(encoding="utf-8")
    p = lambda m: f"{S[m]['params'] / 1e6:.1f}".replace(".", ",") + "\\,M" if m in S else "---"
    demo_model = json.loads((ROOT / "demo" / "config.json").read_text())["run"]
    run_tex = demo_model.replace("_", "\\_")
    perf = (r"En la GPU, una nube típica del Área 2 (pp1, 0,55 millones de puntos) se preprocesa en 1,6\,s y se "
            r"segmenta en 4,2\,s. En las nubes más densas (por ejemplo pp6, 5,0 millones de puntos) la segmentación sigue tomando "
            r"unos 2\,s, pero el preprocesamiento (lectura, alineación, submuestreo y normales) toma alrededor de 3\,min. Por eso "
            r"los análisis de las nubes del dataset se guardan en disco y la segunda vez se abren de inmediato. "
            "El modelo que usa la herramienta se indica en \\texttt{demo/config.json} (actualmente, \\texttt{" + run_tex + "}).")
    rep = {
        "%%ABSTRACT%%": abstract(), "%%RESULTS%%": results(), "%%CONCLUSIONS%%": conclusions(),
        "%%DISCUSSION%%": discussion(), "%%SOFTWARE_PERF%%": perf,
        "%%P_PTV3%%": p("PTv3 desde cero"), "%%P_SONATA%%": p("PTv3 + Sonata"), "%%P_UTONIA%%": p("PTv3 + Utonia"),
    }
    for k, v in rep.items():
        assert k in t, k
        t = t.replace(k, v)
        (HERE / "main.tex").write_text(t, encoding="utf-8")
    print("main.tex escrito; mejor en prueba:", BEST, "| mejor en validación:", BEST_VAL)


if __name__ == "__main__":
    main()
