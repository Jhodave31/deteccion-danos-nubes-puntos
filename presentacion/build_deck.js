// Genera la presentación de sustentación a partir de los resultados (runs/summary.json).
// Uso:  node build_deck.js      (desde la carpeta presentacion/; requiere pptxgenjs)
const path = require("path");
const fs = require("fs");
const NODE_MODULES = process.env.PPTX_NODE_MODULES || "";
const pptxgen = require(NODE_MODULES ? path.join(NODE_MODULES, "pptxgenjs") : "pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const FIGS = path.join(ROOT, "entrega_final", "figs");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "runs", "summary.json"), "utf8"));
const OUT = path.join(__dirname, "Sustentacion_Proyecto_Electiva.pptx");

const THEME = {
  name: "Daño post-huracán",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F27", lt1: "FFFFFF", dk2: "2A3442", lt2: "F2F4F7",
    accent1: "E6194B", accent2: "1FA7B5", accent3: "2E7D32", accent4: "E0A800",
    accent5: "7F8C8D", accent6: "1F4E79", hlink: "1FA7B5", folHlink: "7F8C8D",
  },
};
const HEX = THEME.colors;

// ---------- utilidades de datos ----------
const pct = (v, d = 0) => (100 * v).toFixed(d).replace(".", ",");
const num = (v, d = 3) => v.toFixed(d).replace(".", ",");
const M = (name, key = "full_tta") => S[name] && S[name][key];
const mean = (o) => o.mean;
const pm = (o, d = 3) => (o.n > 1 ? `${num(o.mean, d)} ± ${num(o.std, d)}` : num(o.mean, d));
const NAMES = ["PointNet++", "PTv3 desde cero", "PTv3 + Sonata", "PTv3 + Utonia"].filter((n) => S[n] && S[n].full_tta);
const BEST = S._best_by_test || "PTv3 + Sonata";
const seeds = (n) => S[n].runs.length;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13,33 x 7,5 in
pres.author = "Jhonatan Velasco";
pres.title = "Detección de daños estructurales en nubes de puntos de dron";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
const FOOT = "Jhonatan Velasco · Electiva de Ingeniería de Software · USB Cali";
pres.defineSlideMaster({
  title: "PORTADA",
  background: { color: C.text1 },
  objects: [],
  placeholders: [],
});
pres.defineSlideMaster({
  title: "CONTENIDO",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 9, h: 0.3, fontSize: 10, color: C.accent5, isTextBox: true, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  placeholders: [],
});

function titleOf(slide, text, sub) {
  slide.addText(text, { x: 0.6, y: 0.35, w: 12.1, h: 0.75, fontSize: 32, bold: true, color: C.text1,
    fontFace: THEME.headFontFace, isTextBox: true, margin: 0, objectName: "Título" });
  if (sub) slide.addText(sub, { x: 0.6, y: 1.05, w: 12.1, h: 0.45, fontSize: 16, color: C.accent5,
    isTextBox: true, margin: 0, objectName: "Subtítulo" });
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 },
    line: { type: "none" }, objectName: "Tarjeta" });
}
function dot(slide, x, y, color, label) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.55, h: 0.55, fill: { color }, line: { type: "none" }, objectName: "Ícono" });
  slide.addText(label, { x, y, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.background1, align: "center",
    valign: "middle", isTextBox: true, margin: 0 });
}
function imageSize(p) {  // ancho y alto de un PNG leyendo su cabecera
  const b = fs.readFileSync(p);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
}
function img(slide, file, x, y, w, h, alt) {
  const p = path.join(FIGS, file);
  if (!fs.existsSync(p)) return;
  // conserva la proporción de la imagen dentro del recuadro (x, y, w, h), centrada
  const { width: iw, height: ih } = imageSize(p);
  const r = Math.min(w / iw, h / ih), W = iw * r, H = ih * r;
  slide.addImage({ path: p, x: x + (w - W) / 2, y: y + (h - H) / 2, w: W, h: H, altText: alt, objectName: alt });
}

// ================= 1. Portada =================
pres.addSection({ title: "Introducción" });
let s = pres.addSlide({ masterName: "PORTADA", sectionTitle: "Introducción" });
s.addText("Detección de daños estructurales en nubes de puntos de dron", { x: 0.8, y: 1.7, w: 11.7, h: 1.8, fontSize: 44,
  bold: true, color: C.background1, isTextBox: true, margin: 0, objectName: "Título" });
s.addText("Comparación de PointNet++ y Point Transformer V3 preentrenado sobre el benchmark 3DAeroRelief (huracán Ian, 2022)",
  { x: 0.8, y: 3.55, w: 11.2, h: 0.9, fontSize: 20, color: "CBD3DD", isTextBox: true, margin: 0 });
s.addText([{ text: "Jhonatan Velasco · 98251", options: { bold: true, breakLine: true } },
  { text: "Electiva de Ingeniería de Software · Universidad de San Buenaventura Cali" }],
  { x: 0.8, y: 5.4, w: 11, h: 0.9, fontSize: 16, color: C.background1, isTextBox: true, margin: 0 });
s.addShape(pres.shapes.OVAL, { x: 11.3, y: 5.3, w: 1.1, h: 1.1, fill: { color: C.accent1 }, line: { type: "none" }, objectName: "Motivo" });
s.addNotes("Buenos días. Mi proyecto trata de cómo usar inteligencia artificial para detectar automáticamente qué edificaciones quedaron dañadas después de un desastre natural, a partir de modelos 3D tomados con drones.");

// ================= 2. Problema =================
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Introducción" });
titleOf(s, "El problema: inspeccionar el daño tras un desastre", "Hoy se hace a ojo, casa por casa");
const probs = [
  ["Lento", "Un equipo revisa cada estructura en sitio; en un huracán son miles de edificaciones.", C.accent1, "1"],
  ["Subjetivo", "El diagnóstico depende de la experiencia del inspector.", C.accent4, "2"],
  ["Riesgoso", "El personal entra a zonas inestables o de difícil acceso.", C.accent6, "3"],
];
probs.forEach(([h, t, col, n], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 1.9, 3.8, 2.6);
  dot(s, x + 0.3, 2.15, col, n);
  s.addText(h, { x: x + 0.3, y: 2.85, w: 3.2, h: 0.5, fontSize: 22, bold: true, color: C.text1, isTextBox: true, margin: 0 });
  s.addText(t, { x: x + 0.3, y: 3.35, w: 3.25, h: 1.0, fontSize: 15, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
});
card(s, 0.6, 4.85, 12.1, 1.75, C.text2);
s.addText([{ text: "La propuesta: ", options: { bold: true, color: "FF8FA8" } },
  { text: "volar un dron, reconstruir la zona como una nube de puntos 3D y que un modelo de aprendizaje profundo marque cada punto como edificación dañada, edificación sin daño, vía, árbol o fondo.", options: { color: C.background1 } }],
  { x: 0.95, y: 5.0, w: 11.4, h: 1.45, fontSize: 18, isTextBox: true, margin: 0, valign: "middle" });
s.addNotes("La inspección actual es lenta, subjetiva y peligrosa. La idea es reemplazar el primer barrido por un dron y un modelo que clasifique cada punto del modelo 3D en cinco clases.");

// ================= 3. Datos =================
pres.addSection({ title: "Datos y método" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Datos y método" });
titleOf(s, "Los datos: 3DAeroRelief", "Primer benchmark 3D público de daño post-desastre (Scientific Data, 2026)");
const stats = [["64", "nubes de puntos"], ["8", "zonas de Florida"], ["188 M", "puntos con etiqueta"], ["5", "clases"]];
stats.forEach(([v, l], i) => {
  const y = 1.85 + i * 1.2;
  s.addText(v, { x: 0.6, y, w: 2.3, h: 0.75, fontSize: 40, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
  s.addText(l, { x: 2.95, y: y + 0.12, w: 2.6, h: 0.55, fontSize: 17, color: C.text2, isTextBox: true, margin: 0, valign: "middle" });
});
img(s, "distribucion.png", 5.9, 1.75, 6.9, 2.37, "Distribución de clases por área");
card(s, 5.9, 4.4, 6.9, 2.2);
s.addText([{ text: "División oficial del benchmark", options: { bold: true, breakLine: true } },
  { text: "Entrenamiento: áreas 1, 3–7  ·  Validación: área 8  ·  Prueba: área 2 (8 nubes, 31 M puntos) — zonas que el modelo nunca vio.", options: { breakLine: true } },
  { text: "Captura: video de dron → reconstrucción 3D con COLMAP.", options: {} }],
  { x: 6.15, y: 4.55, w: 6.4, h: 1.95, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 6 });
s.addNotes("El dataset tiene 64 nubes reales del huracán Ian. Uso la misma división del artículo: el área 2 se reserva para la prueba final, así se mide si el modelo generaliza a una zona nueva.");

// ================= 4. Problemas del dataset =================
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Datos y método" });
titleOf(s, "Dos problemas del dataset que había que corregir", "Ninguno estaba documentado; sin corregirlos, la altura no significa nada");
const fixes = [
  ["Nubes giradas", "COLMAP entrega cada nube con orientación arbitraria: en varias, el “arriba” era el eje y.",
   "Se alinea el suelo con PCA y el sentido se fija con las normales (apuntan al dron). 62 de 64 nubes verificadas.", C.accent2, "↻"],
  ["Escala distinta", "En las áreas 4 y 6 las casas medían ~3,5 veces más que en el resto.",
   "Se reescalan con la altura típica de las edificaciones (solo áreas de entrenamiento: factores 0,285 y 0,354).", C.accent4, "⇲"],
];
fixes.forEach(([h, prob, sol, col, ic], i) => {
  const x = 0.6 + i * 6.15;
  card(s, x, 1.85, 5.95, 4.75);
  dot(s, x + 0.35, 2.15, col, ic);
  s.addText(h, { x: x + 1.1, y: 2.15, w: 4.5, h: 0.55, fontSize: 24, bold: true, color: C.text1, isTextBox: true, margin: 0, valign: "middle" });
  s.addText([{ text: "Problema", options: { bold: true, color: C.accent1, breakLine: true } }, { text: prob }],
    { x: x + 0.35, y: 3.0, w: 5.3, h: 1.6, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
  s.addText([{ text: "Solución", options: { bold: true, color: C.accent3, breakLine: true } }, { text: sol }],
    { x: x + 0.35, y: 4.6, w: 5.3, h: 1.8, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
});
s.addNotes("Al revisar los datos encontré dos problemas que el artículo no menciona. Las nubes estaban giradas y dos áreas tenían otra escala. Los corregí sin usar las etiquetas de prueba, para no hacer trampa.");

// ================= 5. Método =================
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Datos y método" });
titleOf(s, "Metodología: el mismo protocolo para todos los modelos", "Así la comparación es justa");
const steps = ["Fusionar etiquetas", "Alinear y escalar", "Vóxel 0,1 m + normales", "Bloques 12 × 12 m", "Entrenar y validar (área 8)", "Probar en el área 2"];
steps.forEach((t, i) => {
  const x = 0.6 + i * 2.05;
  card(s, x, 2.0, 1.85, 1.6, i === 5 ? C.text2 : C.background2);
  s.addText(String(i + 1), { x: x + 0.15, y: 2.1, w: 0.6, h: 0.5, fontSize: 22, bold: true, color: i === 5 ? "FF8FA8" : C.accent1, isTextBox: true, margin: 0 });
  s.addText(t, { x: x + 0.15, y: 2.6, w: 1.6, h: 0.9, fontSize: 15, bold: true, color: i === 5 ? C.background1 : C.text1, isTextBox: true, margin: 0, valign: "top" });
});
const mrows = [
  ["Pérdida", "Entropía cruzada ponderada por frecuencia de clase (las vías son solo el 6 %)"],
  ["Selección", "Se guarda el modelo con mejor mIoU en validación; la prueba se mira una sola vez"],
  ["Métricas", "mIoU (principal), IoU por clase, exactitud global y F1, a la resolución original (31 M puntos)"],
  ["Repeticiones", "Hasta 3 semillas por modelo (prioridad a los preentrenados) → media ± desviación"],
];
mrows.forEach(([k, v], i) => {
  const y = 4.0 + i * 0.66;
  s.addText(k, { x: 0.6, y, w: 2.2, h: 0.55, fontSize: 16, bold: true, color: C.accent1, isTextBox: true, margin: 0, valign: "middle" });
  s.addText(v, { x: 2.8, y, w: 9.9, h: 0.55, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "middle" });
});
s.addNotes("Todos los modelos ven exactamente los mismos datos, con la misma pérdida y el mismo criterio de selección. La métrica principal es el mIoU: el promedio, por clase, de qué tanto se superpone la predicción con la etiqueta real.");

// ================= 6. Modelos =================
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Datos y método" });
titleOf(s, "Cuatro modelos comparados", "De una red clásica a transformers que ya “saben” de 3D");
const P = (n) => (S[n] ? (S[n].params / 1e6).toFixed(1).replace(".", ",") + " M parámetros" : "");
const models = [
  ["PointNet++", "2017", "Red clásica: agrupa vecinos por radio y resume cada grupo.", "Desde cero", P("PointNet++"), C.accent5],
  ["PTv3", "2024", "Transformer: ordena la nube con curvas que recorren el espacio y aplica atención.", "Desde cero", P("PTv3 desde cero"), C.accent6],
  ["PTv3 + Sonata", "2025", "Mismo transformer, preentrenado sin etiquetas en miles de escenas 3D (interiores).", "Preentrenado", P("PTv3 + Sonata"), C.accent1],
  ["PTv3 + Utonia", "2026", "Preentrenado sin etiquetas en nubes de muchos dominios, incluidas aéreas.", "Preentrenado", P("PTv3 + Utonia"), C.accent2],
].filter((m) => S[m[0] === "PTv3" ? "PTv3 desde cero" : m[0]]);
const cw = (12.1 - 0.3 * (models.length - 1)) / models.length;
models.forEach(([h, yr, t, tag, p, col], i) => {
  const x = 0.6 + i * (cw + 0.3);
  const dark = tag === "Preentrenado";
  card(s, x, 1.85, cw, 4.75, dark ? C.text2 : C.background2);
  const fg = dark ? C.background1 : C.text1;
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 2.1, w: 0.5, h: 0.5, fill: { color: col }, line: { type: "none" }, objectName: "Ícono" });
  s.addText(yr + " · " + tag, { x: x + 0.95, y: 2.1, w: cw - 1.1, h: 0.5, fontSize: 14, color: dark ? "CBD3DD" : C.accent5, isTextBox: true, margin: 0, valign: "middle" });
  s.addText(h, { x: x + 0.3, y: 2.8, w: cw - 0.5, h: 0.6, fontSize: 22, bold: true, color: fg, isTextBox: true, margin: 0 });
  s.addText(t, { x: x + 0.3, y: 3.45, w: cw - 0.55, h: 2.2, fontSize: 15, color: fg, isTextBox: true, margin: 0, valign: "top" });
  s.addText(p, { x: x + 0.3, y: 5.85, w: cw - 0.5, h: 0.45, fontSize: 14, italic: true, color: dark ? "FF8FA8" : C.accent1, isTextBox: true, margin: 0 });
});
s.addNotes("Comparo cuatro modelos con exactamente el mismo protocolo. Dos se entrenan desde cero: PointNet++, la línea base, y PTv3, el estado del arte de 2024. Los otros dos son el mismo PTv3 pero partiendo de pesos preentrenados sin etiquetas: Sonata, entrenado sobre todo con escenas interiores, y Utonia, que incluye nubes aéreas.");

// ================= 7. Resultado principal =================
pres.addSection({ title: "Resultados" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
const PRE = NAMES.filter((n) => n.includes("Sonata") || n.includes("Utonia"));
const NOPRE = NAMES.filter((n) => !PRE.includes(n));
const preWins = PRE.length && Math.min(...PRE.map((n) => mean(M(n).mIoU))) > Math.max(...NOPRE.map((n) => mean(M(n).mIoU)));
titleOf(s, preWins ? "El preentrenamiento marca la diferencia" : `${BEST} obtiene el mejor resultado`,
  `Área 2 (zona nunca vista), resolución original · ${BEST}: ${seeds(BEST) > 1 ? "media de " + seeds(BEST) + " semillas" : "1 semilla"}`);
const big = [
  [pct(mean(M(BEST).IoU[1])) + " %", "IoU de edificación dañada", `PointNet++: ${pct(mean(M("PointNet++").IoU[1]))} %`],
  [num(mean(M(BEST).mIoU)), "mIoU (5 clases)", `PointNet++: ${num(mean(M("PointNet++").mIoU))} · PTv3 desde cero: ${num(mean(M("PTv3 desde cero").mIoU))}`],
  [pct(mean(M(BEST).OA)) + " %", "de puntos bien clasificados", `PointNet++: ${pct(mean(M("PointNet++").OA))} %`],
];
big.forEach(([v, l, cmp], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 1.85, 3.8, 2.85);
  s.addText(v, { x: x + 0.3, y: 2.05, w: 3.3, h: 1.15, fontSize: 60, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
  s.addText(l, { x: x + 0.3, y: 3.2, w: 3.3, h: 0.5, fontSize: 18, bold: true, color: C.text1, isTextBox: true, margin: 0 });
  s.addText(cmp, { x: x + 0.3, y: 3.75, w: 3.3, h: 0.8, fontSize: 14, color: C.accent5, isTextBox: true, margin: 0, valign: "top" });
});
card(s, 0.6, 5.0, 12.1, 1.6, C.text2);
s.addText([{ text: "Contra el benchmark publicado: ", options: { bold: true, color: "FF8FA8" } },
  { text: `el mejor modelo del artículo (PTv3) obtiene mIoU 0,458 e IoU de daño 0,761. ${BEST} llega a ${num(mean(M(BEST).mIoU))} y ${num(mean(M(BEST).IoU[1]))}. Es una comparación indicativa: aquí además se corrigieron la orientación y la escala de las nubes.`, options: { color: C.background1 } }],
  { x: 0.95, y: 5.1, w: 11.4, h: 1.4, fontSize: 16, isTextBox: true, margin: 0, valign: "middle" });
s.addNotes(`El mejor modelo en la zona de prueba es ${BEST}. Detecta la clase que más importa, edificación dañada, con un IoU de ${num(mean(M(BEST).IoU[1]), 2)}, y supera los valores publicados por los autores del dataset, aunque la comparación no es perfecta porque yo corregí problemas de los datos.`);

// ================= 8. IoU por clase (gráfico nativo) =================
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
titleOf(s, "Resultados por clase", "IoU en el área de prueba (resolución original, media de las semillas)");
const labels = ["Fondo", "Edif. dañada", "Edif. sin daño", "Vía", "Árbol", "mIoU"];
const palette = { "PointNet++": HEX.accent5, "PTv3 desde cero": HEX.accent6, "PTv3 + Sonata": HEX.accent1, "PTv3 + Utonia": HEX.accent2 };
const chartData = NAMES.map((n) => ({
  name: n, labels, values: [0, 1, 2, 3, 4].map((c) => +mean(M(n).IoU[c]).toFixed(3)).concat([+mean(M(n).mIoU).toFixed(3)]),
}));
s.addChart(pres.charts.BAR, chartData, {
  x: 0.6, y: 1.65, w: 8.4, h: 4.95, barDir: "col", barGapWidthPct: 50,
  chartColors: NAMES.map((n) => palette[n]),
  showLegend: true, legendPos: "t", legendFontSize: 13, legendFontFace: "+mn-lt", legendColor: HEX.dk2,
  catAxisLabelFontSize: 13, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.dk2,
  valAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt", valAxisLabelColor: HEX.accent5,
  valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0",
  valGridLine: { color: "E3E7EC", size: 1 }, catGridLine: { style: "none" },
  showValue: false, objectName: "Gráfico IoU por clase",
});
card(s, 9.3, 1.75, 3.4, 4.85);
s.addText([{ text: "Lo que muestra", options: { bold: true, breakLine: true } },
  { text: "Los modelos preentrenados mejoran sobre todo edificación sin daño y árboles.", options: { bullet: true, breakLine: true } },
  { text: "Vía casi no existe en el área de prueba (0,6 % de los puntos): su IoU es poco confiable.", options: { bullet: true, breakLine: true } },
  { text: "Edificación sin daño sigue siendo la clase más difícil.", options: { bullet: true } }],
  { x: 9.55, y: 1.95, w: 3.0, h: 4.5, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8 });
s.addNotes("Aquí está el detalle por clase. Los modelos preentrenados, en rojo y turquesa, mejoran casi todas las clases. La vía casi no aparece en la zona de prueba, por eso su valor es bajo y poco confiable.");

// ================= 9. Variabilidad entre semillas =================
const multi = NAMES.filter((n) => seeds(n) > 1);
if (multi.length) {
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
  titleOf(s, "¿Qué tan estable es el resultado?", "mIoU de prueba de cada repetición (semillas 0, 1 y 2)");
  const seedData = [0, 1, 2].map((k) => ({
    name: `Semilla ${k}`, labels: multi, values: multi.map((n) => +(S[n].full_tta.mIoU.values[k] ?? 0).toFixed(3)),
  }));
  s.addChart(pres.charts.BAR, seedData, {
    x: 0.6, y: 1.65, w: 7.6, h: 4.95, barDir: "col", barGapWidthPct: 60,
    chartColors: ["9AA5B1", "5D6D7E", HEX.dk2],
    showLegend: true, legendPos: "t", legendFontSize: 13, legendFontFace: "+mn-lt", legendColor: HEX.dk2,
    catAxisLabelFontSize: 13, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.dk2,
    valAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt", valAxisLabelColor: HEX.accent5,
    valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisLabelFormatCode: "0.0",
    valGridLine: { color: "E3E7EC", size: 1 }, catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.dk2,
    dataLabelFormatCode: "0.00", objectName: "Gráfico semillas",
  });
  card(s, 8.6, 1.75, 4.1, 4.85);
  const spread = multi.map((n) => { const v = S[n].full_tta.mIoU.values; return `${n}: ${num(Math.min(...v), 2)} a ${num(Math.max(...v), 2)}`; });
  s.addText([{ text: "Lección", options: { bold: true, breakLine: true } },
    { text: "Con la misma configuración, cambiar solo la semilla mueve el mIoU varios puntos.", options: { bullet: true, breakLine: true } },
    { text: spread.join(" · "), options: { bullet: true, breakLine: true } },
    { text: "Por eso se reportan media ± desviación, y las diferencias pequeñas entre modelos no son concluyentes.", options: { bullet: true } }],
    { x: 8.85, y: 1.95, w: 3.65, h: 4.5, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8 });
  s.addNotes("Algo importante que aprendí: repetir el mismo entrenamiento con otra semilla aleatoria cambia bastante el resultado, sobre todo en los modelos preentrenados. Por eso reporto el promedio y la desviación, y no un solo número.");
}

// ================= 10. Cualitativo =================
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
titleOf(s, "Lo que ve cada modelo", "Vista superior de dos nubes de prueba");
img(s, "cualitativo_final.png", 0.6, 1.55, 12.1, 4.9, "Comparación cualitativa de predicciones");
s.addText(`Abajo: PointNet++ marca como “sin daño” (celeste) buena parte del techo dañado; ${BEST} lo reconoce casi completo.`,
  { x: 0.6, y: 6.5, w: 12.1, h: 0.45, fontSize: 15, italic: true, color: C.text2, isTextBox: true, margin: 0 });
s.addNotes("En la nube de abajo se ve la diferencia: PointNet++ confunde gran parte del techo dañado con techo sano, y el modelo preentrenado lo identifica casi completo.");

// ================= 11. Errores =================
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
titleOf(s, "¿Dónde se equivoca?", `Matriz de confusión de ${BEST} (área de prueba)`);
img(s, "confusion_mejor.png", 0.6, 1.6, 5.6, 4.98, "Matriz de confusión del mejor modelo");
const RUN0 = { "PointNet++": null, "PTv3 desde cero": "ptv3_s0", "PTv3 + Sonata": "sonata_s0", "PTv3 + Utonia": "utonia_tiny_s0" }[BEST];
const CM = (() => {
  const f = RUN0 ? path.join(ROOT, "runs", RUN0, "test_tta_fullres.json") : path.join(ROOT, "runs", "pn2", "eval_tta_full.json");
  const m = JSON.parse(fs.readFileSync(f, "utf8")).confusion_full;
  return m.map((r) => { const t = r.reduce((a, b) => a + b, 0); return r.map((v) => v / t); });
})();
const lessons = [
  ["Detecta bien el daño", `El ${pct(CM[1][1])} % de los puntos de edificación dañada se reconoce correctamente.`],
  ["Confunde casas sanas", `El ${pct(CM[2][1])} % de los puntos de edificación sin daño se marca como dañada: un techo intacto y uno con daño leve se parecen mucho.`],
  ["Vía ↔ fondo", `El ${pct(CM[3][0])} % de la vía se confunde con fondo: ambas son superficies planas a ras de suelo y la vía es muy escasa.`],
];
lessons.forEach(([h, t], i) => {
  const y = 1.75 + i * 1.62;
  card(s, 6.6, y, 6.1, 1.45);
  dot(s, 6.85, y + 0.45, [C.accent3, C.accent1, C.accent4][i], String(i + 1));
  s.addText(h, { x: 7.6, y: y + 0.12, w: 4.9, h: 0.45, fontSize: 18, bold: true, color: C.text1, isTextBox: true, margin: 0 });
  s.addText(t, { x: 7.6, y: y + 0.55, w: 4.9, h: 0.85, fontSize: 14, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
});
s.addNotes("La matriz muestra dónde falla. El daño se detecta bien; el problema principal está en las casas sanas, que a veces se confunden con dañadas. Es un límite honesto del modelo y una línea de trabajo futuro.");

// ================= 12. Herramienta =================
pres.addSection({ title: "Software y cierre" });
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Software y cierre" });
titleOf(s, "La herramienta: del modelo a una aplicación", "Aplicación web local, funciona sin internet");
img(s, "demo.png", 0.6, 1.6, 7.6, 4.94, "Captura de la aplicación");
const feats = [["Elegir o subir", "una nube .ply"], ["Analizar", "mismo preprocesamiento del experimento"], ["Ver en 3D", "modelo · color · real · errores"], ["Resumen", "% de lo edificado dañado"]];
feats.forEach(([h, t], i) => {
  const y = 1.7 + i * 1.22;
  dot(s, 8.6, y + 0.1, C.accent1, String(i + 1));
  s.addText(h, { x: 9.35, y, w: 3.4, h: 0.45, fontSize: 18, bold: true, color: C.text1, isTextBox: true, margin: 0 });
  s.addText(t, { x: 9.35, y: y + 0.45, w: 3.4, h: 0.5, fontSize: 14, color: C.text2, isTextBox: true, margin: 0 });
});
s.addText("Código, notebooks y modelos: github.com/Jhodave31/deteccion-danos-nubes-puntos",
  { x: 8.6, y: 6.45, w: 4.2, h: 0.5, fontSize: 12, italic: true, color: C.accent5, isTextBox: true, margin: 0 });
s.addNotes("Como es una materia de ingeniería de software, construí una aplicación. Uno elige una nube, presiona analizar y ve en 3D las casas dañadas en rojo. Reutiliza exactamente el mismo código del experimento. Ahora la muestro en vivo.");

// ================= 13. Conclusiones =================
s = pres.addSlide({ masterName: "PORTADA", sectionTitle: "Software y cierre" });
s.addText("Conclusiones", { x: 0.8, y: 0.6, w: 11.7, h: 0.9, fontSize: 40, bold: true, color: C.background1, isTextBox: true, margin: 0, objectName: "Título" });
const concl = [
  ["Funciona", `En una zona que nunca vio, ${BEST} identifica la edificación dañada con IoU ${num(mean(M(BEST).IoU[1]), 2)}.`],
  [preWins ? "El preentrenamiento importa más que la arquitectura" : "El preentrenamiento ayuda",
   preWins ? `Entrenado desde cero, PTv3 rinde como PointNet++ (${num(mean(M("PTv3 desde cero").mIoU), 2)} vs ${num(mean(M("PointNet++").mIoU), 2)}); preentrenado, lo supera (${PRE.map((n) => num(mean(M(n).mIoU), 2)).join(" y ")}).`
     : `El mejor modelo es preentrenado (${BEST}: mIoU ${num(mean(M(BEST).mIoU), 2)} frente a ${num(mean(M("PointNet++").mIoU), 2)} de PointNet++), aunque la ventaja varía entre semillas.`],
  ["Revisar los datos y repetir", "Corregir orientación y escala fue necesario, y repetir con varias semillas evitó conclusiones apresuradas."],
  ["Siguiente paso", "Estabilizar el ajuste de los modelos preentrenados y distinguir mejor las casas sanas."],
];
concl.forEach(([h, t], i) => {
  const y = 1.8 + i * 1.22;
  s.addShape(pres.shapes.OVAL, { x: 0.8, y: y + 0.08, w: 0.5, h: 0.5, fill: { color: i === 3 ? HEX.accent2 : HEX.accent1 }, line: { type: "none" }, objectName: "Ícono" });
  s.addText(h, { x: 1.6, y, w: 10.8, h: 0.5, fontSize: 22, bold: true, color: C.background1, isTextBox: true, margin: 0 });
  s.addText(t, { x: 1.6, y: y + 0.5, w: 10.8, h: 0.55, fontSize: 16, color: "CBD3DD", isTextBox: true, margin: 0 });
});
s.addText("¡Gracias! · Preguntas", { x: 0.8, y: 6.6, w: 11.7, h: 0.5, fontSize: 18, bold: true, color: "FF8FA8", isTextBox: true, margin: 0 });
s.addNotes("Para cerrar: el sistema funciona en una zona que nunca vio, el preentrenamiento es la clave de la mejora, y revisar los datos y repetir los experimentos fue tan importante como el modelo. Muchas gracias.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("OK", OUT);
})();
