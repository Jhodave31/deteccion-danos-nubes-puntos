#!/usr/bin/env bash
# Regenera tablas, figuras, el documento final (PDF) y la presentación con los resultados de runs/.
# Uso (Git Bash, desde la carpeta del proyecto):  bash actualizar_entregables.sh
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
echo "== 1/4 Agregando resultados y figuras"
(cd code && $PY aggregate.py)
echo "== 2/4 Documento final"
(cd entrega_final && PYTHONIOENCODING=utf-8 $PY build_document.py && "${TECTONIC:-tectonic}" main.tex >/dev/null && cp main.pdf ../Documento_Final_Electiva_Jhonatan_Velasco.pdf)
echo "== 3/4 Presentación"
(cd presentacion && node build_deck.js && cp Sustentacion_Proyecto_Electiva.pptx ../Presentacion_Sustentacion.pptx)
echo "== 4/4 Listo"
ls -l Documento_Final_Electiva_Jhonatan_Velasco.pdf Presentacion_Sustentacion.pptx
