# Detección de daños estructurales en nubes de puntos de dron

Proyecto final de la **Electiva de Ingeniería de Software** – Universidad de San Buenaventura Cali.
Autor: Jhonatan Velasco.

Segmentación semántica de nubes de puntos 3D capturadas con dron después del huracán Ian (dataset
[3DAeroRelief](https://github.com/BinaLab/3DAeroRelief)) en cinco clases: **fondo, edificación dañada,
edificación sin daño, vía y árbol**. Se comparan, con el mismo protocolo, cuatro modelos:

| Modelo | Semillas | mIoU (Área 2, resolución original) | IoU edificación dañada |
|---|---|---|---|
| PointNet++ | 3 | 0,436 ± 0,033 | 0,595 ± 0,092 |
| Point Transformer V3 desde cero | 1 | 0,447 | 0,731 |
| PTv3 + Sonata (preentrenado) | 3 | 0,505 ± 0,045 | 0,745 ± 0,071 |
| **PTv3 + Utonia (preentrenado)** | 3 | **0,545 ± 0,030** | **0,749 ± 0,054** |

Referencia: el mejor modelo publicado en el benchmark (PTv3) obtiene mIoU 0,458 (comparación indicativa: aquí se
corrigieron además la orientación y la escala de las nubes). El informe completo está en
[`Documento_Final_Electiva_Jhonatan_Velasco.pdf`](Documento_Final_Electiva_Jhonatan_Velasco.pdf).

## Contenido

| Carpeta / archivo | Qué contiene |
|---|---|
| `notebooks/` | **Notebooks ejecutados** con salidas: 1) datos y preprocesamiento, 2) modelos y entrenamiento, 3) evaluación y resultados (vuelve a ejecutar un modelo entrenado y genera todas las tablas del informe) |
| `code/` | Preprocesamiento, PointNet++ (PyTorch puro), entrenamiento de PointNet++ y PTv3, evaluación a resolución original, agregación de resultados |
| `runs/` | Métricas, registros de entrenamiento y predicciones de cada corrida (los pesos `.pt` están en la *Release*) |
| `demo/` | Aplicación web local que segmenta una nube y muestra el daño en 3D (`Iniciar_Demo.bat`) |
| `entrega_final/` | Informe en LaTeX (plantilla elsarticle) y sus figuras |
| `presentacion/` | Presentación de la sustentación (`.pptx`) y su generador (`build_deck.js`, con pptxgenjs) |

## Instalación

```
python -m venv .venv
.venv\Scripts\pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install point-transformer-v3 numpy scipy plyfile matplotlib scikit-learn pandas jupyter
```

Probado en Windows 11, Python 3.12, GPU NVIDIA RTX 4060 Ti (8 GB).

## Reproducir

1. **Datos:** descargar 3DAeroRelief (<https://github.com/BinaLab/3DAeroRelief>, ≈21 GB) y descomprimirlo en `data/3DAeroRelief/`.
2. **Preprocesamiento** (desde `code/`):
   ```
   python preprocess.py --input ../data/3DAeroRelief --output ../data/processed --voxel 0.1
   python estimate_scale.py ../data/processed ../data/scale.json
   python preprocess.py --input ../data/3DAeroRelief --output ../data/processed --voxel 0.1 --scale ../data/scale.json --areas Area_4 Area_6
   python add_normals.py ../data/processed
   ```
3. **Experimentos:** `bash run_all.sh` (≈10 h en una RTX 4060 Ti). Para usar los modelos ya entrenados, descargar los
   `.pt` de la *Release* y copiarlos en `runs/<corrida>/best.pt`.
4. **Resultados:** ejecutar los notebooks de `notebooks/`, o `python aggregate.py` (tablas y figuras).

División (oficial del benchmark): entrenamiento Áreas 1 y 3–7 · validación Área 8 · prueba Área 2.

## Demo

**Versión web (funciona en el celular):** <https://jhodave31.github.io/deteccion-danos-nubes-puntos/>. Muestra los análisis ya hechos por el modelo PTv3 + Utonia sobre las 8 nubes de prueba (carpeta `docs/`).

**Versión completa (local):** 
`Iniciar_Demo.bat` (o `python demo/app.py`) abre `http://localhost:8000`: elegir una nube o subir un `.ply`,
presionar **Analizar** y ver la predicción en 3D (vistas: modelo, color, etiqueta real y errores). El modelo se
configura en `demo/config.json`.

## Licencias y créditos

- Dataset: 3DAeroRelief (Le, Karimi y Rahnemoonfar, *Scientific Data*, 2026); no se redistribuye aquí.
- Point Transformer V3 y los codificadores preentrenados Sonata y Utonia: [Pointcept](https://github.com/Pointcept/PointTransformerV3).
  Los pesos de Sonata y Utonia tienen licencia CC BY-NC 4.0, por lo que los modelos ajustados a partir de ellos
  (Release) son solo para uso académico y no comercial.
- Visor 3D: [three.js](https://threejs.org) (MIT), incluido localmente en `demo/static/`.
