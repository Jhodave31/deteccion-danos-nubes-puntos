# Segundo avance – PointNet++ sobre 3DAeroRelief

Código del experimento descrito en la sección "Materiales y métodos" del documento.

## Requisitos

- Python 3.10+ y GPU NVIDIA (probado en RTX 4060 Ti, 8 GB)
- `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- `pip install numpy scipy plyfile matplotlib scikit-learn tqdm`

## Pasos

1. Descargar el dataset desde el repositorio oficial: <https://github.com/BinaLab/3DAeroRelief>
   (enlace de Dropbox, ~21 GB) y descomprimirlo en `data/3DAeroRelief/` (carpetas `Area_1` … `Area_8`).
2. Preprocesar (fusión de etiquetas, alineación vertical, submuestreo a 0,1 m):
   ```
   python preprocess.py --input data/3DAeroRelief --output data/processed --voxel 0.1
   ```
3. Armonizar la escala de las áreas de entrenamiento (usa solo etiquetas de las áreas 1, 3–8)
   y reprocesar las áreas corregidas (4 y 6):
   ```
   python estimate_scale.py data/processed data/scale.json
   python preprocess.py --input data/3DAeroRelief --output data/processed --voxel 0.1 --scale data/scale.json --areas Area_4 Area_6
   ```
   Opcional: `python check_orientation.py data/3DAeroRelief data/processed/stats.json` compara la
   orientación vertical con las normales y con las etiquetas.
4. Entrenar y evaluar (train: Áreas 1,3–7; validación: Área 8; prueba: Área 2):
   ```
   python train.py --data data/processed --out runs/pn2 --block 12 --epochs 40 --iters 400 --bs 16 --val_every 4
   ```
   Se generan `runs/pn2/log.json`, `best.pt`, `results.json` y `test_preds.npz`.
5. Figuras:
   ```
   python make_figures.py --data data/processed --run runs/pn2 --out figs
   ```

## Archivos

| Archivo | Contenido |
|---|---|
| `preprocess.py` | Fusión color→clase, alineación vertical (PCA + normales), escala opcional y submuestreo por vóxel |
| `estimate_scale.py` | Detecta áreas con escala inconsistente (altura de edificaciones) y propone factores |
| `check_orientation.py` | Diagnóstico de la orientación vertical (normales vs. etiquetas) |
| `pointnet2.py` | PointNet++ (SSG) en PyTorch puro (FPS, ball query e interpolación sin extensiones CUDA) |
| `train.py` | Muestreo por bloques, aumento de datos, entrenamiento, ventana deslizante y métricas |
| `make_figures.py` | Curvas, matriz de confusión, distribución de clases y figuras cualitativas |
