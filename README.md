# Práctica 2 · Detección de coches — YOLO desde cero

Detección de coches (una clase, `car`) con un YOLO26n entrenado **desde pesos aleatorios**
sobre 50 imágenes de PASCAL VOC 2007.

## Requisitos

- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- Python ≥ 3.12 (uv lo descarga si hace falta). Probado con Python 3.14.6, `ultralytics==8.4.165`, `torch==2.14.0`.
- CPU (probado en Apple M4, 16 GB).

```bash
uv sync            # instala las dependencias fijadas en uv.lock
```

## Reproducir

```bash
# 1. Datos: PASCAL VOC 2007 trainval (~450 MB)
mkdir -p data/raw && cd data/raw
curl -L -o voc2007.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip
unzip -q voc2007.zip && rm voc2007.zip && cd ../..

# 2. Selección de 50 imágenes y división 35/10/5 (semilla 42)
uv run python src/prepare_data.py
uv run python src/review_labels.py          # mosaicos para revisar etiquetas

# 3. Entrenamiento desde cero: dos configuraciones que solo cambian imgsz
uv run python src/train.py --name img416 --imgsz 416
uv run python src/train.py --name img640 --imgsz 640

# 4. Selección con validación (modelo y umbral de confianza)
uv run python src/select_model.py --runs img416 img640

# 5. Inferencia y galería sobre test
uv run python src/predict.py --split test
```

## Estructura

| Ruta | Contenido |
|---|---|
| `src/prepare_data.py` | VOC → YOLO: filtra `car`, descarta `difficult`, estratifica, elimina casi-duplicados (dHash) y divide |
| `src/review_labels.py` | Mosaicos con el ground truth por partición (`outputs/label_review/`) |
| `src/train.py` | Entrenamiento desde `yolo26n.yaml` (sin `.pt`), guarda `run_info.json` con parámetros, tiempo y hardware |
| `src/select_model.py` | Curvas comparadas, elección por mAP50 de validación y umbral por F1 máximo en validación |
| `src/predict.py` | Predicciones de test (`predictions.json`, `labels/`), tiempos y galería original \| predicción |
| `dataset/` | `images/`, `labels/`, `data.yaml` y `split.csv` (lista de imágenes por partición) |
| `runs/train/<exp>/` | Pesos, `results.csv`, curvas de Ultralytics y `run_info.json` |
| `models/` | Pesos finales elegidos, `selection.json`, `configs_comparison.csv` y curvas comparadas |
| `outputs/` | Revisión de etiquetas y resultados de inferencia |

## Desde cero vs preentrenado

`YOLO("yolo26n.yaml")` construye la red a partir de la **definición de la arquitectura** (capas y
escalado), con inicialización aleatoria. `YOLO("yolo26n.pt")` cargaría un **checkpoint** con pesos ya
entrenados en COCO. `train.py` rechaza cualquier modelo que no sea `.yaml` y entrena con `pretrained=False`.

## Parámetros de entrenamiento

Comunes: 100 épocas, lote 8, `patience=0` (sin early stopping, para ver la curva completa), semilla 0,
`deterministic=True`, CPU, hiperparámetros y aumentos por defecto de Ultralytics 8.4.165.
Parámetro variado: `imgsz` ∈ {416, 640}. Los valores exactos de cada ejecución están en
`runs/train/<exp>/args.yaml` y `run_info.json`.

## Evaluación común

`src/evaluate.py` evalúa cualquier `predictions.json` (YOLO desde cero, YOLOE o VLM) con el mismo
procedimiento: emparejamiento voraz con IoU ≥ 0,5 (cada coche real solo una vez, como en VOC),
TP/FP/FN, precisión, recall y F1 en el umbral fijado en validación, IoU medio de los aciertos,
AP50 y AP50-95 (solo métodos con puntuación), recall por tamaño del coche y métricas por tipo de escena.

```bash
uv run python src/evaluate.py --pred outputs/yolo_scratch/predictions.json outputs/yoloe/predictions.json \
    outputs/vlm/predictions.json --out outputs/comparison
```

Contrastado con el validador de Ultralytics en validación: AP50 0,154 frente a 0,146 (la diferencia
se debe a la interpolación de 101 puntos de Ultralytics frente a la de todos los puntos de VOC).

## Parte 2 (no entregable)

El enunciado pide no entregar el código del detector preentrenado ni del VLM: está en
`part2_no_entregable/` solo para uso interno y su uso se documenta en el informe. Sus dependencias
extra (CLIP para YOLOE) están en el grupo `part2`: `uv sync --group part2`.
Sus resultados (predicciones, selección en validación, tiempos y galería) sí se entregan en `outputs/`.
