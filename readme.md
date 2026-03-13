# SoberLens — Model Pipeline

Pipeline de datos y entrenamiento del modelo de detección de intoxicación por análisis facial. Este repositorio contiene todo el flujo para construir el clasificador binario (sobrio / ebrio) que alimenta la aplicación móvil SoberLens.

> El repositorio de la aplicación PWA se encuentra en [SoberLens_App](https://github.com/tu-org/soberlens-app) *(próximamente)*.

---

## Contexto del proyecto

SoberLens es un proyecto académico del curso terminal de Ciencias de Datos en la Universidad Panamericana. El objetivo es detectar signos de intoxicación alcohólica mediante análisis facial desde la cámara de un smartphone, alertando a un contacto de emergencia cuando se detecta un estado de riesgo.

La base académica del modelo es el trabajo de Mejia et al. (2019), que reporta 81% de precisión con Gradient Boosted Machines usando 68 landmarks faciales (dlib). Este pipeline extiende ese trabajo utilizando MediaPipe FaceLandmarker, que proporciona 478 landmarks de mayor precisión.

**Equipo:**

- Yahwthani Morales Gómez
- Gabriel Torres Zacarias
- Sebastián Avilez Hernández
- Gabriel Zaid Gutiérrez Gonzáles

---

## Resultados del modelo

| Métrica | Valor |
|---|---|
| Algoritmo | Random Forest |
| Accuracy en test set | 89.36% |
| Recall ebrio (threshold 0.30) | 94.2% |
| Recall sobrio (threshold 0.30) | 80.8% |
| Threshold seleccionado | 0.30 |
| Features extraídas | 327 |
| Imágenes de entrenamiento | 14,693 (con augmentation) |
| Imágenes de test | 733 (sin augmentation) |

El threshold de 0.30 fue seleccionado para maximizar el recall de detección de ebriedad, priorizando la seguridad del usuario sobre los falsos positivos.

---

## Arquitectura del pipeline

```
Videos YouTube (CSV)
        │
        ▼
  video_downloader.py
  (descarga secuencial)
        │
        ▼
  face_extractor.py
  (MediaPipe BlazeFace + FaceLandmarker)
  (filtros de calidad: frontalidad, nitidez,
   completitud del rostro, validación anatómica)
        │
        ▼
  output/sober/  output/drunk/
  (imágenes 224x224 px alineadas)
        │
        ▼
  pipeline.py  ←── punto de entrada principal
        │
    ┌───┴───────────────────┐
    │                       │
    ▼                       ▼
  Split 80/20         Test set (original)
    │
    ▼
  augmentation.py
  (solo sobre train)
    │
    ▼
  feature_extractor.py
  (327 features: landmarks XY,
   vectores, distancias, color LAB)
    │
    ▼
  train.py
  (Random Forest, SVM, GBM)
    │
    ▼
  tune_threshold.py
  (optimiza threshold por recall_drunk)
    │
    ▼
  output/models/
  (model.pkl, scaler.pkl,
   features.txt, metadata.txt)
```

---

## Estructura del repositorio

```
SoberLens_Model/
├── src/
│   ├── pipeline.py               # Orquestador principal — ejecuta todo el flujo
│   ├── face_extractor.py         # Extracción de rostros desde video
│   ├── video_downloader.py       # Descarga de videos desde YouTube (yt-dlp)
│   ├── process_existing_images.py # Extracción desde imágenes estáticas
│   ├── augmentation.py           # Augmentation del set de entrenamiento
│   ├── feature_extractor.py      # Extracción de features (327) → CSV
│   ├── train.py                  # Entrenamiento y comparación de clasificadores
│   └── tune_threshold.py         # Optimización del threshold de decisión
├── data/
│   ├── sober_videos.csv          # URLs de videos (personas sobrias)
│   └── drunk_videos.csv          # URLs de videos (personas ebrias)
├── output/
│   ├── sober/                    # Imágenes de rostros sobrios (224x224)
│   ├── drunk/                    # Imágenes de rostros ebrios (224x224)
│   ├── split/                    # Dataset dividido 80/20
│   │   ├── train/sober/
│   │   ├── train/drunk/
│   │   ├── test/sober/
│   │   └── test/drunk/
│   ├── train_features.csv        # Features del set de entrenamiento
│   ├── test_features.csv         # Features del set de test
│   └── models/
│       ├── model.pkl             # Modelo entrenado (Random Forest)
│       ├── scaler.pkl            # StandardScaler (ajustado solo en train)
│       ├── features.txt          # Lista ordenada de los 327 features
│       └── metadata.txt          # Métricas, threshold y configuración
├── temp_videos/                  # Videos temporales (se eliminan automáticamente)
├── requirements.txt
└── README.md
```

---

## Instalación

### Requisitos

- Python 3.10+
- Windows, macOS o Linux
- Espacio en disco: ~2 GB temporales durante la extracción (los videos se eliminan tras procesarse)

### Configurar entorno

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Los modelos de MediaPipe se descargan automáticamente en `src/models/` en la primera ejecución.

---

## Uso

### Opción A: Pipeline completo

Ejecuta todo el flujo desde las imágenes ya extraídas hasta el modelo entrenado:

```bash
cd src
python pipeline.py
```

El pipeline ejecuta en orden: split → augmentation → feature extraction → training → threshold tuning.

### Opción B: Solo extracción de rostros desde videos

```bash
cd src
python main.py
```

Lee los CSVs en `data/`, descarga los videos de a uno, extrae los rostros y elimina el video. Los resultados van a `output/sober/` y `output/drunk/`.

### Opción C: Extracción desde imágenes existentes

```bash
cd src
python process_existing_images.py
```

### Opción D: Ajustar threshold después de entrenar

```bash
cd src
python tune_threshold.py
```

---

## Configuración principal

Los parámetros clave se configuran como constantes en cada script. Los más relevantes:

### Extracción de rostros (`face_extractor.py`)

| Constante | Valor actual | Descripción |
|---|---|---|
| `MIN_CONFIDENCE` | 0.8 | Confianza mínima de detección (BlazeFace) |
| `FACE_OUTPUT_SIZE` | 224 | Tamaño de salida en píxeles |
| `SAMPLE_INTERVAL` | 0.3 | Segundos entre frames muestreados |
| `MAX_FACES_PER_VIDEO` | 100 | Límite de imágenes por video |
| `MAX_FACE_YAW_ASYMMETRY` | 0.35 | Tolerancia de rotación (0 = frontal estricto) |
| `min_sharpness` | 60.0 | Nitidez mínima (Laplacian variance) |

### Pipeline (`pipeline.py`)

| Constante | Valor actual | Descripción |
|---|---|---|
| `TEST_SIZE` | 0.20 | Proporción del set de test |
| `AUGMENTATIONS_PER_IMAGE` | 4 | Imágenes generadas por augmentation |
| `RANDOM_SEED` | 42 | Semilla para reproducibilidad |

### Threshold (`tune_threshold.py`)

| Constante | Valor actual | Descripción |
|---|---|---|
| `OPTIMIZATION_CRITERION` | `"recall_drunk"` | Métrica a maximizar |
| `MIN_SOBER_RECALL` | 0.80 | Recall mínimo aceptable en clase sobrio |

---

## Decisiones de diseño

### Por qué Random Forest sobre Gradient Boosted Machines

El paper de referencia (Mejia et al., 2019) obtuvo sus mejores resultados con GBM. En este pipeline, Random Forest supera a GBM en cross-validation (93.92% vs 90.89%) con menor varianza. La diferencia se atribuye al mayor número de features (327 vs 68 en el paper) y al uso de MediaPipe (478 landmarks vs 68 de dlib).

### Por qué threshold 0.30 en lugar del default 0.50

El contexto de uso es seguridad personal. Un falso negativo (detectar sobrio cuando está ebrio) tiene consecuencias más graves que un falso positivo. Con threshold 0.30, el recall de detección de ebriedad sube de 86.5% a 94.2%, aceptando reducir el recall sobrio de 91.5% a 80.8%.

### Separación estricta train/test antes de augmentation

El augmentation se aplica exclusivamente sobre el set de entrenamiento después del split. Aplicarlo antes generaría data leakage: variantes de la misma imagen original en ambos sets, inflando artificialmente los resultados (diferencia medida: ~4.3 puntos porcentuales).

### Por qué solo vistas frontales

Los landmarks de MediaPipe en vistas de perfil o tres cuartos no tienen el mismo significado geométrico que en vistas frontales. Incluir vistas no frontales introduce ruido sistemático en los features de distancia y simetría. El filtro `MAX_FACE_YAW_ASYMMETRY` controla la tolerancia.

---

## Dataset

El dataset fue construido a partir de videos de YouTube descargados y procesados mediante este pipeline.

| Clase | Imágenes originales |
|---|---|
| Sobrio | 2,146 |
| Ebrio | 1,841 |
| **Total** | **3,987** |

Las imágenes son rostros de 224×224 px, alineados por landmarks de iris (MediaPipe), en escala de grises normalizada.

**Nota:** Las imágenes no se distribuyen en este repositorio por razones de derechos de autor. Los CSVs con las URLs de los videos originales están en `data/`.

---

## Relación con SoberLens_App

Este repositorio produce los artefactos que consume la aplicación:

```
SoberLens_Model/output/models/
├── model.pkl       →  copiado a  →  SoberLens_App/backend/model/model.pkl
├── scaler.pkl      →  copiado a  →  SoberLens_App/backend/model/scaler.pkl
└── features.txt    →  copiado a  →  SoberLens_App/backend/model/features.txt
```

El servidor de la app (FastAPI) usa `model.pkl` y `scaler.pkl` para clasificar en tiempo real. No re-entrena en producción; el re-entrenamiento ocurre en este repositorio con datos nuevos confirmados por los usuarios.

---

## Referencia

Mejia, J. et al. (2019). *Predicting Alcohol Intoxication from Facial Cues*. Worcester Polytechnic Institute.

El paper reporta:

- 81% de precisión con GBM usando 68 landmarks (dlib)
- Features principales: vectores de movimiento, apertura ocular, simetría facial
- Dataset: videos de YouTube de personas sobrias y en estado de intoxicación
