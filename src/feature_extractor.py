"""
Extraccion de features a partir de imagenes de rostros alineadas.

Por cada imagen se extraen 4 grupos de features basados en el paper DrunkSelfie
(Willoughby et al., 2019), adaptados para los 478 landmarks de MediaPipe:

    1. Posiciones X,Y normalizadas de landmarks clave
    2. Vectores desde el centroide hasta cada landmark (distancia y angulo)
    3. Distancias entre pares de landmarks clave (lineas faciales)
    4. Color promedio de la region de frente (enrojecimiento por alcohol)

Salida: CSV con una fila por imagen y columna "label" (0=sober, 1=drunk)
"""

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from pathlib import Path
from typing import Optional
import urllib.request


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"
FACE_LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def download_model(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Descargando modelo: {dest.name} ...")
        urllib.request.urlretrieve(url, str(dest))
        print(f"Modelo descargado.")


# ---------------------------------------------------------------------------
# Landmarks clave seleccionados de la malla de 478 puntos de MediaPipe
# Se usan puntos anatomicamente relevantes para detectar cambios por alcohol:
# ojos (apertura, droopiness), boca (relajacion muscular), mejillas, frente
# ---------------------------------------------------------------------------

KEY_LANDMARKS = [
    # Contorno ojo izquierdo
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    # Contorno ojo derecho
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    # Iris
    468,
    473,
    # Nariz
    1,
    2,
    98,
    327,
    # Boca (contorno exterior)
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    # Boca (contorno interior)
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    # Mejillas
    116,
    123,
    147,
    213,
    345,
    352,
    376,
    433,
    # Frente y contorno facial
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    103,
    67,
    109,
    10,
    152,
    # Menton
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
]
KEY_LANDMARKS = sorted(set(KEY_LANDMARKS))

# Pares de landmarks para calcular distancias de lineas faciales
# Captura cambios en apertura de ojos, boca, y contorno facial
LANDMARK_PAIRS = [
    # Apertura vertical del ojo izquierdo (droopiness)
    (145, 159),
    (144, 160),
    (153, 158),
    (154, 157),
    # Apertura vertical del ojo derecho
    (374, 386),
    (373, 387),
    (380, 385),
    (381, 384),
    # Ancho del ojo izquierdo
    (33, 133),
    # Ancho del ojo derecho
    (362, 263),
    # Apertura vertical de la boca
    (13, 14),
    (82, 87),
    (312, 317),
    # Ancho de la boca
    (61, 291),
    (78, 308),
    # Nariz a boca
    (1, 13),
    (1, 0),
    # Distancia entre ojos
    (33, 263),
    # Distancia ojo-boca
    (159, 13),
    (386, 13),
    # Alto del contorno facial (frente a menton)
    (10, 152),
    # Anchura de mejillas
    (116, 345),
    (123, 352),
    # Angulo de las cejas (inclinacion por relajacion muscular)
    (70, 63),
    (107, 55),  # ceja izquierda
    (300, 293),
    (336, 285),  # ceja derecha
]

# Indices de landmarks en la frente (para el feature de enrojecimiento)
FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 103, 67, 109]


# ---------------------------------------------------------------------------
# Extractor de landmarks
# ---------------------------------------------------------------------------


class LandmarkExtractor:
    def __init__(self):
        download_model(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def extract(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae los 478 landmarks de una imagen ya alineada.

        Returns:
            Array (478, 2) de coordenadas en pixeles, o None si no se detecta cara
        """
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        lms = result.face_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
        return pts


# ---------------------------------------------------------------------------
# Extraccion de features por imagen
# ---------------------------------------------------------------------------


def extract_features(image_bgr: np.ndarray, landmarks: np.ndarray) -> dict:
    """
    Extrae el vector de features de una cara alineada.

    Args:
        image_bgr: imagen de la cara (224x224 BGR)
        landmarks: array (478, 2) de landmarks en pixeles

    Returns:
        Diccionario feature_name -> valor float
    """
    h, w = image_bgr.shape[:2]
    features = {}

    # ------------------------------------------------------------------
    # 1. Posiciones X,Y normalizadas de landmarks clave
    # Normalizadas al rango [0,1] por el tamano de la imagen
    # ------------------------------------------------------------------
    for idx in KEY_LANDMARKS:
        x_norm = float(landmarks[idx, 0]) / w
        y_norm = float(landmarks[idx, 1]) / h
        features[f"lm{idx:03d}_x"] = x_norm
        features[f"lm{idx:03d}_y"] = y_norm

    # ------------------------------------------------------------------
    # 2. Vectores desde el centroide al landmark (distancia y angulo)
    # El centroide es el promedio de los KEY_LANDMARKS seleccionados.
    # El alcohol relaja musculos y desplaza landmarks, cambiando estos vectores.
    # ------------------------------------------------------------------
    key_pts = landmarks[KEY_LANDMARKS]
    centroid = key_pts.mean(axis=0)

    for idx in KEY_LANDMARKS:
        dx = float(landmarks[idx, 0] - centroid[0])
        dy = float(landmarks[idx, 1] - centroid[1])
        dist = float(np.sqrt(dx * dx + dy * dy)) / w  # normalizado por ancho
        angle = float(np.arctan2(dy, dx))  # en radianes [-pi, pi]
        features[f"vec{idx:03d}_dist"] = dist
        features[f"vec{idx:03d}_angle"] = angle

    # ------------------------------------------------------------------
    # 3. Distancias entre pares de landmarks (lineas faciales)
    # Captura apertura de ojos, boca, cambios en mejillas y frente.
    # Normalizadas por la distancia inter-ocular para ser invariantes al tamano.
    # ------------------------------------------------------------------
    left_eye_center = landmarks[33]
    right_eye_center = landmarks[263]
    interocular_dist = float(np.linalg.norm(right_eye_center - left_eye_center))
    if interocular_dist < 1:
        interocular_dist = 1.0

    for i, (a, b) in enumerate(LANDMARK_PAIRS):
        dist = float(np.linalg.norm(landmarks[a] - landmarks[b]))
        features[f"line{i:02d}_dist"] = dist / interocular_dist

    # ------------------------------------------------------------------
    # 4. Color de frente (enrojecimiento por alcohol)
    # Se extrae la region de la frente usando los landmarks de frente,
    # y se calcula el canal R promedio normalizado en LAB.
    # ------------------------------------------------------------------
    forehead_pts = landmarks[FOREHEAD_LANDMARKS].astype(np.int32)
    x_min = max(0, int(forehead_pts[:, 0].min()))
    x_max = min(w - 1, int(forehead_pts[:, 0].max()))
    y_min = max(0, int(forehead_pts[:, 1].min()))
    y_max = min(h - 1, int(forehead_pts[:, 1].max()))

    if x_max > x_min and y_max > y_min:
        forehead_region = image_bgr[y_min:y_max, x_min:x_max]
        lab = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2LAB)
        features["forehead_L"] = float(np.mean(lab[:, :, 0])) / 255.0
        features["forehead_a"] = float(np.mean(lab[:, :, 1])) / 255.0  # verde-rojo
        features["forehead_b"] = float(np.mean(lab[:, :, 2])) / 255.0  # azul-amarillo
        # Canal rojo en BGR original
        features["forehead_R"] = float(np.mean(forehead_region[:, :, 2])) / 255.0
    else:
        features["forehead_L"] = 0.5
        features["forehead_a"] = 0.5
        features["forehead_b"] = 0.5
        features["forehead_R"] = 0.5

    return features


# ---------------------------------------------------------------------------
# Procesamiento del dataset completo
# ---------------------------------------------------------------------------


def process_dataset(
    sober_dir: str,
    drunk_dir: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Recorre las carpetas de imagenes, extrae features de cada imagen
    y guarda el resultado en un CSV.

    Args:
        sober_dir: carpeta con imagenes de personas sobrias (224x224)
        drunk_dir: carpeta con imagenes de personas ebrias (224x224)
        output_csv: ruta del CSV de salida

    Returns:
        DataFrame con todas las features y la columna "label"
    """
    extractor = LandmarkExtractor()

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    rows = []
    stats = {"sober_ok": 0, "sober_fail": 0, "drunk_ok": 0, "drunk_fail": 0}

    def process_folder(folder: str, label: int, label_name: str):
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Carpeta no encontrada: {folder_path}")
            return

        images = sorted(
            [p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )

        print(f"\nProcesando {len(images)} imagenes de {label_name}...")

        for i, img_path in enumerate(images):
            image = cv2.imread(str(img_path))
            if image is None:
                stats[f"{label_name}_fail"] += 1
                continue

            landmarks = extractor.extract(image)
            if landmarks is None or len(landmarks) < 478:
                stats[f"{label_name}_fail"] += 1
                continue

            feat = extract_features(image, landmarks)
            feat["image"] = img_path.name
            feat["label"] = label
            rows.append(feat)
            stats[f"{label_name}_ok"] += 1

            if (i + 1) % 200 == 0:
                print(
                    f"  [{i + 1}/{len(images)}] OK: {stats[f'{label_name}_ok']} | Sin landmarks: {stats[f'{label_name}_fail']}"
                )

        print(
            f"  Completado -> OK: {stats[f'{label_name}_ok']} | Sin landmarks: {stats[f'{label_name}_fail']}"
        )

    process_folder(sober_dir, label=0, label_name="sober")
    process_folder(drunk_dir, label=1, label_name="drunk")

    if not rows:
        raise RuntimeError(
            "No se extrajo ningun feature. Verifica las carpetas de entrada."
        )

    df = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nCSV guardado: {output_path}")
    print(f"Total filas: {len(df)} | Features por fila: {len(df.columns) - 2}")
    print(f"Clase 0 (sober): {(df['label'] == 0).sum()}")
    print(f"Clase 1 (drunk): {(df['label'] == 1).sum()}")

    return df
