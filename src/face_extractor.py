"""
Modulo para detectar, alinear y extraer rostros de videos e imagenes.

Backend principal: MediaPipe Tasks (FaceDetector + FaceLandmarker)
- Deteccion precisa con modelo BlazeFace
- Validacion de landmarks: verifica que los puntos formen una cara real
- Verificacion de completitud: descarta caras que tocan el borde de la imagen
- Alineacion facial usando iris para normalizar rotacion
- Verificacion de calidad: nitidez, brillo y contraste minimos
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional
import os


# ---------------------------------------------------------------------------
# Rutas de modelos (se descargan automaticamente si no existen)
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"

FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

FACE_DETECTOR_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"


def download_model(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Descargando modelo: {dest.name} ...")
        urllib.request.urlretrieve(url, str(dest))
        print(f"Modelo descargado: {dest.name}")


# ---------------------------------------------------------------------------
# Indices de landmarks relevantes (malla de 478 puntos de MediaPipe)
# ---------------------------------------------------------------------------

# Iris (disponibles con FaceLandmarker)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Esquinas de los ojos (fallback si no hay iris)
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Proporcion minima esperada entre distancia ojo-ojo y alto de cara
MIN_EYE_DISTANCE_RATIO = 0.15
MAX_EYE_DISTANCE_RATIO = 0.80


# ---------------------------------------------------------------------------
# Detector principal: MediaPipe Tasks
# ---------------------------------------------------------------------------


class MediaPipeDetector:
    """
    Detector usando MediaPipe Tasks API (>= 0.10.x).
    Combina FaceDetector (bounding box) y FaceLandmarker (478 puntos).
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        import mediapipe as mp

        self.mp = mp
        self.min_confidence = min_detection_confidence

        download_model(FACE_DETECTOR_MODEL_URL, FACE_DETECTOR_MODEL_PATH)
        download_model(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        det_options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_DETECTOR_MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = FaceDetector.create_from_options(det_options)

        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        lm_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
        )
        self.landmarker = FaceLandmarker.create_from_options(lm_options)

    def _to_mp_image(self, bgr_image: np.ndarray):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

    def detect(self, image: np.ndarray) -> List[dict]:
        h, w = image.shape[:2]
        mp_img = self._to_mp_image(image)

        det_result = self.detector.detect(mp_img)
        lm_result = self.landmarker.detect(mp_img)

        lm_map = {}
        for face_lms in lm_result.face_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            lm_map[(round(cx), round(cy))] = pts

        detections = []
        for det in det_result.detections:
            bb = det.bounding_box
            x, y, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
            score = det.categories[0].score

            box_cx = x + bw / 2
            box_cy = y + bh / 2
            best_pts = None
            best_dist = float("inf")
            for (lcx, lcy), pts in lm_map.items():
                dist = (lcx - box_cx) ** 2 + (lcy - box_cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_pts = pts

            detections.append(
                {
                    "box": [x, y, bw, bh],
                    "confidence": score,
                    "landmarks": best_pts,
                }
            )

        return detections


# ---------------------------------------------------------------------------
# Fallback: OpenCV
# ---------------------------------------------------------------------------


class OpenCVDetector:
    """Detector fallback con Haar Cascade."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("No se pudo cargar haarcascade_frontalface_alt2.xml")

    def detect(self, image: np.ndarray) -> List[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5, minSize=(80, 80)
        )
        results = []
        for x, y, w, h in faces:
            results.append(
                {
                    "box": [int(x), int(y), int(w), int(h)],
                    "confidence": 0.8,
                    "landmarks": None,
                }
            )
        return results


# ---------------------------------------------------------------------------
# Validacion de landmarks
# ---------------------------------------------------------------------------


def validate_landmarks(landmarks: Optional[np.ndarray], box: List[int]) -> bool:
    """
    Verifica que los landmarks formen una cara anatomicamente plausible.

    Comprueba:
    1. Distancia entre ojos proporcional al alto de la cara
    2. Nariz ubicada verticalmente entre los ojos y el menton
    3. Ojos en posicion horizontal similar (no excesivamente inclinados)

    Returns:
        True si los landmarks son consistentes con una cara real
    """
    if landmarks is None or len(landmarks) < 468:
        # Sin landmarks no podemos validar, dejamos pasar
        return True

    left_eye = landmarks[[LEFT_EYE_INNER, LEFT_EYE_OUTER]].mean(axis=0)
    right_eye = landmarks[[RIGHT_EYE_INNER, RIGHT_EYE_OUTER]].mean(axis=0)
    nose_tip = landmarks[1]
    chin = landmarks[152]
    forehead = landmarks[10]

    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    face_height = float(np.linalg.norm(forehead - chin))

    if face_height < 1:
        return False

    eye_ratio = eye_distance / face_height

    # Ojos deben estar separados entre 15% y 80% del alto de la cara
    if not (MIN_EYE_DISTANCE_RATIO <= eye_ratio <= MAX_EYE_DISTANCE_RATIO):
        return False

    # Nariz debe estar verticalmente entre los ojos y el menton
    eye_mid_y = float((left_eye[1] + right_eye[1]) / 2)
    if not (eye_mid_y < float(nose_tip[1]) < float(chin[1])):
        return False

    # Ojos no deben estar muy inclinados
    eye_dy = abs(float(right_eye[1]) - float(left_eye[1]))
    if eye_dy > eye_distance * 0.5:
        return False

    return True


# ---------------------------------------------------------------------------
# Alineacion facial
# ---------------------------------------------------------------------------


def align_face(
    image: np.ndarray,
    landmarks: Optional[np.ndarray],
    output_size: int = 224,
) -> Optional[np.ndarray]:
    """
    Alinea el rostro usando los centros de los ojos.
    Si no hay landmarks, hace un resize simple.
    """
    if landmarks is None or len(landmarks) < 478:
        return cv2.resize(
            image, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4
        )

    left_eye = landmarks[LEFT_IRIS_CENTER]
    right_eye = landmarks[RIGHT_IRIS_CENTER]

    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = (
        float((left_eye[0] + right_eye[0]) / 2),
        float((left_eye[1] + right_eye[1]) / 2),
    )

    desired_left_eye_x = 0.30
    desired_eye_y = 0.35
    desired_dist = output_size * (1.0 - 2 * desired_left_eye_x)

    current_dist = np.sqrt(dx * dx + dy * dy)
    scale = desired_dist / (current_dist + 1e-6)

    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    M[0, 2] += output_size * 0.5 - eye_center[0]
    M[1, 2] += output_size * desired_eye_y - eye_center[1]

    aligned = cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


# ---------------------------------------------------------------------------
# FaceExtractor: clase principal
# ---------------------------------------------------------------------------


class FaceExtractor:
    """
    Extrae rostros de videos e imagenes con alta calidad.

    Flujo por cada deteccion:
        1. Verificar confianza minima del detector
        2. Validar que los landmarks formen una cara anatomicamente plausible
        3. Verificar que la cara este COMPLETAMENTE dentro de la imagen (sin tolerancia)
        4. Recortar con padding
        5. Alinear con landmarks
        6. Verificar calidad (nitidez, brillo, contraste)
        7. Guardar
    """

    def __init__(
        self,
        detector_type: str = "mediapipe",
        output_size: int = 224,
        min_confidence: float = 0.5,
        padding: float = 0.25,
        min_face_size: int = 80,
        quality_check: bool = True,
        min_sharpness: float = 60.0,
    ):
        """
        Args:
            detector_type: "mediapipe" (recomendado) u "opencv" (fallback)
            output_size: tamano cuadrado de salida en pixeles (entero)
            min_confidence: confianza minima del detector (0.0 a 1.0)
            padding: fraccion de padding alrededor del box (0.25 = 25%)
            min_face_size: tamano minimo del bounding box en pixeles
            quality_check: si True, descarta imagenes borrosas/oscuras
            min_sharpness: varianza del Laplaciano minima para aceptar
        """
        self.output_size = int(output_size)
        self.min_confidence = min_confidence
        self.padding = padding
        self.min_face_size = min_face_size
        self.quality_check = quality_check
        self.min_sharpness = min_sharpness

        if detector_type == "mediapipe":
            try:
                self.detector = MediaPipeDetector(
                    min_detection_confidence=min_confidence
                )
                print("Detector: MediaPipe Tasks (BlazeFace + FaceLandmarker)")
            except Exception as e:
                print(f"MediaPipe no disponible ({e}), usando OpenCV como fallback")
                self.detector = OpenCVDetector()
        elif detector_type == "opencv":
            self.detector = OpenCVDetector()
            print("Detector: OpenCV Haar Cascade (fallback)")
        else:
            raise ValueError(
                f"detector_type debe ser 'mediapipe' u 'opencv', no '{detector_type}'"
            )

    def _is_face_complete(self, box: List[int], img_h: int, img_w: int) -> bool:
        """
        Verifica que la cara CON padding este completamente dentro de la imagen.
        Sin tolerancia: cualquier cara que toque el borde se rechaza.
        Esto evita caras cortadas como la imagen de Ellen DeGeneres.
        """
        x, y, w, h = box
        side = max(w, h)
        pad = int(side * self.padding)

        x1 = x - pad
        y1 = y - pad
        x2 = x + w + pad
        y2 = y + h + pad

        return x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h

    def _crop_with_padding(
        self, image: np.ndarray, box: List[int]
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int], bool]:
        """
        Recorta el rostro con padding.
        Llamar solo despues de _is_face_complete.
        """
        x, y, w, h = box
        img_h, img_w = image.shape[:2]

        if w < self.min_face_size or h < self.min_face_size:
            return None, (0, 0), False

        side = max(w, h)
        pad = int(side * self.padding)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None, (0, 0), False

        return crop, (x1, y1), True

    def _check_quality(self, face: np.ndarray) -> bool:
        """
        Verifica nitidez, brillo y contraste minimos.
        """
        if not self.quality_check:
            return True
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        return (
            sharpness >= self.min_sharpness and 25 < brightness < 235 and contrast > 15
        )

    def process_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detecta y extrae todos los rostros validos de un frame.

        Returns:
            Lista de imagenes de rostros alineadas (output_size x output_size)
        """
        img_h, img_w = frame.shape[:2]
        detections = self.detector.detect(frame)
        results = []

        for det in detections:
            if det["confidence"] < self.min_confidence:
                continue

            if not validate_landmarks(det["landmarks"], det["box"]):
                continue

            if not self._is_face_complete(det["box"], img_h, img_w):
                continue

            crop, crop_origin, valid = self._crop_with_padding(frame, det["box"])
            if not valid or crop is None:
                continue

            aligned_lms = None
            if det["landmarks"] is not None:
                ox, oy = crop_origin
                aligned_lms = det["landmarks"] - np.array([ox, oy], dtype=np.float32)

            face = align_face(crop, aligned_lms, self.output_size)
            if face is None:
                continue

            if not self._check_quality(face):
                continue

            results.append(face)

        return results

    def process_image(self, image_path: str) -> List[np.ndarray]:
        """Procesa una imagen estatica y retorna los rostros extraidos."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo leer: {image_path}")
            return []
        return self.process_frame(image)

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        sample_interval: float = 0.3,
        max_faces_per_video: int = 100,
    ) -> int:
        """
        Procesa un video, extrae rostros y los guarda.

        Args:
            video_path: ruta al video
            output_dir: directorio donde guardar los rostros
            sample_interval: segundos entre frames analizados (default 0.3s)
            max_faces_per_video: limite de rostros validos por video

        Returns:
            Numero de rostros guardados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: no se puede abrir {video_path}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, int(fps * sample_interval))

        print(f"Procesando: {video_name}")
        print(
            f"  FPS: {fps:.1f} | Frames: {total_frames} | Muestra cada: {sample_rate} frames ({sample_interval}s)"
        )

        face_count = 0
        frame_idx = 0
        frames_analyzed = 0
        rejected_border = 0
        rejected_quality = 0
        rejected_landmarks = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            frames_analyzed += 1
            img_h, img_w = frame.shape[:2]
            detections = self.detector.detect(frame)

            for det in detections:
                if face_count >= max_faces_per_video:
                    break

                if det["confidence"] < self.min_confidence:
                    continue

                if not validate_landmarks(det["landmarks"], det["box"]):
                    rejected_landmarks += 1
                    continue

                if not self._is_face_complete(det["box"], img_h, img_w):
                    rejected_border += 1
                    continue

                crop, crop_origin, valid = self._crop_with_padding(frame, det["box"])
                if not valid or crop is None:
                    continue

                aligned_lms = None
                if det["landmarks"] is not None:
                    ox, oy = crop_origin
                    aligned_lms = det["landmarks"] - np.array(
                        [ox, oy], dtype=np.float32
                    )

                face = align_face(crop, aligned_lms, self.output_size)
                if face is None:
                    continue

                if not self._check_quality(face):
                    rejected_quality += 1
                    continue

                filename = f"{video_name}_f{frame_idx:06d}_r{face_count:04d}.jpg"
                cv2.imwrite(
                    str(output_dir / filename),
                    face,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                face_count += 1

            if face_count >= max_faces_per_video:
                print(f"  Limite alcanzado: {max_faces_per_video} rostros")
                break

            frame_idx += 1

        cap.release()
        print(f"  Frames analizados:      {frames_analyzed}")
        print(f"  Rechazados (borde):     {rejected_border}")
        print(f"  Rechazados (calidad):   {rejected_quality}")
        print(f"  Rechazados (landmarks): {rejected_landmarks}")
        print(f"  Rostros guardados:      {face_count}")
        return face_count
