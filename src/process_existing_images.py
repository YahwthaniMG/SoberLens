"""
Script para procesar un dataset de imagenes existentes.
Detecta rostros, los alinea y guarda en tamano estandar (224x224).

Uso:
    python process_existing_images.py
"""

import os
from pathlib import Path
import cv2

from face_extractor import FaceExtractor


# =============================================================================
# CONFIGURACION
# =============================================================================

INPUT_SOBER = "../data/dataset_images/sober"
INPUT_DRUNK = "../data/dataset_images/drunk"

OUTPUT_SOBER = "../output/sober"
OUTPUT_DRUNK = "../output/drunk"


DETECTOR_TYPE = "mediapipe"
FACE_OUTPUT_SIZE = 224
MIN_CONFIDENCE = 0.5

# Extensiones de imagen aceptadas
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# =============================================================================
# FIN CONFIGURACION
# =============================================================================


def process_folder(
    input_dir: str,
    output_dir: str,
    extractor: FaceExtractor,
    prefix: str,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images_read": 0,
        "images_with_face": 0,
        "images_skipped": 0,
        "faces_saved": 0,
    }

    if not input_dir.exists():
        print(f"Carpeta no encontrada: {input_dir}")
        return stats

    image_files = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if not image_files:
        print(f"Sin imagenes en: {input_dir}")
        return stats

    print(f"Procesando {len(image_files)} imagenes desde: {input_dir}")

    for idx, img_path in enumerate(image_files):
        stats["images_read"] += 1

        faces = extractor.process_image(str(img_path))

        if not faces:
            stats["images_skipped"] += 1
            if (idx + 1) % 20 == 0:
                print(f"  [{idx + 1}/{len(image_files)}] sin rostros: {img_path.name}")
            continue

        stats["images_with_face"] += 1

        for face_idx, face in enumerate(faces):
            filename = f"{prefix}_{idx:05d}_f{face_idx:02d}.jpg"
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["faces_saved"] += 1

        if (idx + 1) % 50 == 0:
            print(
                f"  [{idx + 1}/{len(image_files)}] rostros guardados hasta ahora: {stats['faces_saved']}"
            )

    return stats


def main():
    print("=" * 60)
    print("PROCESAMIENTO DE DATASET DE IMAGENES EXISTENTES")
    print("=" * 60)

    print(f"\nInicializando detector: {DETECTOR_TYPE}")
    extractor = FaceExtractor(
        detector_type=DETECTOR_TYPE,
        output_size=FACE_OUTPUT_SIZE,
        min_confidence=MIN_CONFIDENCE,
        padding=0.3,
        quality_check=True,
        min_sharpness=60.0,
    )

    print("\n" + "-" * 60)
    print("SOBRIOS")
    print("-" * 60)
    sober_stats = process_folder(INPUT_SOBER, OUTPUT_SOBER, extractor, "sober")

    print("\n" + "-" * 60)
    print("EBRIOS")
    print("-" * 60)
    drunk_stats = process_folder(INPUT_DRUNK, OUTPUT_DRUNK, extractor, "drunk")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    print("\nSOBRIOS:")
    print(f"  Imagenes leidas:       {sober_stats['images_read']}")
    print(f"  Con rostro detectado:  {sober_stats['images_with_face']}")
    print(f"  Sin rostro / descarte: {sober_stats['images_skipped']}")
    print(f"  Rostros guardados:     {sober_stats['faces_saved']}")

    print("\nEBRIOS:")
    print(f"  Imagenes leidas:       {drunk_stats['images_read']}")
    print(f"  Con rostro detectado:  {drunk_stats['images_with_face']}")
    print(f"  Sin rostro / descarte: {drunk_stats['images_skipped']}")
    print(f"  Rostros guardados:     {drunk_stats['faces_saved']}")

    total = sober_stats["faces_saved"] + drunk_stats["faces_saved"]
    print(f"\nTOTAL ROSTROS GUARDADOS: {total}")
    print(f"\nSalida:")
    print(f"  Sobrios: {os.path.abspath(OUTPUT_SOBER)}")
    print(f"  Ebrios:  {os.path.abspath(OUTPUT_DRUNK)}")


if __name__ == "__main__":
    main()
