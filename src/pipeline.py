"""
Pipeline end-to-end de SoberLens.
Corregido para evitar data leakage entre augmentaciones y el conjunto de prueba.

Orden correcto:
    1. Separar imagenes ORIGINALES en train (80%) y test (20%)
    2. Data augmentation SOLO sobre las imagenes de train
    3. Extraer features del conjunto train (originales + aumentadas)
    4. Extraer features del conjunto test (solo originales, sin aumentar)
    5. Entrenar y evaluar el clasificador

De esta forma el modelo nunca ve durante entrenamiento variantes de
las mismas imagenes que se usan para evaluarlo.

Uso:
    python pipeline.py

IMPORTANTE: Antes de correr, borra las imagenes _aug que se generaron
en la ejecucion anterior, o configura SOBER_IMAGES_DIR y DRUNK_IMAGES_DIR
apuntando a las carpetas con solo las imagenes originales.
"""

import shutil
import random
from pathlib import Path

from augmentation import augment_folder
from feature_extractor import process_dataset
from train import run_training_from_csvs


# =============================================================================
# CONFIGURACION
# =============================================================================

# Carpetas con las imagenes originales (224x224, sin _aug)
SOBER_IMAGES_DIR = "../output/sober"
DRUNK_IMAGES_DIR = "../output/drunk"

# Carpetas temporales donde se organizara el split
TRAIN_SOBER_DIR = "../output/split/train/sober"
TRAIN_DRUNK_DIR = "../output/split/train/drunk"
TEST_SOBER_DIR = "../output/split/test/sober"
TEST_DRUNK_DIR = "../output/split/test/drunk"

# CSVs de features separados
TRAIN_CSV = "../output/train_features.csv"
TEST_CSV = "../output/test_features.csv"

# Modelo
MODELS_OUTPUT_DIR = "../output/models"

TEST_SIZE = 0.20
AUGMENTATIONS_PER_IMAGE = 4
RANDOM_SEED = 42

# =============================================================================
# FIN CONFIGURACION
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_original_images(folder: str):
    """Retorna solo las imagenes originales (sin _aug en el nombre)."""
    return sorted(
        [
            p
            for p in Path(folder).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and "_aug" not in p.stem
        ]
    )


def split_images(images, test_size: float, seed: int):
    """Divide una lista de imagenes en train y test de forma reproducible."""
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_size))
    return shuffled[n_test:], shuffled[:n_test]


def copy_images(images, dest_dir: str):
    """Copia una lista de archivos a dest_dir."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        shutil.copy2(str(img_path), str(dest / img_path.name))


def clean_split_dirs():
    """Elimina las carpetas temporales del split anterior si existen."""
    for d in [TRAIN_SOBER_DIR, TRAIN_DRUNK_DIR, TEST_SOBER_DIR, TEST_DRUNK_DIR]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(str(p))


def main():
    print("=" * 60)
    print("PIPELINE END-TO-END - SoberLens (sin data leakage)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Paso 0: Limpiar carpetas de split anteriores
    # ------------------------------------------------------------------
    print("\nLimpiando carpetas temporales de split anterior...")
    clean_split_dirs()

    # ------------------------------------------------------------------
    # Etapa 1: Split de imagenes originales en train/test
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("ETAPA 1: SPLIT TRAIN/TEST DE IMAGENES ORIGINALES")
    print("#" * 60)

    sober_images = get_original_images(SOBER_IMAGES_DIR)
    drunk_images = get_original_images(DRUNK_IMAGES_DIR)

    print(f"Imagenes originales encontradas:")
    print(f"  Sober: {len(sober_images)}")
    print(f"  Drunk: {len(drunk_images)}")

    if not sober_images or not drunk_images:
        raise RuntimeError(
            "No se encontraron imagenes originales. "
            "Verifica SOBER_IMAGES_DIR y DRUNK_IMAGES_DIR."
        )

    sober_train, sober_test = split_images(sober_images, TEST_SIZE, RANDOM_SEED)
    drunk_train, drunk_test = split_images(drunk_images, TEST_SIZE, RANDOM_SEED)

    print(f"\nSplit (80/20):")
    print(f"  Sober  -> train: {len(sober_train)} | test: {len(sober_test)}")
    print(f"  Drunk  -> train: {len(drunk_train)} | test: {len(drunk_test)}")

    print("\nCopiando imagenes a carpetas de split...")
    copy_images(sober_train, TRAIN_SOBER_DIR)
    copy_images(drunk_train, TRAIN_DRUNK_DIR)
    copy_images(sober_test, TEST_SOBER_DIR)
    copy_images(drunk_test, TEST_DRUNK_DIR)
    print("Copia completada.")

    # ------------------------------------------------------------------
    # Etapa 2: Data augmentation SOLO sobre el conjunto de entrenamiento
    # ------------------------------------------------------------------
    if AUGMENTATIONS_PER_IMAGE > 0:
        print("\n" + "#" * 60)
        print("ETAPA 2: DATA AUGMENTATION (solo en train)")
        print("#" * 60)
        print("El conjunto de TEST no se toca.")

        sober_aug = augment_folder(
            TRAIN_SOBER_DIR, AUGMENTATIONS_PER_IMAGE, RANDOM_SEED
        )
        drunk_aug = augment_folder(
            TRAIN_DRUNK_DIR, AUGMENTATIONS_PER_IMAGE, RANDOM_SEED
        )

        print(f"\nAugmentation completada:")
        print(f"  Sober train: {len(sober_train)} originales + {sober_aug} aumentadas")
        print(f"  Drunk train: {len(drunk_train)} originales + {drunk_aug} aumentadas")
    else:
        print("\nAugmentation desactivada.")

    # ------------------------------------------------------------------
    # Etapa 3: Extraccion de features
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("ETAPA 3: EXTRACCION DE FEATURES")
    print("#" * 60)

    print("\n[3a] Features del conjunto de ENTRENAMIENTO (originales + aumentadas):")
    df_train = process_dataset(
        sober_dir=TRAIN_SOBER_DIR,
        drunk_dir=TRAIN_DRUNK_DIR,
        output_csv=TRAIN_CSV,
    )

    print("\n[3b] Features del conjunto de PRUEBA (solo originales):")
    df_test = process_dataset(
        sober_dir=TEST_SOBER_DIR,
        drunk_dir=TEST_DRUNK_DIR,
        output_csv=TEST_CSV,
    )

    print(f"\nResumen de features:")
    print(f"  Train: {len(df_train)} muestras")
    print(f"  Test:  {len(df_test)} muestras")

    # ------------------------------------------------------------------
    # Etapa 4: Entrenamiento
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("ETAPA 4: ENTRENAMIENTO DEL CLASIFICADOR")
    print("#" * 60)

    clf, scaler, feature_cols, accuracy = run_training_from_csvs(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        models_output_dir=MODELS_OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"Train: {len(df_train)} muestras (con augmentation)")
    print(f"Test:  {len(df_test)} muestras (solo originales, sin aumentar)")
    print(f"Features por muestra: {len(feature_cols)}")
    print(f"Accuracy real (sin leakage): {accuracy * 100:.2f}%")
    print(f"\nArchivos generados:")
    print(f"  {TRAIN_CSV}")
    print(f"  {TEST_CSV}")
    print(f"  {MODELS_OUTPUT_DIR}/model.pkl")
    print(f"  {MODELS_OUTPUT_DIR}/scaler.pkl")
    print(f"  {MODELS_OUTPUT_DIR}/features.txt")
    print(f"  {MODELS_OUTPUT_DIR}/metadata.txt")


if __name__ == "__main__":
    main()
