"""
Entrenamiento del clasificador sobrio/ebrio.

Recibe dos CSVs ya separados (train y test) desde pipeline.py,
lo que garantiza que no haya data leakage por augmentation.

Pipeline:
    1. Carga train_features.csv y test_features.csv
    2. Normaliza con StandardScaler (fit solo en train, transform en ambos)
    3. Compara clasificadores con 10-fold CV sobre el train
    4. Re-entrena el mejor con todo el train
    5. Evalua en test (imagenes originales sin aumentar)
    6. Guarda modelo, scaler y metadata

Basado en DrunkSelfie (Willoughby et al., 2019).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =============================================================================
# CONFIGURACION (usada cuando se corre train.py directamente)
# =============================================================================

TRAIN_CSV = "../output/train_features.csv"
TEST_CSV = "../output/test_features.csv"
MODELS_OUTPUT_DIR = "../output/models"

RANDOM_SEED = 42
CV_FOLDS = 10

# =============================================================================
# FIN CONFIGURACION
# =============================================================================


def load_csv(csv_path: str):
    """
    Carga un CSV de features.

    Returns:
        X (array de features), y (labels), feature_cols (nombres de columnas)
    """
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in ["image", "label"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    return X, y, feature_cols


def build_classifiers(seed: int):
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=seed,
        ),
        "SVM RBF": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        ),
        "SVM Lineal": SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            random_state=seed,
        ),
    }


def evaluate_classifiers_cv(X_train, y_train, classifiers: dict, seed: int) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    results = {}

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({CV_FOLDS} folds) — conjunto de entrenamiento")
    print(f"Nota: CV aqui mide generalizacion entre augmentaciones del mismo")
    print(f"conjunto train. La evaluacion real es sobre el test set.")
    print(f"{'='*60}")

    for name, clf in classifiers.items():
        scores = cross_val_score(
            clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        mean_acc = float(scores.mean())
        std_acc = float(scores.std())
        results[name] = mean_acc
        print(f"  {name:<25} {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")

    return results


def train_best(X_train, y_train, classifiers: dict, cv_results: dict):
    best_name = max(cv_results, key=cv_results.get)
    best_clf = classifiers[best_name]
    print(f"\nMejor por CV: {best_name} ({cv_results[best_name] * 100:.2f}%)")
    print("Entrenando con el conjunto completo de train...")
    best_clf.fit(X_train, y_train)
    print("Listo.")
    return best_name, best_clf


def evaluate_on_test(clf, X_test, y_test, clf_name: str):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"EVALUACION FINAL — conjunto de PRUEBA (originales sin aumentar)")
    print(f"{'='*60}")
    print(f"Clasificador: {clf_name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"\nReporte por clase:")
    print(classification_report(y_test, y_pred, target_names=["sober", "drunk"]))
    print("Matriz de confusion:")
    print(f"              Predicho sober  Predicho drunk")
    print(f"  Real sober       {cm[0, 0]:5d}          {cm[0, 1]:5d}")
    print(f"  Real drunk       {cm[1, 0]:5d}          {cm[1, 1]:5d}")

    tn, fp, fn, tp = cm.ravel()
    print(
        f"\nFalsos negativos (ebrio -> sober): {fn}  <- ebrios que el modelo no detecta"
    )
    print(
        f"Falsos positivos (sober -> drunk): {fp}  <- sobrios clasificados como ebrios"
    )

    return acc, cm


def save_model(clf, scaler, feature_cols, output_dir, clf_name, train_acc, test_acc):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, output_dir / "model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")

    with open(output_dir / "features.txt", "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    with open(output_dir / "metadata.txt", "w") as f:
        f.write(f"Clasificador: {clf_name}\n")
        f.write(f"CV accuracy (train): {train_acc * 100:.2f}%\n")
        f.write(f"Accuracy real (test, sin leakage): {test_acc * 100:.2f}%\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Clases: 0=sober, 1=drunk\n")
        f.write(f"Nota: test set = imagenes originales sin augmentation\n")

    print(f"\nModelo guardado en: {output_dir}")
    print(f"  model.pkl     -> {clf_name}")
    print(f"  scaler.pkl    -> StandardScaler")
    print(f"  features.txt  -> {len(feature_cols)} features")
    print(f"  metadata.txt  -> metricas y configuracion")


def run_training_from_csvs(
    train_csv: str,
    test_csv: str,
    models_output_dir: str,
    seed: int = RANDOM_SEED,
):
    """
    Funcion principal llamada por pipeline.py.
    Recibe CSVs ya separados (sin leakage).
    """
    print("=" * 60)
    print("PIPELINE DE ENTRENAMIENTO - SoberLens")
    print("=" * 60)

    print(f"\nCargando train: {train_csv}")
    X_train, y_train, feature_cols = load_csv(train_csv)
    print(f"  Muestras: {len(X_train)} | Features: {X_train.shape[1]}")
    print(f"  Sober: {(y_train == 0).sum()} | Drunk: {(y_train == 1).sum()}")

    print(f"\nCargando test: {test_csv}")
    X_test, y_test, _ = load_csv(test_csv)
    print(f"  Muestras: {len(X_test)}")
    print(f"  Sober: {(y_test == 0).sum()} | Drunk: {(y_test == 1).sum()}")

    # Normalizar: fit SOLO en train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = build_classifiers(seed)
    cv_results = evaluate_classifiers_cv(X_train_scaled, y_train, classifiers, seed)

    best_name, best_clf = train_best(X_train_scaled, y_train, classifiers, cv_results)
    best_cv_acc = cv_results[best_name]

    test_acc, cm = evaluate_on_test(best_clf, X_test_scaled, y_test, best_name)

    save_model(
        clf=best_clf,
        scaler=scaler,
        feature_cols=feature_cols,
        output_dir=models_output_dir,
        clf_name=best_name,
        train_acc=best_cv_acc,
        test_acc=test_acc,
    )

    return best_clf, scaler, feature_cols, test_acc


if __name__ == "__main__":
    run_training_from_csvs(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        models_output_dir=MODELS_OUTPUT_DIR,
    )
