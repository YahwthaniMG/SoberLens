"""
Ajuste del threshold de decision del clasificador.

Por defecto, un clasificador binario usa threshold=0.5:
    P(ebrio) >= 0.5 -> clasifica como ebrio

Para una app de seguridad, preferimos minimizar falsos negativos
(ebrios que pasan como sobrios), aunque eso incremente levemente
los falsos positivos (sobrios clasificados como ebrios).

Este script:
    1. Carga el modelo y el test set
    2. Calcula las probabilidades del modelo para cada muestra
    3. Evalua todos los thresholds de 0.1 a 0.9 y muestra la tabla completa
    4. Encuentra el threshold optimo segun el criterio elegido
    5. Guarda el threshold en metadata.txt para que lo use el script de inferencia

Uso:
    python tune_threshold.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
)


# =============================================================================
# CONFIGURACION
# =============================================================================

TEST_CSV = "../output/test_features.csv"
MODELS_DIR = "../output/models"

# Criterio para seleccionar el threshold optimo:
#   "recall_drunk"   -> maximiza recall de la clase drunk (minimiza falsos negativos)
#                       prioridad: no dejar pasar ebrios. Recomendado para seguridad.
#   "f1_drunk"       -> maximiza F1 de la clase drunk (balance precision/recall)
#   "f1_macro"       -> maximiza F1 promedio de ambas clases (balance global)
#   "accuracy"       -> maximiza accuracy total
OPTIMIZATION_CRITERION = "recall_drunk"

# Recall minimo aceptable para la clase sober.
# Evita que al bajar el threshold el modelo clasifique a todos como ebrios.
# 0.80 = el modelo debe clasificar correctamente al menos 80% de los sobrios.
MIN_SOBER_RECALL = 0.80

# =============================================================================
# FIN CONFIGURACION
# =============================================================================


def load_test_data(test_csv: str, models_dir: str):
    """Carga el test set y lo escala con el scaler guardado."""
    df = pd.read_csv(test_csv)
    drop_cols = [c for c in ["image", "label"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    scaler = joblib.load(Path(models_dir) / "scaler.pkl")
    X_scaled = scaler.transform(X)

    return X_scaled, y


def evaluate_threshold(y_true, y_prob_drunk, threshold: float) -> dict:
    """Evalua las metricas del modelo para un threshold dado."""
    y_pred = (y_prob_drunk >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    # Manejar caso donde el modelo clasifica todo como una sola clase
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        return None

    total_drunk = fn + tp
    total_sober = tn + fp

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_drunk": tp / total_drunk if total_drunk > 0 else 0.0,
        "precision_drunk": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "f1_drunk": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_sober": tn / total_sober if total_sober > 0 else 0.0,
        "precision_sober": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_threshold_table(results: list):
    """Imprime la tabla completa de metricas por threshold."""
    print(f"\n{'='*90}")
    print(
        f"{'Threshold':>10} | {'Acc':>6} | {'Rec drunk':>9} | {'Prec drunk':>10} | "
        f"{'F1 drunk':>8} | {'Rec sober':>9} | {'FN':>4} | {'FP':>4}"
    )
    print(f"{'-'*90}")

    for r in results:
        marker = " <--" if r.get("is_selected") else ""
        print(
            f"  {r['threshold']:>8.2f} | "
            f"{r['accuracy']*100:>5.1f}% | "
            f"{r['recall_drunk']*100:>8.1f}% | "
            f"{r['precision_drunk']*100:>9.1f}% | "
            f"{r['f1_drunk']*100:>7.1f}% | "
            f"{r['recall_sober']*100:>8.1f}% | "
            f"{r['false_negatives']:>4} | "
            f"{r['false_positives']:>4}"
            f"{marker}"
        )

    print(f"{'='*90}")


def find_optimal_threshold(
    results: list, criterion: str, min_sober_recall: float
) -> dict:
    """
    Encuentra el threshold optimo segun el criterio elegido,
    respetando el recall minimo de sober como restriccion.
    """
    valid = [r for r in results if r["recall_sober"] >= min_sober_recall]

    if not valid:
        print(
            f"\nAdvertencia: ningun threshold cumple recall_sober >= {min_sober_recall:.0%}"
        )
        print("Usando todos los resultados sin filtrar.")
        valid = results

    criterion_map = {
        "recall_drunk": "recall_drunk",
        "f1_drunk": "f1_drunk",
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
    }

    key = criterion_map.get(criterion, "recall_drunk")
    best = max(valid, key=lambda r: r[key])
    return best


def save_threshold(threshold: float, models_dir: str, metrics: dict):
    """Guarda el threshold en el archivo de metadata."""
    metadata_path = Path(models_dir) / "metadata.txt"

    # Leer metadata existente
    existing = ""
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            existing = f.read()

    # Eliminar linea de threshold anterior si existe
    lines = [l for l in existing.splitlines() if not l.startswith("Threshold")]
    lines.append(f"Threshold de decision: {threshold:.4f}")
    lines.append(f"Recall drunk (con threshold): {metrics['recall_drunk']*100:.2f}%")
    lines.append(f"Recall sober (con threshold): {metrics['recall_sober']*100:.2f}%")
    lines.append(f"Falsos negativos (con threshold): {metrics['false_negatives']}")
    lines.append(f"Falsos positivos (con threshold): {metrics['false_positives']}")

    with open(metadata_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nThreshold guardado en: {metadata_path}")


def run_threshold_tuning():
    print("=" * 60)
    print("AJUSTE DE THRESHOLD - SoberLens")
    print("=" * 60)
    print(f"Criterio de optimizacion: {OPTIMIZATION_CRITERION}")
    print(f"Recall minimo de sober:   {MIN_SOBER_RECALL:.0%}")

    # Cargar modelo y datos
    model_path = Path(MODELS_DIR) / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    clf = joblib.load(model_path)

    if not hasattr(clf, "predict_proba"):
        raise RuntimeError(
            "El modelo no soporta probabilidades. "
            "Asegurate de entrenar con probability=True (SVM) o usar Random Forest."
        )

    print(f"\nCargando test set: {TEST_CSV}")
    X_test, y_test = load_test_data(TEST_CSV, MODELS_DIR)
    print(
        f"  Muestras: {len(y_test)} | Sober: {(y_test==0).sum()} | Drunk: {(y_test==1).sum()}"
    )

    # Probabilidades de la clase drunk (clase 1)
    y_prob = clf.predict_proba(X_test)
    y_prob_drunk = y_prob[:, 1]

    # Evaluar todos los thresholds
    thresholds = np.arange(0.10, 0.91, 0.05)
    results = []
    for t in thresholds:
        r = evaluate_threshold(y_test, y_prob_drunk, round(t, 2))
        if r is not None:
            results.append(r)

    # Encontrar optimo
    best = find_optimal_threshold(results, OPTIMIZATION_CRITERION, MIN_SOBER_RECALL)
    best["is_selected"] = True

    # Mostrar tabla
    print_threshold_table(results)

    # Mostrar comparativa
    default = next((r for r in results if abs(r["threshold"] - 0.50) < 0.01), None)

    print(
        f"\nCOMPARATIVA: threshold 0.50 (default) vs {best['threshold']:.2f} (optimo)"
    )
    print(f"{'Metrica':<30} {'Default 0.50':>14} {'Optimo':>14} {'Cambio':>10}")
    print(f"{'-'*70}")

    if default:
        metrics_to_compare = [
            ("Accuracy", "accuracy"),
            ("Recall drunk", "recall_drunk"),
            ("Precision drunk", "precision_drunk"),
            ("Recall sober", "recall_sober"),
            ("F1 drunk", "f1_drunk"),
            ("Falsos negativos", "false_negatives"),
            ("Falsos positivos", "false_positives"),
        ]
        for label, key in metrics_to_compare:
            def_val = default[key]
            opt_val = best[key]
            if isinstance(def_val, float):
                delta = (opt_val - def_val) * 100
                sign = "+" if delta >= 0 else ""
                print(
                    f"  {label:<28} {def_val*100:>12.1f}%  {opt_val*100:>12.1f}%  {sign}{delta:.1f}pp"
                )
            else:
                delta = opt_val - def_val
                sign = "+" if delta >= 0 else ""
                print(f"  {label:<28} {def_val:>13d}  {opt_val:>13d}  {sign}{delta}")

    # Guardar
    save_threshold(best["threshold"], MODELS_DIR, best)

    print(f"\nThreshold optimo seleccionado: {best['threshold']:.2f}")
    print(
        f"  Recall drunk:  {best['recall_drunk']*100:.1f}%  (ebrios detectados correctamente)"
    )
    print(
        f"  Recall sober:  {best['recall_sober']*100:.1f}%  (sobrios detectados correctamente)"
    )
    print(f"  Falsos negativos: {best['false_negatives']} ebrios no detectados")
    print(f"  Falsos positivos: {best['false_positives']} sobrios mal clasificados")

    return best["threshold"]


if __name__ == "__main__":
    run_threshold_tuning()
