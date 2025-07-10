import joblib
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier 
from src.config import MODELS_PATH, TARGET_COLUMN, FEATURES

def load_model(filename: str = "random_forest_model.joblib") -> RandomForestClassifier:
    """
    Carga un modelo serializado utilizando joblib.

    Args:
        filename (str): Nombre del archivo del modelo a cargar.

    Returns:
        RandomForestClassifier: El modelo cargado.
    """
    filepath = f"{MODELS_PATH}{filename}"
    print(f"Cargando modelo desde: {filepath}")
    model = joblib.load(filepath)
    return model

def make_predictions(model: RandomForestClassifier, X_data: pd.DataFrame) -> list[int]:
    """
    Realiza predicciones de etiquetas binarias utilizando un modelo entrenado.

    Args:
        model (RandomForestClassifier): El modelo entrenado.
        X_data (pd.DataFrame): DataFrame de características para las predicciones.

    Returns:
        list[int]: Lista de etiquetas predichas (0 o 1).
    """
    print("Realizando predicciones...")
    preds_proba = model.predict_proba(X_data)
    # Convertir probabilidades de clase 1 a etiquetas binarias redondeando
    preds_labels = [int(p[1].round()) for p in preds_proba]
    print("Predicciones completadas.")
    return preds_labels

def evaluate_predictions(y_true: pd.Series, y_pred: list[int]) -> float:
    """
    Evalúa las predicciones calculando el F1-score y muestra un reporte de clasificación.

    Args:
        y_true (pd.Series): Valores verdaderos de la variable objetivo.
        y_pred (list[int]): Valores predichos de la variable objetivo.

    Returns:
        float: El F1-score calculado.
    """
    f1 = f1_score(y_true, y_pred)
    print(f"\n--- Resultados de Evaluación ---")
    print(f"F1-score: {f1:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred))
    return f1