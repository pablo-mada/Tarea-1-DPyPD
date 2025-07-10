import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.config import RF_N_ESTIMATORS, RF_MAX_DEPTH, RANDOM_STATE, MODELS_PATH, FEATURES, TARGET_COLUMN

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Entrena un modelo RandomForestClassifier.

    Args:
        X_train (pd.DataFrame): DataFrame de características para el entrenamiento.
        y_train (pd.Series): Serie de la variable objetivo para el entrenamiento.

    Returns:
        RandomForestClassifier: El modelo RandomForest entrenado.
    """
    print("Entrenando RandomForestClassifier...")
    rfc = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RANDOM_STATE)
    rfc.fit(X_train, y_train)
    print("Entrenamiento completado.")
    return rfc

def save_model(model: RandomForestClassifier, filename: str = "random_forest_model.joblib"):
    """
    Guarda el modelo entrenado utilizando joblib.

    Args:
        model (RandomForestClassifier): El modelo a guardar.
        filename (str): Nombre del archivo para guardar el modelo.
    """
    filepath = f"{MODELS_PATH}{filename}"
    joblib.dump(model, filepath)
    print(f"Modelo guardado en: {filepath}")