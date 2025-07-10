import pandas as pd
import os
from src.data.dataset import load_raw_data, basic_clean, create_target_variable
from src.features.build_features import create_time_features, select_and_cast_features
from src.modeling.train import train_model, save_model
from src.config import FEATURES, TARGET_COLUMN, SAMPLE_SIZE, MODELS_PATH

def main():
    """
    Función principal para entrenar el modelo.
    Carga datos de enero de 2020, preprocesa, entrena y guarda el modelo.
    """
    # Asegurarse de que el directorio de modelos exista
    os.makedirs(MODELS_PATH, exist_ok=True)

    print("--- Inicio del Proceso de Entrenamiento ---")

    # 1. Cargar y limpiar datos
    df_raw = load_raw_data(month_year='2020-01', sample_size=SAMPLE_SIZE) # Entrenar con enero
    df_cleaned = basic_clean(df_raw.copy())
    df_with_target = create_target_variable(df_cleaned.copy())

    # 2. Generar características
    df_with_features = create_time_features(df_with_target.copy())
    df_final = select_and_cast_features(df_with_features.copy())

    # Preparar datos para entrenamiento
    X_train = df_final[FEATURES]
    y_train = df_final[TARGET_COLUMN]

    # 3. Entrenar el modelo
    model = train_model(X_train, y_train)

    # 4. Guardar el modelo
    save_model(model, "random_forest_model.joblib")

    print("--- Proceso de Entrenamiento Finalizado ---")

if __name__ == "__main__":
    main()