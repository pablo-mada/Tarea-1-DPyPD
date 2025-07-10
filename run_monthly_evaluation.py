import pandas as pd
import os
from src.data.dataset import load_raw_data, basic_clean, create_target_variable
from src.features.build_features import create_time_features, select_and_cast_features
from src.modeling.predict import load_model, make_predictions, evaluate_predictions
from src.visualization.plots import plot_monthly_f1_score, plot_monthly_sample_size
from src.config import FEATURES, TARGET_COLUMN, SAMPLE_SIZE, REPORTS_PATH

def main():
    """
    Función principal para la evaluación mensual del modelo.
    Carga el modelo entrenado y lo evalúa sobre datos de diferentes meses.
    """
    # Asegurarse de que el directorio de reportes exista
    os.makedirs(REPORTS_PATH, exist_ok=True)

    print("--- Inicio del Proceso de Evaluación Mensual ---")

    # Cargar el modelo entrenado (se debe ejecutar 'run_training.py' antes)
    model = load_model("random_forest_model.joblib")

    # Definir los meses a evaluar
    # Puedes añadir más meses si los archivos Parquet están disponibles
    months_to_evaluate = ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06','2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12'] # Lista de meses a evaluar
    results = []

    for month_year in months_to_evaluate:
        print(f"\n--- Evaluando para el mes: {month_year} ---")

        # 1. Cargar y limpiar datos para el mes actual
        df_raw = load_raw_data(month_year=month_year, sample_size=SAMPLE_SIZE)
        df_cleaned = basic_clean(df_raw.copy())
        df_with_target = create_target_variable(df_cleaned.copy())

        # 2. Generar características para el mes actual
        df_with_features = create_time_features(df_with_target.copy())
        df_final = select_and_cast_features(df_with_features.copy())

        # Preparar datos para evaluación
        X_test = df_final[FEATURES]
        y_test = df_final[TARGET_COLUMN]

        if X_test.empty:
            print(f"No hay datos para evaluar en {month_year}. Saltando.")
            f1 = None
            num_examples = 0
        else:
            # 3. Realizar predicciones
            preds_labels = make_predictions(model, X_test)

            # 4. Evaluar predicciones
            f1 = evaluate_predictions(y_test, preds_labels)
            num_examples = len(df_final)

        results.append({
            'month': month_year,
            'num_examples': num_examples,
            'f1_score': f1
        })

    results_df = pd.DataFrame(results)
    print("\n--- Tabla de Resultados de Evaluación Mensual ---")
    print(results_df.to_markdown(index=False)) # Imprime en formato Markdown para fácil lectura

    # Guardar resultados en un archivo CSV (opcional)
    results_df.to_csv(f"{REPORTS_PATH}monthly_evaluation_results.csv", index=False)
    print(f"Resultados guardados en: {REPORTS_PATH}monthly_evaluation_results.csv")

    # 5. Visualizar resultados
    plot_monthly_f1_score(results_df.dropna(subset=['f1_score'])) # Solo grafica meses con F1-score válido
    plot_monthly_sample_size(results_df)

    print("\n--- Proceso de Evaluación Mensual Finalizado ---")

if __name__ == "__main__":
    main()