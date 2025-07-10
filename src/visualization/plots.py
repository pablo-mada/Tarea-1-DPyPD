import matplotlib.pyplot as plt
import pandas as pd
from src.config import REPORTS_PATH

def plot_monthly_f1_score(results_df: pd.DataFrame, filename: str = "monthly_f1_score.png"):
    """
    Genera un gráfico de línea del F1-score mensual.

    Args:
        results_df (pd.DataFrame): DataFrame con columnas 'month' y 'f1_score'.
        filename (str): Nombre del archivo para guardar el gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['month'], results_df['f1_score'], marker='o', linestyle='-', color='skyblue')
    plt.title('Rendimiento del Modelo (F1-score) por Mes', fontsize=16)
    plt.xlabel('Mes', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout() 
    plt.savefig(f"{REPORTS_PATH}{filename}")
    print(f"Gráfico guardado en: {REPORTS_PATH}{filename}")
    plt.show()

def plot_monthly_sample_size(results_df: pd.DataFrame, filename: str = "monthly_sample_size.png"):
    """
    Genera un gráfico de barras del número de ejemplos por mes.

    Args:
        results_df (pd.DataFrame): DataFrame con columnas 'month' y 'num_examples'.
        filename (str): Nombre del archivo para guardar el gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['month'], results_df['num_examples'], color='lightcoral')
    plt.title('Número de Ejemplos por Mes', fontsize=16)
    plt.xlabel('Mes', fontsize=12)
    plt.ylabel('Cantidad de Ejemplos', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}{filename}")
    print(f"Gráfico guardado en: {REPORTS_PATH}{filename}")
    plt.show()