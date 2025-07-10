import pandas as pd
from src.config import BASE_DATA_URL, TARGET_COLUMN, EPS

def load_raw_data(month_year: str, sample_size: int = None) -> pd.DataFrame:
    """
    Carga los datos brutos de viajes en taxi para un mes y año específico desde una URL.

    Args:
        month_year (str): Mes y año en formato 'YYYY-MM'.
        sample_size (int, optional): Número de filas a cargar. Si es None, carga todo el archivo.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    url = f"{BASE_DATA_URL}yellow_tripdata_{month_year}.parquet"
    print(f"Cargando datos desde: {url}")
    df = pd.read_parquet(url)
    if sample_size is not None:
        df = df.head(sample_size)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza una limpieza básica del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    # Evitar división por cero en tip_fraction
    df = df[df['fare_amount'] > 0].reset_index(drop=True)
    return df

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable objetivo 'high_tip'.

    Args:
        df (pd.DataFrame): DataFrame con la columna 'tip_amount' y 'fare_amount'.

    Returns:
        pd.DataFrame: DataFrame con la columna objetivo añadida.
    """
    df['tip_fraction'] = df['tip_amount'] / (df['fare_amount'] ) 
    df[TARGET_COLUMN] = (df['tip_fraction'] > 0.2).astype("int32") # Convertir a int32
    return df