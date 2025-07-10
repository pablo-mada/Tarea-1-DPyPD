import pandas as pd
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURES, TARGET_COLUMN, EPS

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características relacionadas con el tiempo a partir de las columnas de fecha y hora.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'tpep_pickup_datetime' y 'tpep_dropoff_datetime'.

    Returns:
        pd.DataFrame: DataFrame con las nuevas características de tiempo.
    """
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute

    df['work_hours'] = (df['pickup_weekday'] >= 0) & (df['pickup_weekday'] <= 4) & \
                       (df['pickup_hour'] >= 8) & (df['pickup_hour'] <= 18)

    df['work_hours'] = df['work_hours'].astype(float)

    # Calcular trip_time en segundos
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds().fillna(0)
    # Asegurarse que trip_time no sea negativo por algún error de datos 
    df['trip_time'] = df['trip_time'].apply(lambda x: max(x, 0))


    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + EPS)
    return df

def select_and_cast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona las características finales y las convierte al tipo de dato adecuado,
    además de manejar valores nulos.

    Args:
        df (pd.DataFrame): DataFrame con todas las características generadas.

    Returns:
        pd.DataFrame: DataFrame listo para el modelo con las características seleccionadas.
    """
    # Seleccionar solo las columnas de features y la columna objetivo
    selected_df = df[FEATURES + [TARGET_COLUMN]].copy()

    # Convertir todas las features y el target a float32 y manejar nulos
    selected_df[FEATURES + [TARGET_COLUMN]] = selected_df[FEATURES + [TARGET_COLUMN]].astype("float32").fillna(-1.0)

    return selected_df