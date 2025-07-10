# Rutas de datos
BASE_DATA_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/'
MODELS_PATH = 'models/' # Directorio para guardar modelos entrenados
REPORTS_PATH = 'reports/' # Directorio para guardar resultados y visualizaciones

# Columnas y características
TARGET_COLUMN = "high_tip"
NUMERIC_FEATURES = [
    "pickup_weekday",
    "pickup_hour",
    'work_hours',
    "pickup_minute",
    "passenger_count",
    'trip_distance',
    'trip_time',
    'trip_speed'
]
CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Parámetros del modelo (RandomForestClassifier)
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RANDOM_STATE = 42 # Para reproducibilidad

# Constantes matemáticas
EPS = 1e-7

# Tamaño de la muestra para desarrollo/pruebas
SAMPLE_SIZE = None

#SAMPLE_SIZE = 100000 # Para pruebas rápidas, puedes ajustar este valor