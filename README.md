# Tarea 1 de Desarrollo de proyectos y productos de datos

# Instrucciones para configurar y ejecutar el proyecto:

- 1. Crea la estructura de carpetas:

├── models/
├── reports/
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── modeling/
│   │   ├── predict.py
│   │   └── train.py
│   ├── visualization/
│   │   └── plots.py
│   └── config.py
├── run_training.py
└── run_monthly_evaluation.py

- 2. Guarda los archivos: Copia el código de cada sección en su archivo `.py` correspondiente.

- 3. Crea un entorno virtual 

    - `python -m venv venv`
    - `source venv/bin/activate`  # En Linux/macOS
    - `venv\Scripts\activate` # En Windows

- 4. Instala las dependencias:

     `pip install -r requirements.txt`   

- 5. Entrena el modelo con datos de enero

    `python run_training.py `

- 6. Ejecuta la evaluación mensual

    `python run_monthly_evaluation.py `