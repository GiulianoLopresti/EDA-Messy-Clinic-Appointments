"""
pipeline.py
-----------
Ensambla el Pipeline completo de limpieza y transformación.

Por qué un archivo separado para esto:
    transformers.py define las piezas (las clases individuales).
    pipeline.py las ensambla en el orden correcto.
    El notebook solo necesita importar build_pipeline() y llamarla.
    Si cambias el orden de pasos, solo modificas este archivo.

Orden del pipeline explicado:
    1. Pre-limpieza global  → arregla problemas estructurales (fechas, moneda, género)
    2. Limpieza de columnas → elimina las innecesarias y las muy nulas
    3. ColumnTransformer    → bifurca en ruta numérica y ruta categórica en paralelo
        a. Ruta numérica    → capping, escalar
        b. Ruta categórica  → imputar, codificar

Por qué ColumnTransformer:
    Las variables numéricas y categóricas necesitan tratamientos diferentes.
    ColumnTransformer aplica una transformación a un subconjunto de columnas
    y otra transformación a otro subconjunto, todo en paralelo.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
 
from transformers import (
    DropColumnsTransformer,
    DropHighMissingTransformer,
    OutlierCapper,
    DropZeroVarianceTransformer,
    SmartImputerTransformer,
    GenderNormalizerTransformer,
    BillingCleanerTransformer,
    DateFeatureTransformer,
)
from config import (
    COLS_TO_DROP,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    HIGH_MISSING_THRESHOLD,
    SIMPLE_IMPUTE_THRESHOLD,
)
 
 
def build_pipeline(apply_capping: bool = True) -> Pipeline:
    """
    Construye y retorna el Pipeline completo listo para usar.
 
    Args:
        apply_capping: Si True (default), aplica Winsorización a outliers.
                       Ponlo en False para comparar resultados sin capping.
 
    Returns:
        Pipeline de sklearn con todos los pasos de limpieza y transformación.
 
    Ejemplo de uso:
        pipeline = build_pipeline()
        X_clean  = pipeline.fit_transform(df_raw.drop(columns=['follow_up_required']))
    """
 
    # ------------------------------------------------------------------
    # Paso A: Ruta para variables NUMÉRICAS
    # Recibe: age, billing_amount, waiting_days, appointment_dow
    # ------------------------------------------------------------------
    numeric_pipeline = Pipeline([
        # Imputar PRIMERO: billing_amount tiene 50 nulos (5%).
        # El capper no puede operar sobre NaN, así que va antes.
        ("imputer",       SmartImputerTransformer(low_threshold=SIMPLE_IMPUTE_THRESHOLD)),
        ("capper",        OutlierCapper(apply_capping=apply_capping)),
        ("zero_variance", DropZeroVarianceTransformer()),
        ("scaler",        StandardScaler()),
    ])
 
    # ------------------------------------------------------------------
    # Paso B: Ruta para variables CATEGÓRICAS
    # Recibe: gender, department, doctor
    # ------------------------------------------------------------------
    categorical_pipeline = Pipeline([
        ("imputer", SmartImputerTransformer(low_threshold=SIMPLE_IMPUTE_THRESHOLD)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        # OneHotEncoder: convierte 'Cardiology', 'Neurology', etc. en columnas
        # binarias (0/1). handle_unknown='ignore' evita errores si aparece
        # un valor nuevo en producción que no estaba en el entrenamiento.
    ])
 
    # ------------------------------------------------------------------
    # Paso C: ColumnTransformer — aplica A y B en paralelo
    # ------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,      NUMERIC_COLS),
            ("cat", categorical_pipeline,  CATEGORICAL_COLS),
        ],
        remainder="drop",  # elimina cualquier columna no listada
    )
 
    # ------------------------------------------------------------------
    # Pipeline principal: pre-limpieza → limpieza → preprocesamiento
    # ------------------------------------------------------------------
    pipeline = Pipeline([
        # Paso 1: normalizar género (8 variantes → Male/Female)
        ("gender_norm",    GenderNormalizerTransformer()),
 
        # Paso 2: limpiar montos (extraer número de '£425.8' → 425.8)
        ("billing_clean",  BillingCleanerTransformer()),
 
        # Paso 3: procesar fechas (generar waiting_days, appointment_dow)
        ("date_features",  DateFeatureTransformer()),
 
        # Paso 4: eliminar columnas sin valor predictivo
        # (patient_name, booking_date, appointment_date ya procesadas)
        ("drop_cols",      DropColumnsTransformer(COLS_TO_DROP)),
 
        # Paso 5: descartar columnas con demasiados nulos
        ("drop_missing",   DropHighMissingTransformer(threshold=HIGH_MISSING_THRESHOLD)),
 
        # Paso 6: rutas paralelas para numéricas y categóricas
        ("preprocessor",   preprocessor),
    ])
 
    return pipeline