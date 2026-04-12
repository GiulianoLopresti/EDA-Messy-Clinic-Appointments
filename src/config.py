"""
config.py
---------
Centraliza todas las rutas y constantes del proyecto.

Por qué existe este archivo:
    Si  se escribe las rutas directamente en cada notebook o script ("hardcodear"),
    cuando se mueva una carpeta o se cambie el nombre de un archivo
    se tiene que buscar y corregir en todos los archivos. Con config.py
    corrigir en un solo lugar y el resto del proyecto lo recoge automáticamente.
"""
from pathlib import Path
 
# --- Raíz del proyecto -------------------------------------------------
# Path(__file__) es la ruta de este mismo archivo (config.py).
# .parent sube un nivel → carpeta src/
# .parent de nuevo      → raíz del proyecto (clinic-appointments/)
ROOT = Path(__file__).parent.parent
 
# --- Datos ------------------------------------------------------------
DATA_DIR       = ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
 
RAW_CSV        = RAW_DIR  / "messy_clinic_appointments.csv"
METADATA_JSON  = RAW_DIR  / "metadata.json"
CLEAN_CSV      = PROCESSED_DIR / "clean_appointments.csv"
 
# --- Outputs (gráficos exportados) ------------------------------------
OUTPUTS_DIR    = ROOT / "outputs"
 
# --- Constantes del dataset -------------------------------------------
# Columnas que se eliminan ANTES del pipeline (no aportan al modelo)
# - patient_name : identificador personal, sin poder predictivo
# - doctor       : 990 valores únicos en 1000 filas → alta cardinalidad,
#                  OneHotEncoding generaría ~990 columnas (inviable)
# - booking_date / appointment_date : reemplazadas por waiting_days y appointment_dow
COLS_TO_DROP = ["patient_name", "doctor", "booking_date", "appointment_date"]
 
# Variable objetivo (target)
TARGET_COL = "follow_up_required"
 
# Columnas numéricas que entran al pipeline
# appointment_dow se genera en DateFeatureTransformer (0=Lunes, 6=Domingo)
NUMERIC_COLS = ["age", "billing_amount", "waiting_days", "appointment_dow"]
 
# Columnas categóricas que entran al pipeline
# doctor fue eliminado por alta cardinalidad (990 únicos / 1000 filas)
CATEGORICAL_COLS = ["gender", "department"]
 
# Umbral para descartar columnas con demasiados nulos (80%)
HIGH_MISSING_THRESHOLD = 0.80
 
# Umbral para imputación simple vs compleja (10%)
SIMPLE_IMPUTE_THRESHOLD = 0.10