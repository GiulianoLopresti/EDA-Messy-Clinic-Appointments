"""
transformers.py
---------------
Transformers personalizados de Scikit-Learn para el dataset de citas clínicas.

Por qué clases de Scikit-Learn y no funciones normales:
    Un transformer de sklearn tiene dos métodos: fit() y transform().
    - fit()      → aprende parámetros del dataset de entrenamiento (ej. la mediana)
    - transform() → aplica la transformación usando lo que aprendió

    Esto es crítico para no cometer Data Leakage: si calculas la mediana sobre
    todos los datos (incluyendo los de prueba), estás "filtrando" información
    del futuro hacia el pasado. Con fit/transform, aprendes solo en train y
    aplicas en test.

    Además, al heredar de BaseEstimator y TransformerMixin, estas clases se
    pueden encadenar en un Pipeline de sklearn automáticamente.

Transformers en este archivo:
    --- Reutilizados (adaptados del proyecto Bank Marketing) ---
    1. DropColumnsTransformer      → elimina columnas irrelevantes
    2. DropHighMissingTransformer  → descarta columnas con demasiados nulos
    3. OutlierCapper               → limita valores extremos con IQR
    4. DropZeroVarianceTransformer → elimina columnas constantes
    5. SmartImputerTransformer     → imputa nulos con mediana o moda

    --- Nuevos (específicos del dataset de citas clínicas) ---
    6. GenderNormalizerTransformer → unifica las 8 variantes de género
    7. BillingCleanerTransformer   → extrae el número de strings con moneda
    8. DateFeatureTransformer      → parsea fechas y genera features derivadas
"""
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
 
 
# =============================================================================
# BLOQUE 1: Transformers reutilizados (estructurales / genéricos)
# =============================================================================
 
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas especificadas del DataFrame.
 
    Uso principal: quitar columnas que causan Data Leakage o que no tienen
    valor predictivo (ej. patient_name, columnas de fecha ya procesadas).
 
    Args:
        columns_to_drop: Lista de nombres de columnas a eliminar.
    """
    def __init__(self, columns_to_drop: list):
        self.columns_to_drop = columns_to_drop
 
    def fit(self, X, y=None):
        return self  # No necesita aprender nada
 
    def transform(self, X):
        X_copy = X.copy()
        # Solo elimina si la columna existe (evita error si ya fue eliminada antes)
        cols = [col for col in self.columns_to_drop if col in X_copy.columns]
        return X_copy.drop(columns=cols)
 
 
class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """
    Descarta columnas que superan un umbral de valores nulos.
 
    Lógica: si una columna tiene más del 80% de sus valores vacíos, imputarla
    sería inventarse datos. Es más honesto eliminarla.
 
    Args:
        threshold: Porcentaje máximo de nulos permitido (por defecto 0.80 = 80%).
    """
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self.cols_to_drop_ = []  # Se llenará en fit()
 
    def fit(self, X, y=None):
        pct_nulos = X.isnull().mean()
        self.cols_to_drop_ = pct_nulos[pct_nulos > self.threshold].index.tolist()
        if self.cols_to_drop_:
            print(f"🗑️  DropHighMissing: eliminando {self.cols_to_drop_} (>{self.threshold*100:.0f}% nulos)")
        return self
 
    def transform(self, X):
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        return X_copy.drop(columns=cols)
 
 
class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Limita valores extremos usando el método IQR (Rango Intercuartílico).
 
    En lugar de eliminar filas con outliers (perdemos pacientes reales), los
    "recortamos": cualquier valor por encima del límite superior se reemplaza
    por el límite superior, y viceversa. Esto se llama Winsorización o Capping.
 
    El switch apply_capping=False permite desactivarlo para comparar resultados.
 
    Args:
        apply_capping: Si False, devuelve los datos sin cambios.
    """
    def __init__(self, apply_capping: bool = True):
        self.apply_capping = apply_capping
        self.bounds_ = {}
 
    def fit(self, X, y=None):
        if not self.apply_capping:
            return self
        for col in X.select_dtypes(include=["number"]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self
 
    def transform(self, X):
        X_copy = X.copy()
        if not self.apply_capping:
            return X_copy
        for col, (lower, upper) in self.bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], lower, upper)
        return X_copy
 
    def get_feature_names_out(self, input_features=None):
        return input_features
 
 
class DropZeroVarianceTransformer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas numéricas que tienen un solo valor (varianza = 0).
 
    Una columna constante no aporta información al modelo porque no discrimina
    entre registros. Ejemplo: si todos los pacientes fueran del mismo género,
    esa columna sería inútil.
    """
    def __init__(self):
        self.cols_to_drop_ = []
 
    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=["number"]).columns
        self.cols_to_drop_ = [col for col in num_cols if X[col].std() == 0]
        return self
 
    def transform(self, X):
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        return X_copy.drop(columns=cols)
 
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.array([f for f in input_features if f not in self.cols_to_drop_])
 
 
class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Imputa valores nulos con una estrategia basada en el porcentaje de nulos.
 
    Estrategia:
        - < 10% nulos  → imputación simple (mediana para números, moda para texto)
        - 10% - 80%    → actualmente usa la misma estrategia simple (mejora futura: KNN)
        - > 80%        → ya eliminada por DropHighMissingTransformer
 
    Por qué mediana y no promedio:
        La mediana es más robusta ante outliers. Si un paciente tiene billing
        de $10.000 mientras el resto tiene $100, el promedio sube mucho pero
        la mediana apenas se mueve.
 
    Args:
        low_threshold: Límite para considerar imputación "simple" (default 0.10).
    """
    def __init__(self, low_threshold: float = 0.10):
        self.low_threshold = low_threshold
        self.cols_simples_ = []
        self.cols_complejas_ = []
        self.fill_values_ = {}  # Guarda los valores aprendidos en fit()
 
    def fit(self, X, y=None):
        pct_nulos = X.isnull().mean()
        self.cols_simples_ = []
        self.cols_complejas_ = []
        self.fill_values_ = {}
 
        for col in X.columns:
            pct = pct_nulos[col]
            if 0 < pct <= self.low_threshold:
                self.cols_simples_.append(col)
            elif pct > self.low_threshold:
                self.cols_complejas_.append(col)
 
        # Aprende los valores de imputación SOLO del conjunto de entrenamiento
        for col in self.cols_simples_ + self.cols_complejas_:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.fill_values_[col] = X[col].median()
            else:
                self.fill_values_[col] = X[col].mode()[0]
 
        print(f"🧠 SmartImputer - Simples  (<{self.low_threshold*100:.0f}%): {self.cols_simples_}")
        print(f"🚧 SmartImputer - Complejas (>{self.low_threshold*100:.0f}%): {self.cols_complejas_}")
        return self
 
    def transform(self, X):
        X_copy = X.copy()
        for col in self.cols_simples_ + self.cols_complejas_:
            if col in X_copy.columns and col in self.fill_values_:
                X_copy[col] = X_copy[col].fillna(self.fill_values_[col])
        return X_copy
 
    def get_feature_names_out(self, input_features=None):
        return input_features
 
 
# =============================================================================
# BLOQUE 2: Transformers nuevos (específicos del dataset clínico)
# =============================================================================
 
class GenderNormalizerTransformer(BaseEstimator, TransformerMixin):
    """
    Unifica las 8 variantes de género en dos valores estándar: 'Male' / 'Female'.
 
    Problema detectado en el EDA:
        El dataset tiene: 'Male', 'M', 'male', '1' → todos significan masculino
                          'Female', 'F', 'female', '0' → todos femenino
                          NaN → 50 registros sin valor
 
    Por qué no usar UnknownToNaNTransformer para esto:
        UnknownToNaNTransformer solo convierte el string "unknown" a NaN.
        Aquí el problema es diferente: los valores son válidos pero inconsistentes.
        Necesitamos un mapeo explícito.
 
    Después de este transformer, SmartImputerTransformer imputará los NaN.
    """
    # Diccionario de mapeo: variante → valor estándar
    GENDER_MAP = {
        "male":   "Male",  "m": "Male",  "1": "Male",
        "female": "Female","f": "Female","0": "Female",
    }
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X_copy = X.copy()
        if "gender" in X_copy.columns:
            # Paso 1: normalizar texto (minúsculas, sin espacios extra)
            # Usamos .str para operar sobre toda la columna a la vez (vectorización)
            normalized = X_copy["gender"].str.strip().str.lower()
 
            # Paso 2: mapear al valor estándar; lo que no esté en el mapa → NaN
            X_copy["gender"] = normalized.map(self.GENDER_MAP)
 
        return X_copy
 
 
class BillingCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Extrae el valor numérico de la columna billing_amount, que contiene
    strings con símbolos de moneda mezclados (£, €, $, Rs).
 
    Problema detectado en el EDA:
        '£425.8'   → 425.8
        '€344.26'  → 344.26
        'Rs374.63' → 374.63
        '$84.44'   → 84.44
 
    Decisión de negocio documentada:
        Descartamos el símbolo de moneda porque todas las citas pertenecen
        al mismo sistema clínico y asumimos que los montos son comparables.
        En un proyecto real habría que convertir divisas con tipos de cambio.
 
    Técnica usada:
        Expresión regular (regex) para extraer solo dígitos y punto decimal.
        '[\\d.]+' significa "uno o más caracteres que sean dígito o punto".
    """
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X_copy = X.copy()
        if "billing_amount" in X_copy.columns:
            # Extraemos solo la parte numérica con regex y convertimos a float
            X_copy["billing_amount"] = (
                X_copy["billing_amount"]
                .astype(str)
                .str.extract(r"([\d.]+)")[0]  # grupo 0 = primer match
                .astype(float)
            )
        return X_copy
 
 
class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Parsea las columnas de fecha y genera features nuevas derivadas.
 
    Problema detectado en el EDA:
        Las fechas usan 4 formatos distintos en la misma columna:
        '2026/02/26', '05/23/2025', '30-Nov-2025', 'May 18, 25'
 
    Features generadas:
        waiting_days    → días entre booking_date y appointment_date
                          (¿los pacientes que esperan más necesitan seguimiento?)
        appointment_dow → día de la semana de la cita (0=lunes, 6=domingo)
                          (¿hay días con más no-shows?)
 
    Por qué eliminamos las fechas originales:
        Un modelo de ML no puede operar sobre strings como '2026/02/26'.
        Solo puede usar números. Las features derivadas capturan la información
        útil de las fechas en formato numérico.
 
    Nota técnica sobre errors='coerce':
        Si una fecha no se puede parsear (ej. un valor corrupto), en lugar de
        lanzar un error la convierte a NaT (Not a Time), el equivalente de NaN
        para fechas. SmartImputerTransformer se encargará de esos nulos después.
    """
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X_copy = X.copy()
 
        appt_col    = "appointment_date"
        booking_col = "booking_date"
 
        if appt_col in X_copy.columns and booking_col in X_copy.columns:
            # Parseo flexible: Pandas intenta múltiples formatos automáticamente
            appt_dates    = pd.to_datetime(X_copy[appt_col],    format="mixed", errors="coerce")
            booking_dates = pd.to_datetime(X_copy[booking_col], format="mixed", errors="coerce")
 
            # Feature 1: días de espera (como número entero)
            X_copy["waiting_days"] = (appt_dates - booking_dates).dt.days
 
            # Feature 2: día de la semana (0 = lunes, 6 = domingo)
            X_copy["appointment_dow"] = appt_dates.dt.dayofweek
 
            # Eliminamos las columnas de texto originales (ya no son útiles)
            X_copy = X_copy.drop(columns=[appt_col, booking_col], errors="ignore")
 
        return X_copy
 