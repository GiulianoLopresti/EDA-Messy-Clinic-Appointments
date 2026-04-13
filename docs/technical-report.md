# Informe Técnico: Preparación de Datos — Messy Clinic Appointments

**Equipo:** Marco Bona - Giuliano Lopresti - Lorena Ormeño
**Asignatura:** SCY1101 - Programación para la Ciencia de Datos
**Fecha:** 15/04/26

---

## 1. Resumen Ejecutivo

Este proyecto aborda la preparación de datos del dataset *Messy Clinic Appointments*
con el objetivo de predecir si un paciente requiere seguimiento médico post-consulta
(`follow_up_required`). El dataset presenta problemas deliberados de calidad de datos:
múltiples representaciones para el mismo valor, fechas en formatos inconsistentes,
montos con símbolos de moneda mezclados y columnas de alta cardinalidad.

Se implementó un flujo de trabajo reproducible compuesto por:
auditoría de integridad SHA-256, análisis exploratorio exhaustivo, transformers
personalizados de Scikit-Learn, y un Pipeline de procesamiento automatizado.

**Resultados clave:**
- Dataset procesado: 1.000 filas × 10 features numéricas limpias, sin valores nulos.
- Variable objetivo balanceada: ~51% requiere seguimiento / ~49% no requiere.
- Reducción de columnas de 9 a 10 features útiles (eliminación de 2 columnas de alta
  cardinalidad / identidad, generación de 2 features de fecha, 6 columnas OneHot).
- Reducción de memoria por downcasting: [insertar % del notebook].
- Hash SHA-256 del dataset original: `[pegar hash del metadata.json]`

---

## 2. Análisis Exploratorio Inicial (EDA)

### 2.1 Caracterización del Dataset

El dataset contiene **1.000 registros** y **10 columnas** originales correspondientes
a citas médicas en una clínica. Las variables cubren información demográfica del
paciente, información temporal de la cita, datos económicos y el resultado clínico.

*(Insertar tabla de df.info() / df.describe() del notebook)*

| Columna | Tipo original | Descripción |
|---|---|---|
| `patient_id` | int64 | Identificador del paciente |
| `patient_name` | object | Nombre completo |
| `age` | int64 | Edad del paciente |
| `gender` | object | Género (con inconsistencias) |
| `appointment_date` | object | Fecha de cita (múltiples formatos) |
| `booking_date` | object | Fecha de reserva |
| `doctor` | object | Nombre del médico |
| `department` | object | Departamento clínico |
| `billing_amount` | object | Monto con símbolo de moneda |
| `follow_up_required` | object | Variable objetivo |

### 2.2 Variable Objetivo

*(Insertar gráfico `target_distribution.png`)*

La variable `follow_up_required` presentó **6 representaciones distintas** para dos
valores semánticos:
- Positivo (requiere seguimiento): `Yes`, `Y`, `1` → 514 registros (51.4%)
- Negativo (no requiere): `No`, `N`, `0` → 486 registros (48.6%)

El **balance de clases** es favorable: no se requieren técnicas de sobremuestreo
(SMOTE) ni ajuste de pesos para el modelado posterior.

### 2.3 Calidad de Datos — Problemas Identificados

*(Insertar gráfico `nulos_heatmap.png`)*

| Columna | Problema | Registros afectados |
|---|---|---|
| `gender` | 8 variantes + nulos reales | 50 nulos + inconsistencia en 950 |
| `billing_amount` | Texto con símbolo de moneda | 50 nulos |
| `appointment_date` | 4 formatos de fecha distintos | 1.000 (todos) |
| `booking_date` | 4 formatos de fecha distintos | 1.000 (todos) |
| `doctor` | 990 valores únicos (alta cardinalidad) | 1.000 |
| `follow_up_required` | 6 variantes para 2 valores | 1.000 |

### 2.4 Análisis de Variables Numéricas

*(Insertar gráficos `numeric_distributions.png` y `outliers_before_after.png`)*

**Age:** Rango limpio de 18 a 90 años, sin valores imposibles. Distribución
aproximadamente uniforme entre grupos con y sin seguimiento (medias de 52.9 y 54.5
años respectivamente), lo que sugiere que la edad por sí sola no es un predictor
fuerte.

**Billing Amount:** Después de limpiar los símbolos de moneda, el rango es de
\$51.38 a \$499.75, con mediana de \$269.84. Los 50 nulos representan el 5% del total
y se imputaron con la mediana para no distorsionar la distribución.

**Waiting Days:** Días entre la fecha de reserva y la cita. Rango de 2 a 728 días,
con mediana de 258.5 días (~8.6 meses). Sin valores negativos, lo que confirma la
consistencia temporal del dataset.

### 2.5 Análisis de Variables Categóricas

*(Insertar gráfico `categorical_analysis.png`)*

**Department:** 4 departamentos bien distribuidos: Neurology (27.3%), Orthopedics
(26.2%), Cardiology (23.4%), General (23.1%). La tasa de seguimiento es similar
entre departamentos (~49-54%), sin diferencias significativas.

**Doctor — Hallazgo crítico:** La columna presenta **990 valores únicos** en 1.000
filas, lo que la hace prácticamente un identificador único. Aplicar OneHotEncoding
generaría ~990 columnas nuevas, causando la *maldición de la dimensionalidad* y
haciendo el modelo inviable. **Decisión:** se eliminó del pipeline.

### 2.6 Feature Engineering — Fechas

*(Insertar gráfico `date_features.png`)*

Las columnas de fecha originales (`appointment_date`, `booking_date`) no pueden
utilizarse directamente en un modelo de ML por estar en formato texto con 4
variantes distintas. Se generaron dos features derivadas:

- **`waiting_days`**: diferencia en días entre reserva y cita. Captura la urgencia
  de la atención y la carga del sistema de salud.
- **`appointment_dow`**: día de la semana de la cita (0=Lunes, 6=Domingo). Permite
  detectar patrones semanales en la necesidad de seguimiento.

### 2.7 Matriz de Correlación

*(Insertar gráfico `correlation_matrix.png`)*

Las correlaciones entre variables numéricas son bajas (todas por debajo de |0.05|),
lo que indica ausencia de multicolinealidad. Esto es favorable para la mayoría de
los algoritmos de clasificación.

---

## 3. Metodología de Transformación

### 3.1 Arquitectura del Proyecto

El proyecto sigue una arquitectura modular donde la lógica de transformación
está completamente separada de los notebooks de análisis:

```
src/
├── config.py        ← rutas y constantes centralizadas
├── audit.py         ← verificación de integridad SHA-256
├── optimization.py  ← downcasting y procesamiento por chunks
├── transformers.py  ← 8 clases sklearn personalizadas
└── pipeline.py      ← ensamblaje del pipeline completo
```

Esta separación permite que cualquier miembro del equipo pueda modificar
un transformer sin afectar los demás, y que el pipeline sea reutilizable
en otros datasets del mismo dominio.

### 3.2 Decisiones Técnicas y Justificaciones

#### Eliminación de columnas (Data Leakage y cardinalidad)

Se eliminaron `patient_name` (identificador personal sin poder predictivo)
y `doctor` (990 valores únicos, inutilizable para encoding). Ninguna de estas
columnas aporta información generalizable sobre la necesidad de seguimiento.

#### Agrupación Múltiple y Enriquecimiento con Merge

Se calcularon estadísticas agregadas por departamento (`groupby` múltiple:
conteo, billing promedio, días de espera, tasa de seguimiento) y se incorporaron
al dataset mediante un `pd.merge(..., how='left')`. Esto permite a cada registro
"conocer" el contexto estadístico de su departamento, un patrón común en
feature engineering para datos tabulares.

Adicionalmente, se generó una `pivot_table` cruzando `department` × `gender`
sobre `billing_amount` para identificar si el monto facturado varía por combinación
de ambas dimensiones. El resultado muestra que las diferencias son menores a $30
entre géneros dentro del mismo departamento, confirmando que el género no es un
determinante del monto.

#### Normalización de género

En lugar de tratar las 8 variantes como valores distintos, se aplicó un mapeo
explícito mediante diccionario antes del pipeline formal. Los 50 nulos resultantes
se imputaron con la moda (el valor más frecuente), lo cual es apropiado para
variables categóricas con bajo porcentaje de nulos (<5%).

#### Limpieza y Normalización de billing_amount

Se utilizó una expresión regular para extraer el componente numérico y detectar
el símbolo de moneda. A diferencia de simplemente descartarlo, se implementó una
**conversión real a USD** mediante tasas de cambio fijas:

| Símbolo | Moneda | Tasa a USD |
|---|---|---|
| `£` | Libra esterlina (GBP) | × 1.27 |
| `€` | Euro (EUR) | × 1.09 |
| `Rs` | Rupia india (INR) | × 0.012 |
| `$` | Dólar (USD) | × 1.00 |

**Justificación:** Sin esta conversión, el modelo trataría 100 Rupias como
equivalentes a 100 Libras, introduciendo un sesgo sistemático. Al normalizar
a USD, la escala es comparable entre todos los registros. Los 50 nulos (5%)
se imputaron con la mediana en USD, más robusta que la media ante valores extremos.

#### Feature Engineering de fechas

Se optó por generar features derivadas en lugar de codificar las fechas directamente,
porque las fechas en sí mismas no generalizan (una fecha específica no aparecerá
en nuevos datos), pero los patrones temporales como días de espera y día de la
semana sí son consistentes.

#### Tratamiento de outliers (Winsorización / Capping)

Se aplicó el método IQR para detectar outliers en variables numéricas. En lugar
de eliminar filas con valores extremos (lo cual reduciría el dataset y perdería
información de pacientes reales), se aplicó *Capping*: los valores por encima del
límite superior se reemplazan por ese límite, y análogamente para el inferior.

#### Escalado (StandardScaler)

Las variables numéricas se escalaron a media=0 y desviación estándar=1. Esto
es necesario para evitar que variables con rangos grandes (como `waiting_days`,
que puede llegar a 728) dominen sobre variables con rangos pequeños (como
`appointment_dow`, de 0 a 6) en algoritmos sensibles a la escala (SVM, KNN,
regresión logística).

#### OneHotEncoding

Las variables categóricas `gender` y `department` se codificaron con OneHotEncoder.
Se eligió este método sobre LabelEncoder porque las categorías no tienen orden
intrínseco. El parámetro `handle_unknown='ignore'` garantiza que el modelo
no falle si en producción aparece un valor de departamento no visto en el
entrenamiento.

### 3.3 Flujo completo del Pipeline

*(Insertar captura del diagrama interactivo del Pipeline de sklearn)*

| Paso | Transformer | Entrada | Salida |
|---|---|---|---|
| 1 | `GenderNormalizerTransformer` | 8 variantes | `Male` / `Female` / NaN |
| 2 | `BillingCleanerTransformer` | `"£425.8"` | `425.8` |
| 3 | `DateFeatureTransformer` | 2 cols texto | `waiting_days`, `appointment_dow` |
| 4 | `DropColumnsTransformer` | 9 columnas | 7 columnas |
| 5 | `DropHighMissingTransformer` | 7 columnas | 7 columnas (ninguna descartada) |
| 6a | `SmartImputer` (num) | 50 nulos en billing | 0 nulos |
| 6b | `OutlierCapper` | valores extremos | valores limitados por IQR |
| 6c | `StandardScaler` | escala original | media=0, std=1 |
| 6d | `SmartImputer` (cat) | 50 nulos en gender | 0 nulos |
| 6e | `OneHotEncoder` | 2 cols categóricas | 6 columnas binarias |

---

## 4. Resultados y Validación Técnica

### 4.1 Integridad del Dataset

El archivo original fue validado mediante SHA-256 antes de cualquier
transformación, generando la firma:

```
SHA-256: [pegar hash de data/raw/metadata.json]
Tamaño: [pegar size_mb] MB
```

Esta firma se almacena en `data/raw/metadata.json` y puede verificarse
en cualquier momento con `src/audit.py`.

### 4.2 Optimización de Memoria

*(Insertar resultados de optimize_memory_usage del notebook)*

Se aplicó downcasting sobre las columnas numéricas del dataset procesado,
logrando una reducción del **[insertar %]%** en el uso de memoria.
Adicionalmente, se demostró el procesamiento por bloques (*chunking*) con
bloques de 250 filas, simulando el manejo de datasets de gran escala.

### 4.3 Validación del Pipeline

El dataset procesado supera los siguientes controles de calidad implementados
en el notebook `02_Pipelines.ipynb`:

- Sin valores nulos (0 NaN en 1.000 × 10 = 10.000 valores)
- Sin columnas de varianza cero
- Variables numéricas escaladas (media ≈ 0)
- Columnas OneHot en rango binario [0, 1]

*(Insertar gráfico `pipeline_before_after.png`)*

---

## 5. Conclusiones y Recomendaciones

### Conclusiones

El flujo de preparación implementado logró transformar un dataset intencionalmente
ruidoso en una matriz matemática limpia y lista para modelado, sin perder ningún
registro del dataset original (1.000 filas preservadas). Las decisiones de
transformación están documentadas y justificadas por reglas de negocio concretas,
lo que permite reproducir el proceso completo en nuevos datos del mismo dominio
ejecutando una sola llamada: `pipeline.fit_transform(X)`.

La arquitectura modular adoptada (separación entre `src/`, `notebooks/` y `data/`)
demostró su valor al permitir iterar sobre los transformers individuales sin
afectar el resto del sistema, y al facilitar la colaboración en equipo.

### Lecciones Aprendidas

- La exploración del dato antes de codificar es imprescindible. El hallazgo de
  990 doctores únicos, detectado en el EDA, evitó un error grave de diseño en el
  pipeline (dimensionalidad explosiva por OneHotEncoding).
- El orden de los pasos en un Pipeline importa: el `SmartImputerTransformer` debe
  ir antes del `OutlierCapper` porque este último no puede operar con valores NaN.
- El uso de `format='mixed'` en `pd.to_datetime()` (Pandas 2.x) fue necesario para
  parsear los 4 formatos de fecha distintos en una sola columna.

### Mejoras Futuras

- **Imputación avanzada:** reemplazar la imputación por mediana/moda con KNN
  Imputer para variables con mayor porcentaje de nulos, capturando relaciones
  entre variables.
- **Encoding de doctor:** explorar Target Encoding o frecuency encoding para
  aprovechar la información del médico sin la explosión de cardinalidad.
- **Validación cruzada del pipeline:** integrar el pipeline en un `cross_val_score`
  para asegurar que no hay Data Leakage entre folds.
- **Conversión de divisas:** implementar conversión real de `billing_amount` usando
  tipos de cambio históricos según la fecha de la cita.