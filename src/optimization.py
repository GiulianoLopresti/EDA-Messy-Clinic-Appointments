"""
optimization.py
---------------
Módulo de optimización de memoria y procesamiento.

Por qué existe este archivo:
    Los DataFrames de Pandas usan por defecto los tipos de datos más "grandes"
    posibles (int64, float64). Esto desperdicia memoria cuando los valores son
    pequeños. Este módulo reduce ese gasto sin perder información.

    También demuestra el procesamiento por bloques (chunks), aunque
    nuestro dataset sea pequeño.
"""
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce el uso de memoria del DataFrame haciendo 'downcasting' de tipos.

    Downcasting significa convertir un tipo grande a uno más pequeño cuando
    los valores caben en él. Ejemplo: la edad máxima es 90 → no necesita int64
    (que soporta hasta 9 trillones), puede usar int8 (soporta hasta 127).

    Args:
        df: DataFrame original a optimizar.

    Returns:
        DataFrame con los mismos datos pero usando menos memoria.
    """
    
    logging.info("Iniciando optimización de memoria (downcasting)...")

    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logging.info(f"Memoria inicial: {start_mem:.4f} MB")

    df_optimized = df.copy()

    # Columnas enteras: reducir de int64/int32 al tipo mínimo que las contenga
    int_columns = df_optimized.select_dtypes(include=["int64", "int32"]).columns
    for col in int_columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="integer")

    # Columnas decimales: reducir de float64 a float32
    float_columns = df_optimized.select_dtypes(include=["float64"]).columns
    for col in float_columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

    end_mem = df_optimized.memory_usage(deep=True).sum() / 1024 ** 2
    reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0

    logging.info(f"Memoria optimizada: {end_mem:.4f} MB")
    logging.info(f"Reducción lograda:  {reduction:.1f}%")

    return df_optimized


def process_large_file_in_chunks(
    file_path: str,
    chunk_size: int = 10_000,
    sep: str = ","
) -> int:
    """
    Lee y procesa un archivo CSV en bloques para no saturar la RAM.

    Por qué es útil:
        Un CSV de 10 GB no cabe entero en RAM. Al leerlo en bloques de
        10.000 filas, procesamos cada bloque y lo descartamos antes de
        cargar el siguiente.

    Args:
        file_path:  Ruta al archivo CSV.
        chunk_size: Número de filas por bloque.
        sep:        Separador del CSV (por defecto coma).

    Returns:
        Total de filas procesadas.
    """
    logging.info(f"Procesando {file_path} en bloques de {chunk_size} filas...")

    total_rows = 0
    try:
        # chunksize hace que Pandas devuelva un iterador en lugar de cargar todo
        chunk_iterator = pd.read_csv(file_path, sep=sep, chunksize=chunk_size)

        for i, chunk in enumerate(chunk_iterator):
            rows_in_chunk = len(chunk)
            total_rows += rows_in_chunk
            logging.info(f"  Bloque {i + 1}: {rows_in_chunk} filas procesadas.")
            # Aquí iría la lógica de transformación por bloque si fuera necesario

        logging.info(f"Procesamiento completado. Total de filas: {total_rows}")
        return total_rows

    except FileNotFoundError:
        logging.error(f"Archivo no encontrado: {file_path}")
        return 0


# --- Ejecución directa ------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import RAW_CSV

    # Prueba 1: procesamiento por bloques
    process_large_file_in_chunks(str(RAW_CSV), chunk_size=250)

    # Prueba 2: downcasting sobre el dataset completo
    try:
        df_test = pd.read_csv(str(RAW_CSV))
        optimize_memory_usage(df_test)
    except FileNotFoundError:
        logging.error("CSV no encontrado para la prueba de downcasting.")