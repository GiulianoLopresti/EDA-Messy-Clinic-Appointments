"""
audit.py
--------
Módulo de auditoría de datos.
Verifica integridad y procedencia del dataset usando metadatos y hashing SHA-256.

Por qué existe este archivo:
    Antes de limpiar cualquier dato poder probar que el archivo 
    es exactamente el que se recibio originalmente. Si alguien modificó el CSV
    por error (o intencionalmente), el hash cambia y este módulo lo detecta.

IMPORTANTE:
    La función create_metadata_file() se ejecuta UNA SOLA VEZ cuando obtienes
    el CSV por primera vez. Después, solo usas verify_data_integrity().
"""

import hashlib
import logging
import os
import json
from typing import Optional, Dict

# Configuración del sistema de logs (registro de eventos)
# INFO  → mensajes informativos normales
# ERROR → algo salió mal pero el programa puede continuar
# CRITICAL → algo salió muy mal (ej. datos corruptos)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_checksum(file_path: str) -> Optional[str]:
    """
    Genera un hash SHA-256 del archivo indicado.

    Un hash SHA-256 es una "huella digital" única del archivo: si se cambia
    aunque sea un byte del contenido, el hash resultante es completamente
    diferente. Esto permite detectar cualquier modificación.

    Args:
        file_path: Ruta al archivo a hashear.

    Returns:
        String hexadecimal de 64 caracteres, o None si el archivo no existe.
    """

    try:
        # Abrimos en modo binario ('rb') porque el hash se calcula sobre bytes,
        # no sobre texto. Leer como texto podría alterar saltos de línea.
        with open(file_path, "rb") as file:
            file_bytes = file.read()
            return hashlib.sha256(file_bytes).hexdigest()
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado en: {file_path}")
        return None


def get_file_metadata(file_path: str) -> Optional[Dict]:
    """
    Obtiene el tamaño en MB y el hash SHA-256 de un archivo.

    Args:
        file_path: Ruta al archivo a inspeccionar.

    Returns:
        Diccionario con nombre, tamaño y checksum, o None si no existe.
    """
    logging.info(f"Extrayendo metadatos de {file_path}...")
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        file_hash = generate_checksum(file_path)

        return {
            "file_name": os.path.basename(file_path),
            "size_mb": round(size_mb, 4),
            "sha256_checksum": file_hash
        }
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado: {file_path}")
        return None


def create_metadata_file(file_path: str, metadata_path: str) -> None:
    """
    Crea el archivo oficial metadata.json (ejecutar solo una vez).

    Args:
        file_path:     Ruta al CSV original.
        metadata_path: Ruta donde guardar el metadata.json.
    """
    metadata = get_file_metadata(file_path)
    if metadata:
        # Creamos el directorio si no existe (evita errores si data/raw/ no existe aún)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        with open(metadata_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)
        logging.info(f"Metadatos oficiales guardados en: {metadata_path}")
        logging.info(f"Hash SHA-256 registrado: {metadata['sha256_checksum']}")
    else:
        logging.error("No se pudo crear el archivo de metadatos.")


def verify_data_integrity(file_path: str, metadata_path: str) -> bool:
    """
    Compara el hash actual del CSV con el registrado en metadata.json.

    Si los hashes coinciden → el archivo no fue modificado.
    Si difieren → alguien o algo alteró el archivo (corrupción o manipulación).

    Args:
        file_path:     Ruta al CSV que queremos verificar.
        metadata_path: Ruta al metadata.json con el hash oficial.

    Returns:
        True si la integridad es correcta, False si hay discrepancia.
    """
    logging.info(f"Verificando integridad de: {os.path.basename(file_path)}")

    # Paso 1: leer el hash oficial que guardamos antes
    try:
        with open(metadata_path, "r") as json_file:
            official_metadata = json.load(json_file)
            expected_hash = official_metadata.get("sha256_checksum")
    except FileNotFoundError:
        logging.error(
            f"metadata.json no encontrado en {metadata_path}. "
            "Ejecuta create_metadata_file() primero."
        )
        return False

    # Paso 2: calcular el hash del archivo en este momento
    current_hash = generate_checksum(file_path)

    # Paso 3: comparar ambos hashes
    if current_hash == expected_hash:
        logging.info("Integridad verificada. El dataset no ha sido modificado.")
        return True
    else:
        logging.critical("ALERTA: Discrepancia detectada en el dataset.")
        logging.critical(f"  Hash esperado: {expected_hash}")
        logging.critical(f"  Hash actual:   {current_hash}")
        return False


# --- Ejecución directa (para pruebas desde terminal) ------------------
if __name__ == "__main__":
    # Importamos las rutas desde config para no escribirlas a mano
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import RAW_CSV, METADATA_JSON

    # Si no existe el JSON todavía, lo creamos
    if not os.path.exists(METADATA_JSON):
        logging.info("No se encontró metadata.json. Generando por primera vez...")
        create_metadata_file(str(RAW_CSV), str(METADATA_JSON))

    # Verificamos la integridad del CSV crudo
    verify_data_integrity(str(RAW_CSV), str(METADATA_JSON))