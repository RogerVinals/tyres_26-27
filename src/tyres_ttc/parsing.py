# src/parsing.py
import pandas as pd
from typing import List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def cargar_dat_ttc(filepath: Path) -> pd.DataFrame:
    """Carga archivos .dat del TTC ignorando metadatos y unidades."""
    if not filepath.exists():
        msg = f"No se encontró el archivo: {filepath}"
        logger.error(msg)
        raise FileNotFoundError(f"[Error] {msg}")

    logger.info(f"Cargando telemetría TTC: {filepath.name}...")

    # sep='\s+' separa por tabulaciones/espacios.
    # skiprows=[0, 2] ignora el título y las unidades, dejando las variables como columnas.
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=[0, 2])

    return df


def cargar_multiples_runs(
    directorio: Path,
    patron: str = "*.dat",
    columna_id: str = "run_id",
) -> pd.DataFrame:
    """
    Carga y concatena múltiples archivos .dat del TTC en un único DataFrame.

    Cada fila queda etiquetada con el nombre del archivo de origen (sin
    extensión) en la columna `columna_id`.
    """
    archivos: List[Path] = sorted(directorio.glob(patron))

    if not archivos:
        msg = f"No se encontraron archivos con el patrón '{patron}' en {directorio}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    dataframes: List[pd.DataFrame] = []

    for archivo in archivos:
        df_run = cargar_dat_ttc(archivo)

        if df_run.empty:
            msg = f"El archivo '{archivo.name}' se cargó pero el DataFrame está vacío."
            logger.error(msg)
            raise ValueError(msg)

        df_run[columna_id] = archivo.stem
        dataframes.append(df_run)

    df_final = pd.concat(dataframes, ignore_index=True)
    return df_final
