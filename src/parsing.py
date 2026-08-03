# src/parsing.py
import pandas as pd
from typing import List
from pathlib import Path


def cargar_dat_ttc(filepath: str) -> pd.DataFrame:
    """Carga archivos .dat del TTC ignorando metadatos y unidades."""
    ruta = Path(filepath)
    if not ruta.exists():
        raise FileNotFoundError(f"[Error] No se encontró el archivo: {ruta}")

    print(f"Cargando telemetría TTC: {ruta.name}...")

    # sep='\s+' separa por tabulaciones/espacios.
    # skiprows=[0, 2] ignora el título y las unidades, dejando las variables como columnas.
    df = pd.read_csv(ruta, sep=r'\s+', skiprows=[0, 2])

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
        raise FileNotFoundError(
            f"No se encontraron archivos con el patrón '{patron}' en {directorio}"
        )

    dataframes: List[pd.DataFrame] = []

    for archivo in archivos:
        df_run = cargar_dat_ttc(archivo)

        if df_run is None:
            raise ValueError(
                f"cargar_dat_ttc() devolvió None al procesar '{archivo.name}'. "
                f"Revisa el formato de ese archivo o la lógica de parsing."
            )
        if df_run.empty:
            raise ValueError(
                f"El archivo '{archivo.name}' se cargó pero el DataFrame está vacío."
            )

        df_run[columna_id] = archivo.stem
        dataframes.append(df_run)

    df_final = pd.concat(dataframes, ignore_index=True)
    return df_final
