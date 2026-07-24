import pandas as pd
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
