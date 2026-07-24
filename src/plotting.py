## src/plotting.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def graficar_desde_parquet(
    archivo_parquet: Path,
    directorio_salida: Path,
    columna_x: str = "SA",
    columna_y: str = "FY",
    agrupar_por: str = "run_id"
) -> None:
    """Lee un dataset en .parquet y genera una gráfica de dispersión por run."""
    
    if not archivo_parquet.exists():
        raise FileNotFoundError(f"[Error] No se encontró el archivo: {archivo_parquet}")
        
    print(f"Leyendo datos desde {archivo_parquet.name}...")
    df = pd.read_parquet(archivo_parquet)
    
    directorio_salida.mkdir(exist_ok=True)
    runs = df[agrupar_por].unique()

    for run_id in runs:
        grupo = df[df[agrupar_por] == run_id]

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(
            grupo[columna_x],
            grupo[columna_y],
            s=8,
            c="darkorange",
            alpha=0.6,
            label=f"Datos crudos ({len(grupo)} ptos)"
        )

        ax.set_title(f"Fuerza Lateral vs Ángulo de Deslizamiento — {run_id}")
        ax.set_xlabel(f"{columna_x} (deg)")
        ax.set_ylabel(f"{columna_y} (N)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ruta_salida = directorio_salida / f"plot_raw_{run_id}.png"
        fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[plotting] Gráfica guardada: {ruta_salida}")
