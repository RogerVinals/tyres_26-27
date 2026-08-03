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

def graficar_analisis(
    df: pd.DataFrame,
    columna_x: str,
    columna_y: str,
    agrupar_por: str = "FZ_nom",
    directorio_salida: Path = None
) -> plt.Figure:
    """Genera una gráfica diferenciando por FZ_nom."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    grupos = sorted(df[agrupar_por].unique())
    
    for valor in grupos:
        datos_grupo = df[df[agrupar_por] == valor]
        
        ax.scatter(
            datos_grupo[columna_x], 
            datos_grupo[columna_y], 
            s=10, 
            alpha=0.6,
            label=f"FZ = {valor} N"
        )
        
    ax.set_title(f"{columna_y} vs {columna_x} (Diferenciado por FZ)")
    ax.set_xlabel(columna_x)
    ax.set_ylabel(columna_y)
    
    if len(grupos) > 0:
        ax.legend(title="Carga Vertical")
        
    ax.grid(True, alpha=0.3)
    
    if directorio_salida:
        directorio_salida.mkdir(exist_ok=True)
        ruta_salida = directorio_salida / f"analysis_{columna_y}_vs_{columna_x}.png"
        fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
        print(f"[plotting] Gráfica guardada: {ruta_salida}")
        
    return fig
