import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def graficar_desde_parquet(
    archivo_parquet: Path,
    directorio_salida: Path,
    agrupar_por: str = "run_id"
) -> None:
    """Lee un dataset en .parquet y genera la gráfica adecuada según el tipo de ensayo predominante."""
    
    if not archivo_parquet.exists():
        raise FileNotFoundError(f"[Error] No se encontró el archivo: {archivo_parquet}")
        
    print(f"Leyendo datos desde {archivo_parquet.name} para graficar...")
    df = pd.read_parquet(archivo_parquet)
    
    directorio_salida.mkdir(exist_ok=True)
    runs = df[agrupar_por].unique()
    
    # Identificar cuál es nuestra columna de Slip corregida
    col_sl = 'SL_effective' if 'SL_effective' in df.columns else 'SL'

    
    for run_id in runs:
        grupo = df[df[agrupar_por] == run_id]

        # Leemos la clasificación que ya hizo processing.py
        tipo_ensayo = grupo['test_type'].iloc[0]

        if tipo_ensayo in ['Pure Longitudinal', 'Combined']:
            col_x, col_y = col_sl, 'FX'
            titulo_base = "Fuerza Longitudinal vs Slip Ratio"
            xlabel = "Slip Ratio (SL)"
        else:
            col_x, col_y = 'SA', 'FY'
            titulo_base = "Fuerza Lateral vs Ángulo de Deslizamiento"
            xlabel = "SA (deg)"

        fig, ax = plt.subplots(figsize=(9, 6))
        # ... (el resto del código de plt.scatter sigue igual)

        ax.scatter(
            grupo[col_x],
            grupo[col_y],
            s=8,
            c="darkorange",
            alpha=0.6,
            label=f"Raw data ({len(grupo)} ptos)"
        )

        ax.set_title(f"{titulo_base} — {run_id}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{col_y} (N)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ruta_salida = directorio_salida / f"plot_raw_{col_y}_vs_{col_x}_{run_id}.png"
        fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[plotting] Gráfica guardada: {ruta_salida.name}")

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
