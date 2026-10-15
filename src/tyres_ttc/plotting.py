import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

COL_SL = 'SL_effective'  # columna de slip preferida


def _col_sl(df: pd.DataFrame) -> str:
    """Devuelve la columna de slip disponible."""
    return COL_SL if COL_SL in df.columns else 'SL'


def _guardar_figura(fig: plt.Figure, ruta: Path, dpi: int = 150) -> None:
    """Guarda y cierra la figura siempre para evitar memory leaks."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plotting] Gráfica guardada: {ruta}")


def graficar_desde_parquet(
    archivo_parquet: Path,
    directorio_salida: Path,
    agrupar_por: str = "run_id",
    df: Optional[pd.DataFrame] = None,
) -> None:
    """Gráficas raw por run según su clasificación.

    Si se pasa `df`, se usa directamente en lugar de leer el parquet.
    """
    if df is not None:
        df_plot = df
    else:
        if not archivo_parquet.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {archivo_parquet}")
        print(f"Leyendo datos desde {archivo_parquet.name} para graficar...")
        df_plot = pd.read_parquet(archivo_parquet)

    directorio_salida.mkdir(exist_ok=True)

    col_sl = _col_sl(df_plot)

    for run_id, grupo in df_plot.groupby(agrupar_por):
        tipo_ensayo = grupo['test_type'].iloc[0]

        if tipo_ensayo in ('Pure Longitudinal', 'Combined'):
            col_x, col_y = col_sl, 'FX'
            titulo_base = "Fuerza Longitudinal vs Slip Ratio"
            xlabel = "Slip Ratio (SL)"
        else:
            col_x, col_y = 'SA', 'FY'
            titulo_base = "Fuerza Lateral vs Ángulo de Deslizamiento"
            xlabel = "SA (deg)"

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(
            grupo[col_x], grupo[col_y],
            s=8, c="darkorange", alpha=0.6,
            label=f"Raw data ({len(grupo)} ptos)"
        )
        ax.set_title(f"{titulo_base} — {run_id}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{col_y} (N)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        _guardar_figura(
            fig,
            directorio_salida / f"plot_raw_{col_y}_vs_{col_x}_{run_id}.png",
        )


def graficar_analisis(
    df: pd.DataFrame,
    columna_x: str,
    columna_y: str,
    agrupar_por: str = "FZ_nom",
    directorio_salida: Optional[Path] = None,
) -> plt.Figure:
    """Gráfica diferenciando por FZ_nom. Siempre cierra la figura al guardar."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for valor in sorted(df[agrupar_por].unique()):
        datos_grupo = df[df[agrupar_por] == valor]
        ax.scatter(
            datos_grupo[columna_x], datos_grupo[columna_y],
            s=10, alpha=0.6, label=f"FZ = {valor} N"
        )

    ax.set_title(f"{columna_y} vs {columna_x} (Diferenciado por FZ)")
    ax.set_xlabel(columna_x)
    ax.set_ylabel(columna_y)
    ax.legend(title="Carga Vertical")
    ax.grid(True, alpha=0.3)

    if directorio_salida:
        _guardar_figura(
            fig,
            directorio_salida / f"analysis_{columna_y}_vs_{columna_x}.png",
        )

    return fig


def _graficar_uno_en_uno(
    df: pd.DataFrame,
    columna_x: str,
    columna_y: str,
    tipos: tuple,
    nombre_archivo: str,
    titulo: str,
    directorio_salida: Path,
) -> None:
    """Helper interno: gráfico todo-en-uno para un par de columnas y tipos."""
    sub = df[df['test_type'].isin(tipos)].copy()
    if sub.empty:
        print(f"[plotting] No hay datos de {nombre_archivo} para graficar.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for fz in sorted(sub['FZ_nom'].unique()):
        g = sub[sub['FZ_nom'] == fz]
        ax.scatter(g[columna_x], g[columna_y], s=6, alpha=0.45, label=f"FZ = {fz} N")

    ax.set_title(titulo)
    ax.set_xlabel(columna_x)
    ax.set_ylabel(columna_y)
    ax.legend(title="Carga Vertical", fontsize=8)
    ax.grid(True, alpha=0.3)

    _guardar_figura(fig, directorio_salida / nombre_archivo)


def graficar_analisis_global_todos(
    df: pd.DataFrame,
    directorio_salida: Optional[Path] = None,
) -> None:
    """
    Dos gráficas todo-en-uno:
      1. FX vs SL  → Pure Longitudinal + Combined
      2. FY vs SA  → Pure Cornering + Combined
    """
    if directorio_salida is None:
        directorio_salida = Path("plots")
    directorio_salida.mkdir(exist_ok=True)

    col_sl = _col_sl(df)

    _graficar_uno_en_uno(
        df,
        col_sl, 'FX',
        ('Pure Longitudinal', 'Combined'),
        "analysis_FX_vs_SL_todos.png",
        "FX vs SL — Todo en uno (Pure Longitudinal + Combined)",
        directorio_salida,
    )

    _graficar_uno_en_uno(
        df,
        'SA', 'FY',
        ('Pure Cornering', 'Combined'),
        "analysis_FY_vs_SA_todos.png",
        "FY vs SA — Todo en uno (Pure Cornering + Combined)",
        directorio_salida,
    )
