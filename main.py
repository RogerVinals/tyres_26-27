# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

# main.py
# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

from pathlib import Path
import pandas as pd
from src.parsing import cargar_multiples_runs
from src.processing import clasificar_datos_ttc
from src.plotting import graficar_desde_parquet, graficar_analisis


def main() -> None:
    """Orquesta el pipeline de carga, guardado y visualización."""

    data_dir = Path("data")
    results_dir = Path("results")
    plots_dir = Path("plots")
    
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    patron_archivos = "*.dat"

    # 1. Cargar raw data
    df_crudo = cargar_multiples_runs(data_dir, patron=patron_archivos)

    # 2. Clasificar los datos
    df_procesado = clasificar_datos_ttc(df_crudo)

    # Mostramos un resumen por pantalla de lo que ha clasificado
    print("\nResumen de datos por Tipo de Test:")
    print(df_procesado['test_type'].value_counts())
    
    print("\nResumen de datos por Carga Vertical (FZ nominal):")
    print(df_procesado['FZ_nom'].value_counts().sort_index())

    # 3. Guardado de resultados
    salida = results_dir / "dataset_clasificado.parquet"
    df_procesado.to_parquet(salida, index=False)
    print(f"\nDataset procesado guardado en: {salida}\n")

    # 4. Creación de gráficas
    df_plot = pd.read_parquet(salida)
    graficar_desde_parquet(
        archivo_parquet=salida,
        directorio_salida=plots_dir
    )

    # Gráficas de análisis
    graficar_analisis(
        df_plot,
        columna_x="SA",
        columna_y="FY",
        agrupar_por="FZ_nom",
        directorio_salida=plots_dir
    )
    graficar_analisis(
        df_plot,
        columna_x="SL",
        columna_y="FX",
        agrupar_por="FZ_nom",
        directorio_salida=plots_dir
    )

if __name__ == "__main__":
    main()
