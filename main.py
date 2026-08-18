# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

import yaml
import logging
from pathlib import Path
import pandas as pd
from src.parsing import cargar_multiples_runs
from src.processing import clasificar_datos_ttc
from src.plotting import graficar_desde_parquet, graficar_analisis

# Configuración de logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "fsae_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config_ttc.yaml") -> dict:
    """Carga la configuración desde un archivo YAML."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main() -> None:
    """Orquesta el pipeline de carga, guardado y visualización."""
    logger.info("Iniciando pipeline de procesamiento...")
    
    # 1. Cargar configuración
    config = load_config()
    
    # 2. Definición de rutas y ajustes desde configuración
    data_dir = Path(config['paths']['data_dir'])
    results_dir = Path(config['paths']['results_dir'])
    plots_dir = Path(config['paths']['plots_dir'])
    patron_archivos = config['settings']['file_pattern']
    
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # 3. Cargar datos crudos
    df_crudo = cargar_multiples_runs(data_dir, patron=patron_archivos)

    # 4. Clasificar los datos
    df_procesado = clasificar_datos_ttc(df_crudo)

    # Resumen de datos
    logger.info("Resumen de datos por Tipo de Test:\n%s", df_procesado['test_type'].value_counts())
    logger.info("Resumen de datos por Carga Vertical (FZ nominal):\n%s", df_procesado['FZ_nom'].value_counts().sort_index())

    # 5. Guardado del resultado
    salida = results_dir / "dataset_clasificado.parquet"
    df_procesado.to_parquet(salida, index=False)
    logger.info("Dataset procesado guardado en: %s", salida)

    # 6. Generación de gráficas
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
    
    logger.info("Pipeline completado exitosamente.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Error crítico durante la ejecución del pipeline: %s", e)
