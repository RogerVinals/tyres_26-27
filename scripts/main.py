# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

import os
import sys
import yaml
import logging
from pathlib import Path
import pandas as pd
from tyres_ttc.parsing import cargar_multiples_runs
from tyres_ttc.processing import (
    clasificar_datos_ttc,
    filtrar_para_ajuste_mf,
    cargar_umbrales,
    WARMUP_FRAC_PARADO,
    WARMUP_MAX_SA,
    WARMUP_MAX_STD_SL,
    SL_DISPERSION_BAJA,
    SA_MEDIANA_RECTA,
)
from tyres_ttc.plotting import (
    graficar_desde_parquet,
    graficar_analisis,
    graficar_analisis_global_todos,
)

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
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


def _preguntar_ajuste_mf() -> bool:
    """
    Pregunta si se quiere fittear.
    - Si la variable de entorno MF_FIT está definida, usa su valor (sin preguntar).
    - Si no está definida y hay terminal interactivo, pregunta por input().
    - Si no hay terminal (CI/servidor), omite el ajuste.
    """
    env = os.environ.get("MF_FIT", "").strip().lower()
    if env in ("1", "si", "sí", "s", "yes", "y"):
        return True
    if env in ("0", "no", "false", "n"):
        return False

    # Variable no definida: intentar input interactivo
    if not sys.stdin.isatty():
        logger.info("No hay terminal interactivo: se omite el ajuste MF5.2.")
        return False

    respuesta = input(
        "\n¿Ajustar coeficientes MF5.2 y generar .tir? (si/no): "
    ).strip().lower()
    return respuesta in ("si", "sí", "s", "yes", "y")


def main() -> None:
    """Orquesta el pipeline de carga, guardado, visualización y ajuste MF."""
    logger.info("Iniciando pipeline de procesamiento...")

    # 1. Config
    config = load_config()
    data_dir = Path(config['paths']['data_dir'])
    results_dir = Path(config['paths']['results_dir'])
    plots_dir = Path(config['paths']['plots_dir'])
    patron_archivos = config['settings'].get('file_pattern', "*.dat")

    # Umbrales de warmup/clasificación desde config
    umbrales = cargar_umbrales()

    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # 2. Cargar raw data
    df_crudo = cargar_multiples_runs(data_dir, patron=patron_archivos)

    # 3. Clasificar (elimina warmup internamente) con umbrales del config
    df_procesado = clasificar_datos_ttc(
        df_crudo,
        sl_dispersion_baja=umbrales.get('sl_dispersion_baja', SL_DISPERSION_BAJA),
        sa_mediana_recta=umbrales.get('sa_mediana_recta', SA_MEDIANA_RECTA),
        warmup_frac_parado=umbrales.get('warmup_frac_parado', WARMUP_FRAC_PARADO),
        warmup_max_sa=umbrales.get('warmup_max_sa', WARMUP_MAX_SA),
        warmup_max_std_sl=umbrales.get('warmup_max_std_sl', WARMUP_MAX_STD_SL),
    )

    # Resumen
    logger.info("Resumen por Tipo de Test:\n%s",
                df_procesado['test_type'].value_counts())
    logger.info("Resumen por FZ_nom:\n%s",
                df_procesado['FZ_nom'].value_counts().sort_index())

    # 4. Guardar parquet (una sola vez)
    salida = results_dir / "dataset_clasificado.parquet"
    df_procesado.to_parquet(salida, index=False)
    logger.info("Dataset guardado: %s", salida)

    # 5. Gráficas (reusa df_procesado en memoria para no leer el parquet dos veces)
    logger.info("Generando gráficas raw por run...")
    graficar_desde_parquet(salida, plots_dir, df=df_procesado)

    logger.info("Generando gráficas de análisis global...")
    df_lateral = df_procesado[df_procesado['test_type'] == 'Pure Cornering']
    if not df_lateral.empty:
        graficar_analisis(df_lateral, "SA", "FY", "FZ_nom", plots_dir)

    df_longitudinal = df_procesado[
        df_procesado['test_type'] == 'Pure Longitudinal'
    ]
    if not df_longitudinal.empty:
        col_sl = 'SL_effective' if 'SL_effective' in df_procesado.columns else 'SL'
        graficar_analisis(df_longitudinal, col_sl, "FX", "FZ_nom", plots_dir)

    graficar_analisis_global_todos(df_procesado, plots_dir)

    # 6. Ajuste opcional MF5.2 (un único .tir con todos los datos)
    if _preguntar_ajuste_mf():
        logger.info("Iniciando ajuste MF5.2 con todos los datos...")
        from tyres_ttc.fit_ttc import main as fit_main
        df_mf = filtrar_para_ajuste_mf(df_procesado)
        fit_main(
            parquet_path=salida,
            tir_path=results_dir / "coeficientes_mf52.tir",
            verbose=0,
            df_mf=df_mf,
        )
        logger.info(".tir generado: %s", results_dir / "coeficientes_mf52.tir")
    else:
        logger.info("Ajuste MF5.2 omitido por el usuario.")

    logger.info("Pipeline completado exitosamente.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Error crítico: %s", e)
        raise
