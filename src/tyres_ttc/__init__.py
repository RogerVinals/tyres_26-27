# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

"""tyres_ttc - Procesador de datos de neumáticos TTC para Magic Formula."""

from .parsing import cargar_dat_ttc, cargar_multiples_runs
from .processing import (
    clasificar_datos_ttc,
    limpiar_datos,
    filtrar_datos,
    filtrar_para_ajuste_mf,
    detectar_runs_warmup,
    marcar_warmup_despues_de_calibracion,
)
from .plotting import (
    graficar_desde_parquet,
    graficar_analisis,
    graficar_analisis_global_todos,
)
from .pacejka_mf52 import (
    MF52Coeffs,
    pure_slip_fx,
    pure_slip_fy,
    pure_slip_mz,
    combined_slip,
)
from .fit_ttc import (
    ajustar_coeficientes_mf52,
    guardar_tir,
    main as fit_main,
)

__all__ = [
    "cargar_dat_ttc",
    "cargar_multiples_runs",
    "clasificar_datos_ttc",
    "limpiar_datos",
    "filtrar_datos",
    "filtrar_para_ajuste_mf",
    "detectar_runs_warmup",
    "marcar_warmup_despues_de_calibracion",
    "graficar_desde_parquet",
    "graficar_analisis",
    "graficar_analisis_global_todos",
    "MF52Coeffs",
    "pure_slip_fx",
    "pure_slip_fy",
    "pure_slip_mz",
    "combined_slip",
    "ajustar_coeficientes_mf52",
    "guardar_tir",
    "fit_main",
]
