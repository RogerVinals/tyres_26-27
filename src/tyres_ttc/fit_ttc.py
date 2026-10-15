# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

"""
fit_ttc.py
==========
Ajuste de coeficientes Pacejka Magic Formula 5.2 a datos TTC filtrados
mediante mínimos cuadrados no lineales (scipy.optimize.least_squares).

Genera un archivo .tir con todos los coeficientes MF5.2.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from .pacejka_mf52 import (
    MF52Coeffs,
    combined_slip,
    pure_slip_fx,
    pure_slip_fy,
    pure_slip_mz,
)

from .processing import filtrar_para_ajuste_mf

logger = logging.getLogger(__name__)

DEG2RAD = np.pi / 180.0

# ============================================================================
# Coeficientes por etapa de ajuste
# ============================================================================

FY0_PARAMS = [
    "pCy1", "pDy1", "pDy2", "pKy1", "pKy2", "pKy3",
    "pEy1", "pEy2", "pHy1", "pHy2", "pVy1", "pVy2", "pVy3", "pVy4",
]

FX0_PARAMS = [
    "pCx1", "pDx1", "pDx2", "pEx1", "pEx2", "pKx1", "pKx2",
    "pHx1", "pVx1", "pVx2",
]

MZ0_PARAMS = [
    "qBz1", "qBz2", "qBz3", "qBz4", "qCz1", "qDz1", "qDz2",
    "qEz1", "qEz2", "qHz1", "qHz2", "qDz6", "qDz7",
]

COMBINED_PARAMS = [
    "rBx1", "rBx2", "rCx1", "rEx1", "rEx2",
    "rBy1", "rBy2", "rBy3", "rCy1", "rEy1", "rEy2",
    "rHy1", "rHy2", "rVy1", "rVy2", "rVy3", "rVy4", "rVy5", "rVy6",
]

# Puntos máximos por etapa (subsampling para velocidad)
# None = usar todos los datos (sin downsampling)
# Valores por defecto; se sobrescriben si se pasa config_ttc.yaml
MAX_PUNTOS = {
    "fy0": None,
    "mz0": None,
    "fx0": None,
    "combined": None,
}
}


def cargar_config_fitting(config_path: str | Path = "config_ttc.yaml") -> dict:
    """Carga parámetros de fitting desde config_ttc.yaml."""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        fitting = cfg.get('fitting', {})
        return {
            'max_puntos_fy0': fitting.get('max_puntos_fy0', MAX_PUNTOS['fy0']),
            'max_puntos_mz0': fitting.get('max_puntos_mz0', MAX_PUNTOS['mz0']),
            'max_puntos_fx0': fitting.get('max_puntos_fx0', MAX_PUNTOS['fx0']),
            'max_puntos_combined': fitting.get('max_puntos_combined', MAX_PUNTOS['combined']),
            'verbose': fitting.get('verbose', 0),
        }
    except (FileNotFoundError, KeyError, TypeError):
        return {
            'max_puntos_fy0': MAX_PUNTOS['fy0'],
            'max_puntos_mz0': MAX_PUNTOS['mz0'],
            'max_puntos_fx0': MAX_PUNTOS['fx0'],
            'max_puntos_combined': MAX_PUNTOS['combined'],
            'verbose': 0,
        }


# -------------------------------------------------------------------------
# Progreso visual en tiempo real (spinner + contador)
# -------------------------------------------------------------------------

class _ProgresoFit:
    """
    Muestra un spinner en la misma línea de la consola mientras el fitting
    está en ejecución, para que no parezca que el programa está parado.
    Se ejecuta en un hilo daemon separado.
    """

    def __init__(self, nombre: str):
        self.nombre = nombre
        self.activo = False
        self.hilo: Optional[threading.Thread] = None
        self.eval_count = 0
        self.coste_actual = 0.0
        self.t0 = 0.0
        self._chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._i = 0

    def iniciar(self) -> None:
        self.activo = True
        self.eval_count = 0
        self.coste_actual = 0.0
        self.t0 = time.time()
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _bucle(self) -> None:
        while self.activo:
            elapsed = time.time() - self.t0
            print(
                f"\r{self.nombre} {self._chars[self._i % len(self._chars)]} "
                f"eval={self.eval_count:>5d} coste={self.coste_actual:.4e} "
                f"t={elapsed:5.1f}s",
                end="",
                flush=True,
            )
            time.sleep(0.2)
            self._i += 1

    def actualizar(self, n_eval: int, coste: float) -> None:
        self.eval_count = n_eval
        self.coste_actual = coste

    def detener(self, mensaje_final: str) -> None:
        self.activo = False
        if self.hilo is not None:
            self.hilo.join(timeout=0.5)
        print(f"\r{mensaje_final}" + " " * 30, flush=True)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _subsample(arrs: list, max_puntos: int) -> list:
    """Muestreo uniforme si excede max_puntos."""
    if len(arrs[0]) <= max_puntos:
        return arrs
    idx = np.linspace(0, len(arrs[0]) - 1, max_puntos).astype(int)
    resultado = []
    for a in arrs:
        if isinstance(a, pd.DataFrame):
            resultado.append(a.iloc[idx].reset_index(drop=True))
        else:
            resultado.append(np.asarray(a)[idx])
    return resultado


def _preparar_datos(
    df: pd.DataFrame,
    test_type: str,
    max_puntos: Optional[int] = None,
):
    """Filtra tipo de ensayo y devuelve arrays listos para Pacejka."""
    sub = df[df["test_type"] == test_type].copy()
    if sub.empty:
        return None

    alpha = sub["SA"].values * DEG2RAD
    kappa = sub["SL_effective"].values
    Fz = (-sub["FZ"].values).clip(min=1.0)  # SAE FZ negativo → positivo
    gamma = sub["IA"].values * DEG2RAD

    if max_puntos is not None:
        alpha, kappa, Fz, gamma, sub = _subsample(
            [alpha, kappa, Fz, gamma, sub], max_puntos
        )

    return alpha, kappa, Fz, gamma, sub


def _set_coeffs(c: MF52Coeffs, params: dict) -> MF52Coeffs:
    for k, v in params.items():
        if hasattr(c, k):
            setattr(c, k, float(v))
    return c


def _get_coeffs(c: MF52Coeffs, names: list) -> np.ndarray:
    return np.array([getattr(c, n) for n in names], dtype=float)


# -------------------------------------------------------------------------
# Ajustes por régimen
# -------------------------------------------------------------------------

def fit_fy0(
    df: pd.DataFrame,
    c: MF52Coeffs,
    verbose: int = 0,
) -> MF52Coeffs:
    """Ajusta coeficientes de fuerza lateral pura (Fy0) con datos Pure Cornering."""
    datos = _preparar_datos(df, "Pure Cornering", MAX_PUNTOS["fy0"])
    if datos is None:
        logger.warning("fit_fy0: sin datos Pure Cornering.")
        return c

    alpha, _, Fz, gamma, sub = datos
    FY = sub["FY"].values

    x0 = _get_coeffs(c, FY0_PARAMS)

    print(f">>> FY0 (Pure Cornering) — {len(FY)} puntos, {len(FY0_PARAMS)} coeficientes")
    progreso = _ProgresoFit("FY0")
    progreso.iniciar()

    contador = {"n": 0}

    def residuals(x):
        contador["n"] += 1
        c_t = _set_coeffs(c, dict(zip(FY0_PARAMS, x)))
        Fy0, _ = pure_slip_fy(alpha, Fz, gamma, c_t)
        res = Fy0 - FY
        # Actualizar progreso con coste parcial (RMS de residuales actuales)
        progreso.actualizar(contador["n"], float(np.sqrt(np.mean(res ** 2))))
        return res

    res = least_squares(residuals, x0, max_nfev=5000, verbose=verbose)

    progreso.detener(
        f"<<< FY0 OK — RMSE={float(np.sqrt(np.mean(res.fun ** 2))):.2f} N, "
        f"coste={res.cost:.4e}"
    )
    c = _set_coeffs(c, dict(zip(FY0_PARAMS, res.x)))
    logger.info("fit_fy0: RMSE=%.2f N, coste=%.4f, puntos=%d",
                float(np.sqrt(np.mean(res.fun ** 2))), res.cost, len(FY))
    return c


def fit_mz0(
    df: pd.DataFrame,
    c: MF52Coeffs,
    verbose: int = 0,
) -> MF52Coeffs:
    """Ajusta coeficientes de momento autoalignante puro (Mz0) con datos Pure Cornering."""
    datos = _preparar_datos(df, "Pure Cornering", MAX_PUNTOS["mz0"])
    if datos is None:
        logger.warning("fit_mz0: sin datos Pure Cornering.")
        return c

    alpha, _, Fz, gamma, sub = datos
    MZ = sub["MZ"].values

    x0 = _get_coeffs(c, MZ0_PARAMS)

    print(f">>> MZ0 (Pure Cornering) — {len(MZ)} puntos, {len(MZ0_PARAMS)} coeficientes")
    progreso = _ProgresoFit("MZ0")
    progreso.iniciar()

    contador = {"n": 0}

    def residuals(x):
        contador["n"] += 1
        c_t = _set_coeffs(c, dict(zip(MZ0_PARAMS, x)))
        Mz0, _ = pure_slip_mz(alpha, Fz, gamma, c_t)
        res = Mz0 - MZ
        progreso.actualizar(contador["n"], float(np.sqrt(np.mean(res ** 2))))
        return res

    res = least_squares(residuals, x0, max_nfev=5000, verbose=verbose)

    progreso.detener(
        f"<<< MZ0 OK — RMSE={float(np.sqrt(np.mean(res.fun ** 2))):.2f} N*m, "
        f"coste={res.cost:.4e}"
    )
    c = _set_coeffs(c, dict(zip(MZ0_PARAMS, res.x)))
    logger.info("fit_mz0: RMSE=%.2f N*m, coste=%.4f, puntos=%d",
                float(np.sqrt(np.mean(res.fun ** 2))), res.cost, len(MZ))
    return c


def fit_fx0(
    df: pd.DataFrame,
    c: MF52Coeffs,
    verbose: int = 0,
) -> MF52Coeffs:
    """Ajusta coeficientes de fuerza longitudinal pura (Fx0) con datos Pure Longitudinal."""
    datos = _preparar_datos(df, "Pure Longitudinal", MAX_PUNTOS["fx0"])
    if datos is None:
        logger.warning("fit_fx0: sin datos Pure Longitudinal (usa valores por defecto).")
        return c

    _, kappa, Fz, gamma, sub = datos
    FX = sub["FX"].values

    x0 = _get_coeffs(c, FX0_PARAMS)

    print(f">>> FX0 (Pure Longitudinal) — {len(FX)} puntos, {len(FX0_PARAMS)} coeficientes")
    progreso = _ProgresoFit("FX0")
    progreso.iniciar()

    contador = {"n": 0}

    def residuals(x):
        contador["n"] += 1
        c_t = _set_coeffs(c, dict(zip(FX0_PARAMS, x)))
        Fx0, _ = pure_slip_fx(kappa, Fz, gamma, c_t)
        res = Fx0 - FX
        progreso.actualizar(contador["n"], float(np.sqrt(np.mean(res ** 2))))
        return res

    res = least_squares(residuals, x0, max_nfev=5000, verbose=verbose)

    progreso.detener(
        f"<<< FX0 OK — RMSE={float(np.sqrt(np.mean(res.fun ** 2))):.2f} N, "
        f"coste={res.cost:.4e}"
    )
    c = _set_coeffs(c, dict(zip(FX0_PARAMS, res.x)))
    logger.info("fit_fx0: RMSE=%.2f N, coste=%.4f, puntos=%d",
                float(np.sqrt(np.mean(res.fun ** 2))), res.cost, len(FX))
    return c


def fit_combined(
    df: pd.DataFrame,
    c: MF52Coeffs,
    verbose: int = 0,
) -> MF52Coeffs:
    """Ajusta coeficientes de slip combinado con datos Combined."""
    datos = _preparar_datos(df, "Combined", MAX_PUNTOS["combined"])
    if datos is None:
        logger.warning("fit_combined: sin datos Combined.")
        return c

    alpha, kappa, Fz, gamma, sub = datos
    FX = sub["FX"].values
    FY = sub["FY"].values

    x0 = _get_coeffs(c, COMBINED_PARAMS)

    print(f">>> Combined (slip mixto) — {len(FX)} puntos, {len(COMBINED_PARAMS)} coeficientes")
    progreso = _ProgresoFit("Combined")
    progreso.iniciar()

    contador = {"n": 0}

    def residuals(x):
        contador["n"] += 1
        c_t = _set_coeffs(c, dict(zip(COMBINED_PARAMS, x)))
        Fx, Fy, _ = combined_slip(alpha, kappa, Fz, gamma, c_t)
        res = np.concatenate([Fx - FX, Fy - FY])
        progreso.actualizar(contador["n"], float(np.sqrt(np.mean(res ** 2))))
        return res

    res = least_squares(residuals, x0, max_nfev=8000, verbose=verbose)

    progreso.detener(
        f"<<< Combined OK — RMSE={float(np.sqrt(np.mean(res.fun ** 2))):.2f} N, "
        f"coste={res.cost:.4e}"
    )
    c = _set_coeffs(c, dict(zip(COMBINED_PARAMS, res.x)))
    logger.info("fit_combined: RMSE=%.2f N, coste=%.4f, puntos=%d",
                float(np.sqrt(np.mean(res.fun ** 2))), res.cost, len(FX))
    return c


# -------------------------------------------------------------------------
# Orquestador
# -------------------------------------------------------------------------

def ajustar_coeficientes_mf52(
    df: pd.DataFrame,
    c_inicial: Optional[MF52Coeffs] = None,
    hacer_fy0: bool = True,
    hacer_mz0: bool = True,
    hacer_fx0: bool = True,
    hacer_combined: bool = True,
    verbose: int = 0,
) -> MF52Coeffs:
    """Ajuste secuencial de coeficientes MF5.2 con todos los datos."""
    c = c_inicial if c_inicial is not None else MF52Coeffs()

    if hacer_fy0:
        c = fit_fy0(df, c, verbose=verbose)
    if hacer_mz0:
        c = fit_mz0(df, c, verbose=verbose)
    if hacer_fx0:
        c = fit_fx0(df, c, verbose=verbose)
    if hacer_combined:
        c = fit_combined(df, c, verbose=verbose)

    return c


# -------------------------------------------------------------------------
# Escritura .tir
# -------------------------------------------------------------------------

def _seccion_tir(nombre: str, items: dict) -> str:
    lineas = [f"[{nombre}]"]
    for clave, valor in items.items():
        lineas.append(f"{clave} = {valor:.10f}")
    lineas.append("")
    return "\n".join(lineas)


def guardar_tir(c: MF52Coeffs, filepath: Path | str) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    d = c.as_dict()

    contenido = (
        "*=======================================================================\n"
        "*  Archivo .tir - Coeficientes Pacejka Magic Formula 5.2\n"
        "*  Generado por fit_ttc.py (repo tyres_26-27)\n"
        "*=======================================================================\n\n"
        "*  NOTA: Ajuste con todos los datos (todos los niveles de carga,\n"
        "*  camber y presion). La presion de referencia es 100 kPa (1 bar),\n"
        "*  valor tipico para neumaticos de competicion. La MF5.2 estandar\n"
        "*  no usa la presion directamente como variable de entrada.\n"
        "*=======================================================================\n\n"
        f"{_seccion_tir('TIRE', {'FZ0': d['Fz0'], 'R0': d['R0']})}"
        f"{_seccion_tir('LONGITUDINAL_PURO_FX0', {k.upper(): d[k] for k in FX0_PARAMS})}"
        f"{_seccion_tir('LATERAL_PURO_FY0', {k.upper(): d[k] for k in FY0_PARAMS})}"
        f"{_seccion_tir('ALINAMIENTO_PURO_MZ0', {k.upper(): d[k] for k in MZ0_PARAMS})}"
        f"{_seccion_tir('COMBINADO_SLIP', {k.upper(): d[k] for k in COMBINED_PARAMS})}"
        f"{_seccion_tir('FACTORES_ESCALA_LAMBDA', {k.upper(): d[k] for k in d if k.startswith('l')})}"
        "*=======================================================================\n"
        "*  Fin del archivo .tir\n"
        "*=======================================================================\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(contenido)

    logger.info("Archivo .tir guardado en: %s", filepath)


# -------------------------------------------------------------------------
# Punto de entrada
# -------------------------------------------------------------------------

def main(
    parquet_path: str | Path = "results/dataset_clasificado.parquet",
    tir_path: str | Path = "results/coeficientes_mf52.tir",
    verbose: int = 0,
    df_mf: Optional[pd.DataFrame] = None,
) -> MF52Coeffs:
    """Carga el dataset clasificado, ajusta coeficientes MF5.2 y guarda .tir.

    Ajusta una única Magic Formula con TODOS los datos disponibles
    (todos los niveles de carga vertical, camber y presión).
    La presión de referencia se asume en 1 bar (100 kPa), típica de
    neumáticos de competición; la MF5.2 estándar no usa la presión
    directamente como variable de entrada.

    Parameters
    ----------
    parquet_path : str | Path
        Ruta al archivo .parquet clasificado.
    tir_path : str | Path
        Ruta donde guardar el archivo .tir.
    verbose : int
        Nivel de verbosidad de scipy (0 = silencioso).
    df_mf : pd.DataFrame, optional
        DataFrame ya filtrado para ajuste MF. Si no se pasa, se lee del
        parquet y se filtra con filtrar_para_ajuste_mf().
    """
    parquet_path = Path(parquet_path)
    tir_path = Path(tir_path)

    # Cargar config de fitting y actualizar MAX_PUNTOS
    cfg_fitting = cargar_config_fitting()
    MAX_PUNTOS['fy0'] = cfg_fitting['max_puntos_fy0']
    MAX_PUNTOS['mz0'] = cfg_fitting['max_puntos_mz0']
    MAX_PUNTOS['fx0'] = cfg_fitting['max_puntos_fx0']
    MAX_PUNTOS['combined'] = cfg_fitting['max_puntos_combined']
    if verbose == 0:
        verbose = cfg_fitting['verbose']

    if df_mf is not None:
        df = df_mf
    else:
        if not parquet_path.exists():
            raise FileNotFoundError(f"No se encontró el parquet: {parquet_path}")
        logger.info("Cargando dataset: %s", parquet_path)
        df = pd.read_parquet(parquet_path)
        df = filtrar_para_ajuste_mf(df)

    logger.info("Tipos de ensayo:\n%s", df["test_type"].value_counts())

    c = ajustar_coeficientes_mf52(df, verbose=verbose)
    guardar_tir(c, tir_path)

    print("\n=== RESUMEN MF5.2 ===")
    print(f"Dataset : {parquet_path}")
    print(f"Salida  : {tir_path}")
    print(f"Puntos  : {len(df)}")
    print(f"Cornering: {(df['test_type'] == 'Pure Cornering').sum()}")
    print(f"Combined : {(df['test_type'] == 'Combined').sum()}")
    print(f"Longitud.: {(df['test_type'] == 'Pure Longitudinal').sum()}")
    print(f"Presion ref.: 100 kPa (1 bar, valor por defecto MF5.2)")
    print("====================\n")

    return c


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    main()
