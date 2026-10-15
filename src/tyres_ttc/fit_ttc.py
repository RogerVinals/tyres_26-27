# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

"""
fit_ttc.py
==========
Ajuste de coeficientes Pacejka Magic Formula 5.2 a datos TTC filtrados
mediante mínimos cuadrados no lineales (scipy.optimize.least_squares).

Genera un archivo .tir con todos los coeficientes MF5.2.

Mejoras respecto a la versión inicial:
  * Detección y eliminación de ensayos transitorios (cambios bruscos de FZ/SA).
  * Ajuste en dos pasadas con eliminación de outliers basada en MAD.
  * Función de pérdida robusta (soft_l1) para que los errores grandes no dominen.
  * Bounds físicos y x_scale='jac' para evitar mínimos locales y mejorar convergencia.
  * Salida .tir con formato IPG Carmaker completo (dimensiones fijas, coeficientes
    MF5.2 desde el objeto ajustado, valores por defecto para lo restante).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

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


# ============================================================================
# Umbrales y bounds físicos
# ============================================================================

# Filtro de estado estacionario: se eliminan puntos donde FZ o SA cambian
# bruscamente dentro de una ventana (transitorios de la máquina de ensayo).
WINDOW_ESTACIONARIO = 25          # puntos de ventana móvil
UMBRAL_STD_SA = 0.8               # deg, |SA| no debe variar más de esto en la ventana
UMBRAL_STD_FZ = 40.0              # N, |FZ| no debe variar más de esto en la ventana

# Umbral de outliers (múltiplos del MAD de los residuos)
OUTLIER_K_MAD = 3.0


# Bounds físicos aproximados para coeficientes MF5.2 de llanta de competición
# (llanta Hoosier 16.0X7.5-10 o similar). Si un parámetro no está listado,
# se deja sin bounds (-inf, inf).
FY0_BOUNDS = {
    "pCy1": (0.5, 2.0),       # forma
    "pDy1": (0.3, 3.0),       # fricción pico
    "pDy2": (-2.0, 0.5),      # sensibilidad a carga
    "pKy1": (-2000.0, -1.0),  # rigidez inicial (negativa por convención)
    "pKy2": (1.0, 500.0),     # argumento de sin
    "pKy3": (0.0, 2.0),       # factor camber
    "pEy1": (0.0, 3.0),       # curvatura
    "pEy2": (-2.0, 2.0),      # sensibilidad carga curvatura
    "pHy1": (-10.0, 10.0),    # desplazamiento horizontal
    "pHy2": (-2.0, 2.0),      # desplazamiento horizontal carga
    "pVy1": (-2.0, 2.0),      # desplazamiento vertical
    "pVy2": (-2.0, 2.0),      # desplazamiento vertical carga
    "pVy3": (-5.0, 5.0),      # desplazamiento vertical camber
    "pVy4": (-5.0, 5.0),      # desplazamiento vertical camber carga
}

FX0_BOUNDS = {
    "pCx1": (0.5, 2.0),
    "pDx1": (0.3, 3.0),
    "pDx2": (-2.0, 0.5),
    "pEx1": (0.0, 3.0),
    "pEx2": (-2.0, 2.0),
    "pKx1": (1.0, 2000.0),
    "pKx2": (-2.0, 2.0),
    "pHx1": (-5.0, 5.0),
    "pVx1": (-1.0, 1.0),
    "pVx2": (-2.0, 2.0),
}

MZ0_BOUNDS = {
    "qBz1": (-5.0, 5.0),
    "qBz2": (-5.0, 5.0),
    "qBz3": (-5.0, 5.0),
    "qBz4": (-5.0, 5.0),
    "qCz1": (0.5, 2.0),
    "qDz1": (-5.0, 5.0),
    "qDz2": (-2.0, 2.0),
    "qEz1": (0.0, 3.0),
    "qEz2": (-2.0, 2.0),
    "qHz1": (-5.0, 5.0),
    "qHz2": (-2.0, 2.0),
    "qDz6": (-5.0, 5.0),
    "qDz7": (-2.0, 2.0),
}

COMBINED_BOUNDS = {
    "rBx1": (0.1, 5.0),
    "rBx2": (0.1, 10.0),
    "rCx1": (0.1, 3.0),
    "rEx1": (-2.0, 2.0),
    "rEx2": (-2.0, 2.0),
    "rBy1": (0.1, 50.0),
    "rBy2": (0.1, 50.0),
    "rBy3": (-5.0, 5.0),
    "rCy1": (0.5, 2.0),
    "rEy1": (-2.0, 2.0),
    "rEy2": (-2.0, 2.0),
    "rHy1": (-5.0, 5.0),
    "rHy2": (-2.0, 2.0),
    "rVy1": (-2.0, 2.0),
    "rVy2": (-2.0, 2.0),
    "rVy3": (-2.0, 2.0),
    "rVy4": (0.1, 10.0),
    "rVy5": (0.1, 10.0),
    "rVy6": (0.1, 10.0),
}

BOUNDS_MAP = {
    "fy0": FY0_BOUNDS,
    "mz0": MZ0_BOUNDS,
    "fx0": FX0_BOUNDS,
    "combined": COMBINED_BOUNDS,
}


# ============================================================================
# Carga de configuración
# ============================================================================

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


# ============================================================================
# Progreso visual en tiempo real (spinner + contador)
# ============================================================================

class _ProgresoFit:
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
# Helpers de datos
# -------------------------------------------------------------------------

def _subsample(arrs: list, max_puntos: Optional[int]) -> list:
    """Muestreo uniforme si excede max_puntos. None = sin subsampling."""
    if max_puntos is None or len(arrs[0]) <= max_puntos:
        return arrs
    idx = np.linspace(0, len(arrs[0]) - 1, max_puntos).astype(int)
    resultado = []
    for a in arrs:
        if isinstance(a, pd.DataFrame):
            resultado.append(a.iloc[idx].reset_index(drop=True))
        else:
            resultado.append(np.asarray(a)[idx])
    return resultado


def _filtrar_estacionario(
    sub: pd.DataFrame,
    ventana: int = WINDOW_ESTACIONARIO,
    umbral_std_sa: float = UMBRAL_STD_SA,
    umbral_std_fz: float = UMBRAL_STD_FZ,
) -> pd.DataFrame:
    """
    Elimina puntos de transitorio (cambios bruscos de FZ o SA) usando
    desviación estándar móvil. Útil para quitar los escalones de carga
    y los cambios bruscos de ángulo que no son estado estacionario.
    """
    if len(sub) < ventana:
        return sub

    sub = sub.sort_values("ET").reset_index(drop=True) if "ET" in sub.columns else sub.reset_index(drop=True)

    sa_std = sub["SA"].rolling(window=ventana, center=True, min_periods=1).std()
    fz_std = sub["FZ"].abs().rolling(window=ventana, center=True, min_periods=1).std()

    mascara = (sa_std <= umbral_std_sa) & (fz_std <= umbral_std_fz)
    n_descartados = (~mascara).sum()
    if n_descartados > 0:
        logger.info(
            "[Estacionario] Descartados %d punto(s) de transitorio (%.1f%%)",
            n_descartados, 100 * n_descartados / len(sub),
        )
    return sub[mascara].reset_index(drop=True)


def _preparar_datos(
    df: pd.DataFrame,
    test_type: str,
    max_puntos: Optional[int] = None,
    estacionario: bool = True,
):
    """
    Filtra tipo de ensayo y devuelve arrays listos para Pacejka.
    Opcionalmente elimina puntos de transitorio (no estacionario).
    """
    sub = df[df["test_type"] == test_type].copy()
    if sub.empty:
        return None

    if estacionario:
        sub = _filtrar_estacionario(sub)

    if sub.empty:
        return None

    alpha = sub["SA"].values * DEG2RAD
    kappa = sub["SL_effective"].values
    Fz = (-sub["FZ"].values).clip(min=1.0)   # SAE FZ negativo → positivo
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


def _get_bounds(names: list, etapa: str):
    """Devuelve (lb, ub) arrays para los coeficientes dados, o (-inf, inf) si no hay bound."""
    bounds_dict = BOUNDS_MAP.get(etapa, {})
    lb = []
    ub = []
    for n in names:
        if n in bounds_dict:
            lb.append(bounds_dict[n][0])
            ub.append(bounds_dict[n][1])
        else:
            lb.append(-np.inf)
            ub.append(np.inf)
    return (np.array(lb), np.array(ub))


# -------------------------------------------------------------------------
# Ajuste robusto en dos pasadas
# -------------------------------------------------------------------------

def _ajustar_robusto(
    nombre: str,
    c: MF52Coeffs,
    x0: np.ndarray,
    params_names: list,
    etapa: str,
    residual_builder: Callable[[MF52Coeffs], np.ndarray],
    verbose: int = 0,
    max_nfev1: int = 2000,
    max_nfev2: int = 5000,
):
    """
    Ajuste en dos pasadas:
      1. Ajuste rápido con pérdida cuadrada para detectar outliers.
      2. Filtrado por MAD de residuos y ajuste final con pérdida soft_l1.

    Parameters
    ----------
    nombre : str
        Nombre de la etapa (para logs y progreso).
    c : MF52Coeffs
        Coeficientes actuales (se mutan con el resultado).
    x0 : np.ndarray
        Valores iniciales de los coeficientes a ajustar.
    params_names : list
        Nombres de los coeficientes a ajustar (en orden de x).
    etapa : str
        Clave para buscar bounds ('fy0', 'mz0', 'fx0', 'combined').
    residual_builder : callable
        Función que recibe un MF52Coeffs y devuelve el array de residuales.
    verbose : int
        Nivel de verbosidad de scipy (0 = silencioso).
    max_nfev1, max_nfev2 : int
        Máximo número de evaluaciones en pasada 1 y 2.
    """
    progreso = _ProgresoFit(nombre)
    progreso.iniciar()

    contador = {"n": 0}

    def residuals(x):
        contador["n"] += 1
        c_t = _set_coeffs(c, dict(zip(params_names, x)))
        res = residual_builder(c_t)
        progreso.actualizar(contador["n"], float(np.sqrt(np.mean(res ** 2))))
        return res

    # ---- Primera pasada: ajuste rápido para detectar outliers ----
    lb, ub = _get_bounds(params_names, etapa)
    res1 = least_squares(
        residuals, x0, bounds=(lb, ub),
        max_nfev=max_nfev1, verbose=0, x_scale='jac',
    )

    # Calcular MAD de residuos para detectar outliers
    res_vals = res1.fun
    med = np.median(res_vals)
    mad = np.median(np.abs(res_vals - med))
    if mad < 1e-8:
        mad = np.std(res_vals)

    umbral = OUTLIER_K_MAD * mad
    mascara = np.abs(res_vals - med) < umbral
    n_descartados = int((~mascara).sum())

    if n_descartados > 0:
        logger.info(
            "[%s] Descartados %d outlier(s) (%.1f%% de %d puntos)",
            nombre, n_descartados, 100 * n_descartados / len(res_vals), len(res_vals),
        )
    else:
        logger.info("[%s] No se detectaron outliers significativos", nombre)

    # ---- Segunda pasada: ajuste con datos filtrados y pérdida robusta ----
    if n_descartados > 0:
        def residuals_filtrados(x):
            contador["n"] += 1
            c_t = _set_coeffs(c, dict(zip(params_names, x)))
            res = residual_builder(c_t)
            return res[mascara]

        res2 = least_squares(
            residuals_filtrados, res1.x, bounds=(lb, ub),
            max_nfev=max_nfev2, verbose=verbose, loss='soft_l1', x_scale='jac',
        )
        res_final = res2
    else:
        res_final = res1

    progreso.detener(
        f"<<< {nombre} OK — RMSE={float(np.sqrt(np.mean(res_final.fun ** 2))):.2f}, "
        f"coste={res_final.cost:.4e}"
    )
    return res_final


# -------------------------------------------------------------------------
# Ajustes por régimen
# -------------------------------------------------------------------------

def fit_fy0(
    df: pd.DataFrame,
    c: MF52Coeffs,
    verbose: int = 0,
) -> MF52Coeffs:
    """Ajusta coeficientes de fuerza lateral pura (Fy0) con datos Pure Cornering."""
    datos = _preparar_datos(df, "Pure Cornering", MAX_PUNTOS["fy0"], estacionario=True)
    if datos is None:
        logger.warning("fit_fy0: sin datos Pure Cornering.")
        return c

    alpha, _, Fz, gamma, sub = datos
    FY = sub["FY"].values

    x0 = _get_coeffs(c, FY0_PARAMS)

    print(f">>> FY0 (Pure Cornering) — {len(FY)} puntos, {len(FY0_PARAMS)} coeficientes")

    def residual_builder(c_t: MF52Coeffs) -> np.ndarray:
        Fy0, _ = pure_slip_fy(alpha, Fz, gamma, c_t)
        return Fy0 - FY

    res = _ajustar_robusto(
        "FY0", c, x0, FY0_PARAMS, "fy0", residual_builder, verbose=verbose,
    )

    c = _set_coeffs(c, dict(zip(FY0_PARAMS, res.x)))
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    logger.info("fit_fy0: RMSE=%.2f N, coste=%.4f, puntos=%d",
                rmse, res.cost, len(FY))
    return c


def fit_mz0(
    df: pd.DataFrame,
    c: MF52Coeffs,
    verbose: int = 0,
) -> MF52Coeffs:
    """Ajusta coeficientes de momento autoalignante puro (Mz0) con datos Pure Cornering."""
    datos = _preparar_datos(df, "Pure Cornering", MAX_PUNTOS["mz0"], estacionario=True)
    if datos is None:
        logger.warning("fit_mz0: sin datos Pure Cornering.")
        return c

    alpha, _, Fz, gamma, sub = datos
    MZ = sub["MZ"].values

    x0 = _get_coeffs(c, MZ0_PARAMS)

    print(f">>> MZ0 (Pure Cornering) — {len(MZ)} puntos, {len(MZ0_PARAMS)} coeficientes")

    def residual_builder(c_t: MF52Coeffs) -> np.ndarray:
        Mz0, _ = pure_slip_mz(alpha, Fz, gamma, c_t)
        return Mz0 - MZ

    res = _ajustar_robusto(
        "MZ0", c, x0, MZ0_PARAMS, "mz0", residual_builder, verbose=verbose,
    )

    c = _set_coeffs(c, dict(zip(MZ0_PARAMS, res.x)))
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    logger.info("fit_mz0: RMSE=%.2f N*m, coste=%.4f, puntos=%d",
                rmse, res.cost, len(MZ))
    return c


def fit_fx0(
    df: pd.DataFrame,
    c: MF52Coeffs,
    verbose: int = 0,
) -> MF52Coeffs:
    """Ajusta coeficientes de fuerza longitudinal pura (Fx0) con datos Pure Longitudinal."""
    datos = _preparar_datos(df, "Pure Longitudinal", MAX_PUNTOS["fx0"], estacionario=True)
    if datos is None:
        logger.warning("fit_fx0: sin datos Pure Longitudinal (usa valores por defecto).")
        return c

    _, kappa, Fz, gamma, sub = datos
    FX = sub["FX"].values

    x0 = _get_coeffs(c, FX0_PARAMS)

    print(f">>> FX0 (Pure Longitudinal) — {len(FX)} puntos, {len(FX0_PARAMS)} coeficientes")

    def residual_builder(c_t: MF52Coeffs) -> np.ndarray:
        Fx0, _ = pure_slip_fx(kappa, Fz, gamma, c_t)
        return Fx0 - FX

    res = _ajustar_robusto(
        "FX0", c, x0, FX0_PARAMS, "fx0", residual_builder, verbose=verbose,
    )

    c = _set_coeffs(c, dict(zip(FX0_PARAMS, res.x)))
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    logger.info("fit_fx0: RMSE=%.2f N, coste=%.4f, puntos=%d",
                rmse, res.cost, len(FX))
    return c


def fit_combined(
    df: pd.DataFrame,
    c: MF52Coeffs,
    verbose: int = 0,
) -> MF52Coeffs:
    """Ajusta coeficientes de slip combinado con datos Combined."""
    datos = _preparar_datos(df, "Combined", MAX_PUNTOS["combined"], estacionario=True)
    if datos is None:
        logger.warning("fit_combined: sin datos Combined.")
        return c

    alpha, kappa, Fz, gamma, sub = datos
    FX = sub["FX"].values
    FY = sub["FY"].values

    x0 = _get_coeffs(c, COMBINED_PARAMS)

    print(f">>> Combined (slip mixto) — {len(FX)} puntos, {len(COMBINED_PARAMS)} coeficientes")

    def residual_builder(c_t: MF52Coeffs) -> np.ndarray:
        Fx, Fy, _ = combined_slip(alpha, kappa, Fz, gamma, c_t)
        return np.concatenate([Fx - FX, Fy - FY])

    res = _ajustar_robusto(
        "Combined", c, x0, COMBINED_PARAMS, "combined", residual_builder, verbose=verbose,
    )

    c = _set_coeffs(c, dict(zip(COMBINED_PARAMS, res.x)))
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    logger.info("fit_combined: RMSE=%.2f N, coste=%.4f, puntos=%d",
                rmse, res.cost, len(FX))
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


# -------------------------------------------------------------------------
# Valores por defecto para secciones NO-MF5.2 (dimensiones, presiones, etc.)
# Estos NO provienen de MF52Coeffs, son valores fijos del neumático.
# -------------------------------------------------------------------------

DIMENSION_VALUES = {
    "UNLOADED_RADIUS": 0.2032,
    "WIDTH": 0.1905,
    "ASPECT_RATIO": 0.4,
    "RIM_RADIUS": 0.127,
    "RIM_WIDTH": 0.1778,
}

OPERATING_CONDITIONS_VALUES = {
    "INFPRES": 82737.0,
    "NOMPRES": 82737.0,
}

VERTICAL_VALUES = {
    "VERTICAL_STIFFNES": 132500.0,
    "VERTICAL_DAMPING": 50.0,
    "BREFF": 8.0,
    "DREFF": 0.24,
    "FREFF": 0.01,
    "FNOMIN": 1000.0,
}

VERTICAL_FORCE_RANGE_VALUES = {
    "FZMIN": 0.0,
    "FZMAX": 1800.0,
}

LONG_SLIP_RANGE_VALUES = {
    "KPUMIN": -1.0,
    "KPUMAX": 1.0,
}

SLIP_ANGLE_RANGE_VALUES = {
    "ALPMIN": -0.96,
    "ALPMAX": 0.96,
}

INCLINATION_ANGLE_RANGE_VALUES = {
    "CAMMIN": -0.105,
    "CAMMAX": 0.105,
}

# Coeficientes MF5.2 que NO están en las listas de parámetros pero que
# se pueden ajustar o tener valores por defecto razonables.
# Se usa el valor de MF52Coeffs si existe, si no el valor por defecto.
MF52_DEFAULT_VALUES = {
    # [TIRE]
    "FZ0": 4500.0,
    "R0": 0.3,
    # Longitudinal coeficientes no en FX0_PARAMS
    "PDX3": 19.26,
    "PPX1": -0.7053,
    "PPX2": -0.5895,
    "PPX3": -0.2428,
    "PPX4": 0.2052,
    "PVX3": 0.0,
    "PVX4": 0.0,
    "RBX3": 0.0,
    "RHX2": 0.0,
    # Lateral coeficientes no en FY0_PARAMS
    "PDY3": 9.70,
    "PEY5": 0.0,
    "PKY4": 2.0,
    "PKY5": 0.0,
    "PKY6": 0.0,
    "PKY7": 0.0,
    "PHY3": 0.0,
    "PPY5": 0.0,
    "RBY4": 0.0,
    # Aligning coeficientes no en MZ0_PARAMS
    "QBZ5": 4.954,
    "QBZ9": 16.725,
    "QBZ10": 0.809,
    "QDZ5": -19.378,
    "QDZ8": -2.229,
    "QDZ9": 0.61,
    "QDZ10": 0.0,
    "QDZ11": 0.0,
    "QHZ3": -0.332,
    "QHZ4": -0.332,
    "PPZ1": 0.0,
    "PPZ2": 0.0,
    # Factores de escala lambda no en MF52Coeffs por defecto
    "LGAMMAX": 1.0,
    "LGAMMAY": 1.0,
    "LGAMMAZ": 1.0,
    "LT": 1.0,
    "LR": 1.0,
    "LXAL": 1.0,
    "LYKA": 1.0,
    "LVYKA": 1.0,
    "LS": 1.0,
    "LKYC": 1.07,
    "LKZC": 1.0,
    "LVMX": 1.0,
    "LMX": 1.0,
    "LMY": 1.0,
    "LMP": 1.0,
    # Overturning
    "QSX1": -0.0227,
    "QSX2": 2.1630,
    "QSX3": 0.3030,
    "QSX4": 0.1625,
    "QSX5": 0.0,
    "QSX6": 0.0,
    "QSX7": 0.0701,
    "QSX8": 0.0,
    "QSX9": 0.0,
    "QSX10": 0.0,
    "QSX11": 0.0,
    "QSX12": 0.0,
    "QSX13": 0.0,
    "QSX14": 0.0,
    "PPMX1": 0.4034,
    # Rolling
    "QSY1": 0.01,
    "QSY2": 0.0,
    "QSY3": 0.0004,
    "QSY4": 4e-05,
    "QSY5": 0.0,
    "QSY6": 0.0,
    "QSY7": 0.0,
    "QSY8": 0.0,
    # Turnslip - todos cero
    "PDXP1": 0.0,
    "PDXP2": 0.0,
    "PDXP3": 0.0,
    "PKYP1": 0.0,
    "PDYP1": 0.0,
    "PDYP2": 0.0,
    "PDYP3": 0.0,
    "PDYP4": 0.0,
    "PHYP1": 0.0,
    "PHYP2": 0.0,
    "PHYP3": 0.0,
    "PHYP4": 0.0,
    "PECP1": 0.0,
    "PECP2": 0.0,
    "QDTP1": 0.0,
    "QCRP1": 0.0,
    "QCRP2": 0.0,
    "QBRP1": 0.0,
    "QDRP1": 0.0,
}


def _obtener_valor_mf52(d: dict, clave: str, default: float = 0.0) -> float:
    """Devuelve el valor de un coeficiente MF5.2 del diccionario d, o el valor por defecto."""
    if clave in d:
        return float(d[clave])
    return float(MF52_DEFAULT_VALUES.get(clave, default))


def _obtener_valor_fijo(valores: dict, clave: str) -> float:
    """Devuelve un valor fijo de dimensión/parámetro (NO proviene de MF52Coeffs)."""
    return float(valores.get(clave, 0.0))


def guardar_tir(c: MF52Coeffs, filepath: Path | str) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    d = c.as_dict()

    # Dimensiones y parámetros no-MF5.2: SIEMPRE valores fijos, nunca de d
    unloaded_radius = _obtener_valor_fijo(DIMENSION_VALUES, "UNLOADED_RADIUS")
    width = _obtener_valor_fijo(DIMENSION_VALUES, "WIDTH")
    aspect_ratio = _obtener_valor_fijo(DIMENSION_VALUES, "ASPECT_RATIO")
    rim_radius = _obtener_valor_fijo(DIMENSION_VALUES, "RIM_RADIUS")
    rim_width = _obtener_valor_fijo(DIMENSION_VALUES, "RIM_WIDTH")

    infpres = _obtener_valor_fijo(OPERATING_CONDITIONS_VALUES, "INFPRES")
    nompres = _obtener_valor_fijo(OPERATING_CONDITIONS_VALUES, "NOMPRES")

    vertical_stiffnes = _obtener_valor_fijo(VERTICAL_VALUES, "VERTICAL_STIFFNES")
    vertical_damping = _obtener_valor_fijo(VERTICAL_VALUES, "VERTICAL_DAMPING")
    breff = _obtener_valor_fijo(VERTICAL_VALUES, "BREFF")
    dreff = _obtener_valor_fijo(VERTICAL_VALUES, "DREFF")
    freff = _obtener_valor_fijo(VERTICAL_VALUES, "FREFF")
    fnom = _obtener_valor_fijo(VERTICAL_VALUES, "FNOMIN")

    fzmin = _obtener_valor_fijo(VERTICAL_FORCE_RANGE_VALUES, "FZMIN")
    fzmax = _obtener_valor_fijo(VERTICAL_FORCE_RANGE_VALUES, "FZMAX")

    kpumin = _obtener_valor_fijo(LONG_SLIP_RANGE_VALUES, "KPUMIN")
    kpumax = _obtener_valor_fijo(LONG_SLIP_RANGE_VALUES, "KPUMAX")

    alpmin = _obtener_valor_fijo(SLIP_ANGLE_RANGE_VALUES, "ALPMIN")
    alpmax = _obtener_valor_fijo(SLIP_ANGLE_RANGE_VALUES, "ALPMAX")

    cammin = _obtener_valor_fijo(INCLINATION_ANGLE_RANGE_VALUES, "CAMMIN")
    cammax = _obtener_valor_fijo(INCLINATION_ANGLE_RANGE_VALUES, "CAMMAX")

    contenido = (
        "*=======================================================================\n"
        "*  Archivo .tir - Coeficientes Pacejka Magic Formula 5.2\n"
        "*  Generado por fit_ttc.py (repo tyres_26-27)\n"
        "*=======================================================================\n\n"
        "[MDI_HEADER]\n"
        "FILE_TYPE                    = 'tir'             \n"
        "FILE_VERSION                 = 3                 \n"
        "FILE_FORMAT                  = 'ASCII'           \n"
        "$-----------------------------------------------------------------units\n"
        "[UNITS]\n"
        "LENGTH                       = 'meter'           \n"
        "FORCE                        = 'newton'          \n"
        "ANGLE                        = 'radians'         \n"
        "MASS                         = 'kg'              \n"
        "TIME                         = 'second'          \n"
        "$-----------------------------------------------------------------model\n"
        "[MODEL]\n"
        "PROPERTY_FILE_FORMAT         = 'MF_05'           \n"
        "USE_MODE                     = 14                \n"
        "TYRESIDE                     = 'LEFT'            \n"
        "LONGVL                       = 11.176            \n"
        "VXLOW                        = 1                 \n"
        "$-------------------------------------------------------------dimension\n"
        "[DIMENSION]\n"
        f"UNLOADED_RADIUS              = {unloaded_radius:.6f}   \n"
        f"WIDTH                        = {width:.6f}   \n"
        f"ASPECT_RATIO                 = {aspect_ratio:.6f}      \n"
        f"RIM_RADIUS                   = {rim_radius:.6f}    \n"
        f"RIM_WIDTH                    = {rim_width:.6f}   \n"
        "$--------------------------------------------------operating conditions\n"
        "[OPERATING_CONDITIONS]\n"
        f"INFPRES                      = {infpres:.4f}    \n"
        f"NOMPRES                      = {nompres:.4f}    \n"
        "$---------------------------------------------------------------inertia\n"
        "[INERTIA]\n"
        "$--------------------------------------------------------------vertical\n"
        "[VERTICAL]\n"
        f"VERTICAL_STIFFNES            = {vertical_stiffnes:.1f}   \n"
        f"VERTICAL_DAMPING             = {vertical_damping:.1f}   \n"
        f"BREFF                        = {breff:.1f}   \n"
        f"DREFF                        = {dreff:.2f}   \n"
        f"FREFF                        = {freff:.2f}   \n"
        f"FNOMIN                       = {fnom:.0f}   \n"
        "$--------------------------------------------------structural\n"
        "[STRUCTURAL]\n"
        "$---------------------------------------------------------contact patch\n"
        "[CONTACT_PATCH]\n"
        "$----------------------------------------------inflation pressure range\n"
        "[INFLATION_PRESSURE_RANGE]\n"
        "$--------------------------------------------------vertical force range\n"
        "[VERTICAL_FORCE_RANGE]\n"
        f"FZMIN                        = {fzmin:.1f}   \n"
        f"FZMAX                        = {fzmax:.1f}   \n"
        "$-------------------------------------------------------long slip range\n"
        "[LONG_SLIP_RANGE]\n"
        f"KPUMIN                       = {kpumin:.1f}   \n"
        f"KPUMAX                       = {kpumax:.1f}   \n"
        "$------------------------------------------------------slip angle range\n"
        "[SLIP_ANGLE_RANGE]\n"
        f"ALPMIN                       = {alpmin:.2f}   \n"
        f"ALPMAX                       = {alpmax:.2f}   \n"
        "$-----------------------------------------------inclination angle range\n"
        "[INCLINATION_ANGLE_RANGE]\n"
        f"CAMMIN                       = {cammin:.3f}   \n"
        f"CAMMAX                       = {cammax:.3f}   \n"
        "$--------------------------------------------------scaling coefficients\n"
        "[SCALING_COEFFICIENTS]\n"
        f"LFZ0                         = {_obtener_valor_mf52(d, 'lFz0', 1.0):.3f}                 \n"
        f"LCX                          = {_obtener_valor_mf52(d, 'lCx', 1.0):.3f}                  \n"
        f"LMUX                         = {_obtener_valor_mf52(d, 'lMUx', 1.0):.3f}                 \n"
        f"LEX                          = {_obtener_valor_mf52(d, 'lEx', 1.0):.3f}                 \n"
        f"LKX                          = {_obtener_valor_mf52(d, 'lKx', 1.0):.3f}                  \n"
        f"LHX                          = {_obtener_valor_mf52(d, 'lHx', 1.0):.3f}                  \n"
        f"LVX                          = {_obtener_valor_mf52(d, 'lVx', 1.0):.3f}                 \n"
        f"LGAMMAX                      = {_obtener_valor_mf52(d, 'lGammax', 1.0):.3f}                 \n"
        f"LCY                          = {_obtener_valor_mf52(d, 'lCy', 1.0):.3f}                 \n"
        f"LMUY                         = {_obtener_valor_mf52(d, 'lMUy', 1.0):.3f}                  \n"
        f"LEY                          = {_obtener_valor_mf52(d, 'lEy', 1.0):.3f}                 \n"
        f"LKY                          = {_obtener_valor_mf52(d, 'lKy', 1.0):.3f}                  \n"
        f"LHY                          = {_obtener_valor_mf52(d, 'lHy', 1.0):.3f}                  \n"
        f"LVY                          = {_obtener_valor_mf52(d, 'lVy', 1.0):.3f}                 \n"
        f"LGAMMAY                      = {_obtener_valor_mf52(d, 'lGammay', 1.0):.3f}                 \n"
        f"LGAMMAZ                      = {_obtener_valor_mf52(d, 'lGammaz', 1.0):.3f}                 \n"
        f"LT                           = {_obtener_valor_mf52(d, 'lt', 1.0):.3f}                  \n"
        f"LR                           = {_obtener_valor_mf52(d, 'lr', 1.0):.3f}                  \n"
        f"LXAL                         = {_obtener_valor_mf52(d, 'lXal', 1.0):.3f}                  \n"
        f"LYKA                         = {_obtener_valor_mf52(d, 'lYka', 1.0):.3f}                  \n"
        f"LVYKA                        = {_obtener_valor_mf52(d, 'lVyka', 1.0):.3f}                 \n"
        f"LS                           = {_obtener_valor_mf52(d, 'ls', 1.0):.3f}                \n"
        f"LKYC                         = {_obtener_valor_mf52(d, 'lKYc', 1.07):.3f}                 \n"
        f"LKZC                         = {_obtener_valor_mf52(d, 'lKZc', 1.0):.3f}                 \n"
        f"LVMX                         = {_obtener_valor_mf52(d, 'lVMx', 1.0):.3f}                  \n"
        f"LMX                          = {_obtener_valor_mf52(d, 'lMX', 1.0):.3f}                  \n"
        f"LMY                          = {_obtener_valor_mf52(d, 'lMY', 1.0):.3f}                  \n"
        f"LMP                          = {_obtener_valor_mf52(d, 'lMP', 1.0):.3f}                 \n"
        "$---------------------------------------------longitudinal coefficients\n"
        "[LONGITUDINAL_COEFFICIENTS]\n"
        f"PCX1                         = {_obtener_valor_mf52(d, 'pCx1', 1.0):.10f}                \n"
        f"PDX1                         = {_obtener_valor_mf52(d, 'pDx1', 1.0):.10f}                \n"
        f"PDX2                         = {_obtener_valor_mf52(d, 'pDx2', 0.0):.10f}                \n"
        f"PDX3                         = {_obtener_valor_mf52(d, 'pDx3', 19.26):.10f}              \n"
        f"PEX1                         = {_obtener_valor_mf52(d, 'pEx1', 0.0):.10f}                 \n"
        f"PEX2                         = {_obtener_valor_mf52(d, 'pEx2', 0.0):.10f}                \n"
        f"PEX3                         = {_obtener_valor_mf52(d, 'pEx3', 0.0):.10f}                   \n"
        f"PEX4                         = {_obtener_valor_mf52(d, 'pEx4', 0.0):.10f}                 \n"
        f"PKX1                         = {_obtener_valor_mf52(d, 'pKx1', 30.0):.10f}                \n"
        f"PKX2                         = {_obtener_valor_mf52(d, 'pKx2', 0.0):.10f}                 \n"
        f"PKX3                         = {_obtener_valor_mf52(d, 'pKx3', 0.0):.10f}                 \n"
        f"PHX1                         = {_obtener_valor_mf52(d, 'pHx1', 0.0):.10f}                   \n"
        f"PHX2                         = {_obtener_valor_mf52(d, 'pHx2', 0.0):.10f}                   \n"
        f"PVX1                         = {_obtener_valor_mf52(d, 'pVx1', 0.0):.10f}                   \n"
        f"PVX2                         = {_obtener_valor_mf52(d, 'pVx2', 0.0):.10f}                  \n"
        f"PVX3                         = {_obtener_valor_mf52(d, 'pVx3', 0.0):.10f}                   \n"
        f"PVX4                         = {_obtener_valor_mf52(d, 'pVx4', 0.0):.10f}                  \n"
        f"PPX1                         = {_obtener_valor_mf52(d, 'ppCx1', -0.7053):.10f}                 \n"
        f"PPX2                         = {_obtener_valor_mf52(d, 'ppCx2', -0.5895):.10f}                 \n"
        f"PPX3                         = {_obtener_valor_mf52(d, 'ppCx3', -0.2428):.10f}                 \n"
        f"PPX4                         = {_obtener_valor_mf52(d, 'ppCx4', 0.2052):.10f}                 \n"
        f"RBX1                         = {_obtener_valor_mf52(d, 'rBx1', 1.0):.10f}                  \n"
        f"RBX2                         = {_obtener_valor_mf52(d, 'rBx2', 1.0):.10f}                  \n"
        f"RBX3                         = {_obtener_valor_mf52(d, 'rBx3', 0.0):.10f}                  \n"
        f"RCX1                         = {_obtener_valor_mf52(d, 'rCx1', 1.0):.10f}                  \n"
        f"REX1                         = {_obtener_valor_mf52(d, 'rEx1', 0.0):.10f}                 \n"
        f"REX2                         = {_obtener_valor_mf52(d, 'rEx2', 0.0):.10f}                \n"
        f"RHX1                         = {_obtener_valor_mf52(d, 'rHx1', 0.0):.10f}                  \n"
        f"RHX2                         = {_obtener_valor_mf52(d, 'rHx2', 0.0):.10f}                  \n"
        "$----------------------------------------------overturning coefficients\n"
        "[OVERTURNING_COEFFICIENTS]\n"
        f"QSX1                         = {_obtener_valor_mf52(d, 'qSX1', -0.0227):.4f}                \n"
        f"QSX2                         = {_obtener_valor_mf52(d, 'qSX2', 2.1630):.4f}                \n"
        f"QSX3                         = {_obtener_valor_mf52(d, 'qSX3', 0.3030):.4f}                \n"
        f"QSX4                         = {_obtener_valor_mf52(d, 'qSX4', 0.1625):.4f}                \n"
        f"QSX5                         = {_obtener_valor_mf52(d, 'qSX5', 0.0):.4f}                  \n"
        f"QSX6                         = {_obtener_valor_mf52(d, 'qSX6', 0.0):.4f}                  \n"
        f"QSX7                         = {_obtener_valor_mf52(d, 'qSX7', 0.0701):.4f}                \n"
        f"QSX8                         = {_obtener_valor_mf52(d, 'qSX8', 0.0):.4f}                  \n"
        f"QSX9                         = {_obtener_valor_mf52(d, 'qSX9', 0.0):.4f}                 \n"
        f"QSX10                        = {_obtener_valor_mf52(d, 'qSX10', 0.0):.4f}                 \n"
        f"QSX11                        = {_obtener_valor_mf52(d, 'qSX11', 0.0):.4f}                 \n"
        f"QSX12                        = {_obtener_valor_mf52(d, 'qSX12', 0.0):.4f}                 \n"
        f"QSX13                        = {_obtener_valor_mf52(d, 'qSX13', 0.0):.4f}                 \n"
        f"QSX14                        = {_obtener_valor_mf52(d, 'qSX14', 0.0):.4f}                 \n"
        f"PPMX1                        = {_obtener_valor_mf52(d, 'qPPMX1', 0.4034):.4f}                         \n"
        "$--------------------------------------------------lateral coefficients\n"
        "[LATERAL_COEFFICIENTS]\n"
        f"PCY1                         = {_obtener_valor_mf52(d, 'pCy1', 1.0):.10f}                  \n"
        f"PDY1                         = {_obtener_valor_mf52(d, 'pDy1', 1.0):.10f}                \n"
        f"PDY2                         = {_obtener_valor_mf52(d, 'pDy2', 0.0):.10f}              \n"
        f"PDY3                         = {_obtener_valor_mf52(d, 'pDy3', 9.70):.10f}                   \n"
        f"PEY1                         = {_obtener_valor_mf52(d, 'pEy1', 0.0):.10f}               \n"
        f"PEY2                         = {_obtener_valor_mf52(d, 'pEy2', 0.0):.10f}               \n"
        f"PEY3                         = {_obtener_valor_mf52(d, 'pEy3', 0.0):.10f}                \n"
        f"PEY4                         = {_obtener_valor_mf52(d, 'pEy4', 0.0):.10f}                \n"
        f"PEY5                         = {_obtener_valor_mf52(d, 'pEy5', 0.0):.10f}                  \n"
        f"PKY1                         = {_obtener_valor_mf52(d, 'pKy1', -25.0):.10f}                \n"
        f"PKY2                         = {_obtener_valor_mf52(d, 'pKy2', 2.71):.10f}                  \n"
        f"PKY3                         = {_obtener_valor_mf52(d, 'pKy3', 0.39):.10f}                  \n"
        f"PKY4                         = {_obtener_valor_mf52(d, 'pKy4', 2.0):.10f}                  \n"
        f"PKY5                         = {_obtener_valor_mf52(d, 'pKy5', 0.0):.10f}                  \n"
        f"PKY6                         = {_obtener_valor_mf52(d, 'pKy6', 0.0):.10f}                  \n"
        f"PKY7                         = {_obtener_valor_mf52(d, 'pKy7', 0.0):.10f}                  \n"
        f"PHY1                         = {_obtener_valor_mf52(d, 'pHy1', 0.0):.10f}                   \n"
        f"PHY2                         = {_obtener_valor_mf52(d, 'pHy2', 0.0):.10f}                  \n"
        f"PHY3                         = {_obtener_valor_mf52(d, 'pHy3', 0.0):.10f}                   \n"
        f"PVY1                         = {_obtener_valor_mf52(d, 'pVy1', 0.0):.10f}                   \n"
        f"PVY2                         = {_obtener_valor_mf52(d, 'pVy2', 0.0):.10f}                  \n"
        f"PVY3                         = {_obtener_valor_mf52(d, 'pVy3', 0.0):.10f}                  \n"
        f"PVY4                         = {_obtener_valor_mf52(d, 'pVy4', 0.0):.10f}                  \n"
        f"PPY1                         = {_obtener_valor_mf52(d, 'ppCy1', 0.0):.10f}               \n"
        f"PPY2                         = {_obtener_valor_mf52(d, 'ppCy2', 0.0):.10f}                 \n"
        f"PPY3                         = {_obtener_valor_mf52(d, 'ppCy3', -0.2480):.10f}                 \n"
        f"PPY4                         = {_obtener_valor_mf52(d, 'ppCy4', 0.3440):.10f}                   \n"
        f"PPY5                         = {_obtener_valor_mf52(d, 'ppCy5', 0.0):.10f}                  \n"
        f"RBY1                         = {_obtener_valor_mf52(d, 'rBy1', 1.0):.10f}                  \n"
        f"RBY2                         = {_obtener_valor_mf52(d, 'rBy2', 1.0):.10f}                  \n"
        f"RBY3                         = {_obtener_valor_mf52(d, 'rBy3', 0.0):.10f}               \n"
        f"RBY4                         = {_obtener_valor_mf52(d, 'rBy4', 0.0):.10f}                  \n"
        f"RCY1                         = {_obtener_valor_mf52(d, 'rCy1', 1.0):.10f}                  \n"
        f"REY1                         = {_obtener_valor_mf52(d, 'rEy1', 0.0):.10f}               \n"
        f"REY2                         = {_obtener_valor_mf52(d, 'rEy2', 0.0):.10f}                \n"
        f"RHY1                         = {_obtener_valor_mf52(d, 'rHy1', 0.0):.10f}                  \n"
        f"RHY2                         = {_obtener_valor_mf52(d, 'rHy2', 0.0):.10f}                  \n"
        f"RVY1                         = {_obtener_valor_mf52(d, 'rVy1', 1.0):.10f}                  \n"
        f"RVY2                         = {_obtener_valor_mf52(d, 'rVy2', 1.0):.10f}                  \n"
        f"RVY3                         = {_obtener_valor_mf52(d, 'rVy3', 1.0):.10f}                   \n"
        f"RVY4                         = {_obtener_valor_mf52(d, 'rVy4', 1.0):.10f}                   \n"
        f"RVY5                         = {_obtener_valor_mf52(d, 'rVy5', 1.0):.10f}                  \n"
        f"RVY6                         = {_obtener_valor_mf52(d, 'rVy6', 1.0):.10f}                  \n"
        "$--------------------------------------------------rolling coefficients\n"
        "[ROLLING_COEFFICIENTS]\n"
        f"QSY1                         = {_obtener_valor_mf52(d, 'qSY1', 0.01):.10f}                 \n"
        f"QSY2                         = {_obtener_valor_mf52(d, 'qSY2', 0.0):.10f}                  \n"
        f"QSY3                         = {_obtener_valor_mf52(d, 'qSY3', 0.0004):.10f}                 \n"
        f"QSY4                         = {_obtener_valor_mf52(d, 'qSY4', 4e-05):.10f}                  \n"
        f"QSY5                         = {_obtener_valor_mf52(d, 'qSY5', 0.0):.10f}                  \n"
        f"QSY6                         = {_obtener_valor_mf52(d, 'qSY6', 0.0):.10f}                  \n"
        f"QSY7                         = {_obtener_valor_mf52(d, 'qSY7', 0.0):.10f}                  \n"
        f"QSY8                         = {_obtener_valor_mf52(d, 'qSY8', 0.0):.10f}                  \n"
        "$-------------------------------------------------aligning coefficients\n"
        "[ALIGNING_COEFFICIENTS]\n"
        f"QBZ1                         = {_obtener_valor_mf52(d, 'qBz1', 0.0):.10f}         \n"
        f"QBZ2                         = {_obtener_valor_mf52(d, 'qBz2', 0.0):.10f}                   \n"
        f"QBZ3                         = {_obtener_valor_mf52(d, 'qBz3', 0.0):.10f}                  \n"
        f"QBZ4                         = {_obtener_valor_mf52(d, 'qBz4', 0.0):.10f}                  \n"
        f"QBZ5                         = {_obtener_valor_mf52(d, 'qBz5', 4.954):.10f}                  \n"
        f"QBZ9                         = {_obtener_valor_mf52(d, 'qBz9', 16.725):.10f}                  \n"
        f"QBZ10                        = {_obtener_valor_mf52(d, 'qBz10', 0.809):.10f}                 \n"
        f"QCZ1                         = {_obtener_valor_mf52(d, 'qCz1', 1.0):.10f}                \n"
        f"QDZ1                         = {_obtener_valor_mf52(d, 'qDz1', 0.0):.10f}                \n"
        f"QDZ2                         = {_obtener_valor_mf52(d, 'qDz2', 0.0):.10f}                   \n"
        f"QDZ3                         = {_obtener_valor_mf52(d, 'qDz3', 0.0):.10f}                  \n"
        f"QDZ4                         = {_obtener_valor_mf52(d, 'qDz4', 0.0):.10f}                  \n"
        f"QDZ5                         = {_obtener_valor_mf52(d, 'qDz5', -19.378):.10f}                  \n"
        f"QDZ6                         = {_obtener_valor_mf52(d, 'qDz6', 0.0):.10f}                  \n"
        f"QDZ7                         = {_obtener_valor_mf52(d, 'qDz7', 0.0):.10f}                  \n"
        f"QDZ8                         = {_obtener_valor_mf52(d, 'qDz8', -2.229):.10f}                  \n"
        f"QDZ9                         = {_obtener_valor_mf52(d, 'qDz9', 0.61):.10f}                  \n"
        f"QDZ10                        = {_obtener_valor_mf52(d, 'qDz10', 0.0):.10f}                 \n"
        f"QDZ11                        = {_obtener_valor_mf52(d, 'qDz11', 0.0):.10f}                 \n"
        f"QEZ1                         = {_obtener_valor_mf52(d, 'qEz1', 0.0):.10f}                 \n"
        f"QEZ2                         = {_obtener_valor_mf52(d, 'qEz2', 0.0):.10f}                 \n"
        f"QEZ3                         = {_obtener_valor_mf52(d, 'qEz3', 0.0):.10f}                 \n"
        f"QEZ4                         = {_obtener_valor_mf52(d, 'qEz4', 0.0):.10f}                 \n"
        f"QEZ5                         = {_obtener_valor_mf52(d, 'qEz5', 0.0):.10f}                 \n"
        f"QHZ1                         = {_obtener_valor_mf52(d, 'qHz1', 0.0):.10f}                 \n"
        f"QHZ2                         = {_obtener_valor_mf52(d, 'qHz2', 0.0):.10f}                 \n"
        f"QHZ3                         = {_obtener_valor_mf52(d, 'qHz3', 0.0):.10f}                 \n"
        f"QHZ4                         = {_obtener_valor_mf52(d, 'qHz4', 0.0):.10f}                 \n"
        f"PPZ1                         = {_obtener_valor_mf52(d, 'qPPZ1', 0.0):.10f}                  \n"
        f"PPZ2                         = {_obtener_valor_mf52(d, 'qPPZ2', 0.0):.10f}                  \n"
        f"SSZ1                         = {_obtener_valor_mf52(d, 'qSSZ1', 0.0):.10f}                  \n"
        f"SSZ2                         = {_obtener_valor_mf52(d, 'qSSZ2', 0.0):.10f}                  \n"
        f"SSZ3                         = {_obtener_valor_mf52(d, 'qSSZ3', 0.0):.10f}                  \n"
        f"SSZ4                         = {_obtener_valor_mf52(d, 'qSSZ4', 0.0):.10f}                  \n"
        "$-------------------------------------------------turnslip coefficients\n"
        "[TURNSLIP_COEFFICIENTS]\n"
        f"PDXP1                        = {_obtener_valor_mf52(d, 'PDXP1', 0.0):.10f}                  \n"
        f"PDXP2                        = {_obtener_valor_mf52(d, 'PDXP2', 0.0):.10f}                  \n"
        f"PDXP3                        = {_obtener_valor_mf52(d, 'PDXP3', 0.0):.10f}                  \n"
        f"PKYP1                        = {_obtener_valor_mf52(d, 'PKYP1', 0.0):.10f}                  \n"
        f"PDYP1                        = {_obtener_valor_mf52(d, 'PDYP1', 0.0):.10f}                  \n"
        f"PDYP2                        = {_obtener_valor_mf52(d, 'PDYP2', 0.0):.10f}                  \n"
        f"PDYP3                        = {_obtener_valor_mf52(d, 'PDYP3', 0.0):.10f}                  \n"
        f"PDYP4                        = {_obtener_valor_mf52(d, 'PDYP4', 0.0):.10f}                  \n"
        f"PHYP1                        = {_obtener_valor_mf52(d, 'PHYP1', 0.0):.10f}                  \n"
        f"PHYP2                        = {_obtener_valor_mf52(d, 'PHYP2', 0.0):.10f}                  \n"
        f"PHYP3                        = {_obtener_valor_mf52(d, 'PHYP3', 0.0):.10f}                  \n"
        f"PHYP4                        = {_obtener_valor_mf52(d, 'PHYP4', 0.0):.10f}                  \n"
        f"PECP1                        = {_obtener_valor_mf52(d, 'PECP1', 0.0):.10f}                  \n"
        f"PECP2                        = {_obtener_valor_mf52(d, 'PECP2', 0.0):.10f}                  \n"
        f"QDTP1                        = {_obtener_valor_mf52(d, 'QDTP1', 0.0):.10f}                  \n"
        f"QCRP1                        = {_obtener_valor_mf52(d, 'QCRP1', 0.0):.10f}                  \n"
        f"QCRP2                        = {_obtener_valor_mf52(d, 'QCRP2', 0.0):.10f}                  \n"
        f"QBRP1                        = {_obtener_valor_mf52(d, 'QBRP1', 0.0):.10f}                  \n"
        f"QDRP1                        = {_obtener_valor_mf52(d, 'QDRP1', 0.0):.10f}                  \n"
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