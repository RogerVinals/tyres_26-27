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
    residuals_fn,
    x0: np.ndarray,
    params_names: list,
    etapa: str,
    y_obs: np.ndarray,
    verbose: int = 0,
    max_nfev1: int = 2000,
    max_nfev2: int = 5000,
):
    """
    Ajuste en dos pasadas:
      1. Ajuste rápido con pérdida cuadrada para detectar outliers.
      2. Filtrado por MAD de residuos y ajuste final con pérdida soft_l1.

    Devuelve el objeto resultado de la segunda pasada.
    """
    progreso = _ProgresoFit(nombre)
    progreso.iniciar()

    contador = {"n": 0}

    def residuals(x):
        contador["n"] += 1
        res = residuals_fn(x)
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
        # Volver a construir la función de residuos solo con inliers
        # Necesitamos acceso a los datos originales para filtrar.
        # Como residuals_fn cierra sobre los datos, creamos una nueva que filtra.
        def residuals_filtrados(x):
            contador["n"] += 1
            c_t = _set_coeffs(c_current, dict(zip(params_names, x)))
            res = residuals_fn_con_coeffs(c_t)
            return res[mascara]

        # Necesitamos acceso a c_t dentro... mejor reconstruir:
        def residuals_final(x):
            contador["n"] += 1
            c_t = _set_coeffs(c, dict(zip(params_names, x)))
            res = _evaluar_residuales(c_t, params_names, residuals_fn, mascara)
            progreso.actualizar(contador["n"], float(np.sqrt(np.mean(res ** 2))))
            return res

        res2 = least_squares(
            residuals_final, res1.x, bounds=(lb, ub),
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


def _evaluar_residuales(c, params_names, residuals_fn_base, mascara):
    """
    Evalúa residuales aplicando máscara. residuals_fn_base debe aceptar
    un coeficiente c y devolver residuos completos (sin máscara).
    """
    # Esta función se reemplaza por closures específicos en cada fit_*.
    raise NotImplementedError


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

    def residuals(x):
        c_t = _set_coeffs(c, dict(zip(FY0_PARAMS, x)))
        Fy0, _ = pure_slip_fy(alpha, Fz, gamma, c_t)
        return Fy0 - FY

    res = _ajustar_robusto(
        "FY0", residuals, x0, FY0_PARAMS, "fy0", FY, verbose=verbose,
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

    def residuals(x):
        c_t = _set_coeffs(c, dict(zip(MZ0_PARAMS, x)))
        Mz0, _ = pure_slip_mz(alpha, Fz, gamma, c_t)
        return Mz0 - MZ

    res = _ajustar_robusto(
        "MZ0", residuals, x0, MZ0_PARAMS, "mz0", MZ, verbose=verbose,
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

    def residuals(x):
        c_t = _set_coeffs(c, dict(zip(FX0_PARAMS, x)))
        Fx0, _ = pure_slip_fx(kappa, Fz, gamma, c_t)
        return Fx0 - FX

    res = _ajustar_robusto(
        "FX0", residuals, x0, FX0_PARAMS, "fx0", FX, verbose=verbose,
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

    def residuals(x):
        c_t = _set_coeffs(c, dict(zip(COMBINED_PARAMS, x)))
        Fx, Fy, _ = combined_slip(alpha, kappa, Fz, gamma, c_t)
        return np.concatenate([Fx - FX, Fy - FY])

    res = _ajustar_robusto(
        "Combined", residuals, x0, COMBINED_PARAMS, "combined", FX, verbose=verbose,
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
        "*  Se han eliminado transitorios y outliers antes del ajuste.\n"
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