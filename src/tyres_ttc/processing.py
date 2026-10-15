import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# UMBRALES CONFIGURABLES (valores por defecto; se sobrescriben si se pasa config)
# ============================================================================
WARMUP_FRAC_PARADO = 0.90       # fracción de puntos con V <= 5 kph
WARMUP_MAX_SA = 1.0             # deg, |SA| máximo para línea recta
WARMUP_MAX_STD_SL = 0.01        # dispersión de slip sin barrido
SL_DISPERSION_BAJA = 0.035      # std(SL) umbral para no-longitudinal
SA_MEDIANA_RECTA = 1.0          # |median(SA)| umbral para puro vs combinado


def cargar_umbrales(config_path: str | Path = "config_ttc.yaml") -> dict:
    """Carga umbrales de warmup y clasificación desde config_ttc.yaml."""
    import yaml
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return {
            'warmup_frac_parado': cfg.get('warmup', {}).get('fraccion_parado', WARMUP_FRAC_PARADO),
            'warmup_max_sa': cfg.get('warmup', {}).get('max_sa_grado', WARMUP_MAX_SA),
            'warmup_max_std_sl': cfg.get('warmup', {}).get('max_std_sl', WARMUP_MAX_STD_SL),
            'sl_dispersion_baja': cfg.get('clasificacion', {}).get('sl_dispersion_baja', SL_DISPERSION_BAJA),
            'sa_mediana_recta': cfg.get('clasificacion', {}).get('sa_mediana_recta', SA_MEDIANA_RECTA),
        }
    except (FileNotFoundError, KeyError, TypeError):
        return {}


# -------------------------------------------------------------------------
# Detección de warmup (vectorizada, sin .apply())
# -------------------------------------------------------------------------

def detectar_runs_warmup(df: pd.DataFrame, frac_parado: float = WARMUP_FRAC_PARADO) -> pd.Series:
    """
    Detecta runs de calentamiento por fracción de tiempo parado.
    Criterio 1: >= frac_parado de puntos con V <= 5 kph.
    """
    if 'V' not in df.columns:
        return pd.Series(False, index=df['run_id'].unique())

    frac = (df['V'] <= 5.0).groupby(df['run_id']).mean()
    return frac >= frac_parado


def marcar_warmup_despues_de_calibracion(
    df: pd.DataFrame,
    max_sa: float = WARMUP_MAX_SA,
    max_std_sl: float = WARMUP_MAX_STD_SL,
) -> pd.Series:
    """
    Criterio 2: línea recta (max|SA| < max_sa) y sin dispersión de slip
    (std(SL_effective) < max_std_sl). Vectorizado.
    """
    max_sa_col = df['SA'].abs().groupby(df['run_id']).max()
    std_sl_col = df['SL_effective'].groupby(df['run_id']).std()
    return (max_sa_col < max_sa) & (std_sl_col < max_std_sl)


# -------------------------------------------------------------------------
# Clasificación principal
# -------------------------------------------------------------------------

def clasificar_datos_ttc(
    df: pd.DataFrame,
    eliminar_warmup: bool = True,
    sl_dispersion_baja: float = SL_DISPERSION_BAJA,
    sa_mediana_recta: float = SA_MEDIANA_RECTA,
    warmup_frac_parado: float = WARMUP_FRAC_PARADO,
    warmup_max_sa: float = WARMUP_MAX_SA,
    warmup_max_std_sl: float = WARMUP_MAX_STD_SL,
) -> pd.DataFrame:
    """
    Clasifica datos TTC: elimina warmup, calibra slip, computa nominales
    y asigna test_type por run.

    Parameters
    ----------
    df : pd.DataFrame
        Salida de cargar_multiples_runs.
    eliminar_warmup : bool
        Si True, elimina runs detectados como calentamiento.
    sl_dispersion_baja : float
        Umbral de std(SL) para considerar que no hay barrido longitudinal.
    sa_mediana_recta : float
        Umbral de |median(SA)| para distinguir puro vs combinado.
    warmup_frac_parado : float
        Fracción mínima de puntos parados para considerar warmup.
    warmup_max_sa : float
        |SA| máximo permitido para considerar línea recta (warmup).
    warmup_max_std_sl : float
        std(SL) máximo para considerar sin barrido (warmup).
    """
    print("Clasificando datos para análisis visual...")
    df = df.copy()

    # 0. Detección warmup (antes del filtro de velocidad)
    es_warmup = detectar_runs_warmup(df, frac_parado=warmup_frac_parado)

    # 0. Limpieza física básica
    df = df[df['V'] > 5.0].copy()

    # 1. Calibración de slip
    referencia_sr = df.groupby('run_id')['SR'].transform('median')
    df['SL_calibrado'] = df['SR'] - referencia_sr

    if 'SL' not in df.columns or df['SL'].isna().all():
        df['SL_effective'] = df['SL_calibrado']
    else:
        df['SL_effective'] = df['SL'].fillna(df['SL_calibrado'])

    # Combinar criterio 2 de warmup (necesita SL_effective)
    if not df.empty:
        recto_sin_barrido = marcar_warmup_despues_de_calibracion(
            df, max_sa=warmup_max_sa, max_std_sl=warmup_max_std_sl
        )
        es_warmup = es_warmup.reindex(recto_sin_barrido.index).fillna(False) | recto_sin_barrido

    if eliminar_warmup and es_warmup.any():
        runs_warmup = sorted(es_warmup[es_warmup].index.tolist())
        logger.info("Eliminando %d run(es) de calentamiento: %s",
                    len(runs_warmup), runs_warmup)
        print(f"[Warmup] Eliminando {len(runs_warmup)} run(es): {runs_warmup}")
        df = df[~df['run_id'].isin(runs_warmup)].copy()

    if df.empty:
        msg = "No quedan datos tras eliminar warmup y filtrar V > 5 kph."
        logger.error(msg)
        raise ValueError(msg)

    # 2. Variables nominales
    df['proyecto'] = df['run_id'].str[:5]

    df['FZ_group'] = df.groupby('proyecto')['FZ'].transform(
        lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop')
    )
    df['FZ_nom'] = df.groupby('FZ_group')['FZ'].transform('mean').round(0)

    df['P_group'] = pd.qcut(df['P'], q=4, labels=False)
    p_means = df.groupby('P_group')['P'].mean().round(0)
    df['P_nom'] = df['P_group'].map(p_means)

    df['IA_nom'] = df['IA'].round(0).clip(-4, 4)
    df['V_nom'] = df['V'].round(-1)

    # 3. Clasificación de tipo de ensayo por run
    std_sl = df.groupby('run_id')['SL_effective'].transform('std')
    median_sa = df.groupby('run_id')['SA'].transform('median')

    df['test_type'] = 'Combined'
    df.loc[std_sl < sl_dispersion_baja, 'test_type'] = 'Pure Cornering'
    df.loc[
        (std_sl >= sl_dispersion_baja) & (median_sa.abs() <= sa_mediana_recta),
        'test_type'
    ] = 'Pure Longitudinal'

    return df.sort_values(['run_id', 'FZ_nom', 'P_nom', 'IA_nom', 'test_type'])


def limpiar_datos(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Mediana móvil por run_id para reducir ruido en FX y FY."""
    df_clean = df.copy()
    for col in ['FX', 'FY']:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean.groupby('run_id')[col]
                .transform(lambda x: x.rolling(window=window, center=True).median())
            )
    return df_clean.dropna()


def filtrar_datos(df: pd.DataFrame, criterios: dict) -> pd.DataFrame:
    """Filtra por criterios de columnas nominales (igualdad exacta)."""
    df_filt = df.copy()
    for col, valor in criterios.items():
        if col in df_filt.columns:
            df_filt = df_filt[df_filt[col] == valor]
    return df_filt


# -------------------------------------------------------------------------
# Filtro para ajuste MF
# -------------------------------------------------------------------------

TIPOS_VALIDOS_MF = ('Pure Cornering', 'Pure Longitudinal', 'Combined')


def filtrar_para_ajuste_mf(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo filas con test_type válido para ajuste MF5.2."""
    if 'test_type' not in df.columns:
        raise ValueError(
            "El DataFrame no tiene 'test_type'. Ejecuta clasificar_datos_ttc primero."
        )
    mask = df['test_type'].isin(TIPOS_VALIDOS_MF)
    n_descartados = int((~mask).sum())
    if n_descartados > 0:
        logger.info(
            "Ajuste MF: descartados %d punto(s) de test_type no válido.",
            n_descartados,
        )
    return df[mask].copy()
