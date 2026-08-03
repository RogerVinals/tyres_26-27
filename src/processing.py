import pandas as pd
import numpy as np

def clasificar_datos_ttc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica los datos del TTC de forma robusta para facilitar el ploteo.
    """
    print("Clasificando datos para análisis visual...")
    df = df.copy()

    # 1. Variables Nominales (Discretización para agrupar datos)
    # FZ (Load) dividido en 4 grupos según densidad (cuartiles)
    df['FZ_group'] = pd.qcut(df['FZ'], q=4, labels=False)
    fz_means = df.groupby('FZ_group')['FZ'].mean().round(0)
    df['FZ_nom'] = df['FZ_group'].map(fz_means)
    
    # P (Pressure) dividido en 4 grupos según densidad (cuartiles)
    df['P_group'] = pd.qcut(df['P'], q=4, labels=False)
    p_means = df.groupby('P_group')['P'].mean().round(0)
    df['P_nom'] = df['P_group'].map(p_means)
    
    # IA (Inclination/Camber Angle) restringido a enteros de -4 a 4
    df['IA_nom'] = df['IA'].round(0).clip(-4, 4)
    
    # V (Velocity) a la decena más cercana
    df['V_nom'] = df['V'].round(-1)

    # 2. Clasificación de tipo de ensayo (Test Type)
    tol_sa = 1.0   # Grados
    tol_sl = 0.02  # Slip Ratio
    
    df['test_type'] = 'Combined'
    es_recta = (df['SA'].abs() <= tol_sa)
    es_giro = (df['SA'].abs() > tol_sa)
    es_rodadura_libre = (df['SL'].abs() <= tol_sl)
    es_frenada_acel = (df['SL'].abs() > tol_sl)

    df.loc[es_recta & es_rodadura_libre, 'test_type'] = 'Warmup / Straight'
    df.loc[es_giro & es_rodadura_libre, 'test_type'] = 'Pure Cornering'
    df.loc[es_recta & es_frenada_acel, 'test_type'] = 'Pure Longitudinal'

    return df.sort_values(by=['run_id', 'FZ_nom', 'P_nom', 'IA_nom', 'test_type'])


def limpiar_datos(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Aplica una mediana móvil para reducir ruido por run_id."""
    df_clean = df.copy()
    # Aplicar rolling median sobre columnas de fuerza
    for col in ['FX', 'FY']:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby('run_id')[col].transform(
                lambda x: x.rolling(window=window, center=True).median()
            )
    return df_clean.dropna()


def filtrar_datos(df: pd.DataFrame, criterios: dict) -> pd.DataFrame:
    """Filtra el dataframe según criterios nominales."""
    df_filt = df.copy()
    for col, valor in criterios.items():
        if col in df_filt.columns:
            df_filt = df_filt[df_filt[col] == valor]
    return df_filt
