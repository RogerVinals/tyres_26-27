import pandas as pd
import numpy as np

def clasificar_datos_ttc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica los datos del TTC de forma robusta para facilitar el ploteo.
    Incluye calibración dinámica de Slip, cálculo de FZ nominal por proyecto 
    y clasificación por fuerza dominante del run.
    """
    print("Clasificando datos para análisis visual...")
    df = df.copy()

    # ---------------------------------------------------------
    # 0. LIMPIEZA FÍSICA BÁSICA (Quitar ruido de máquina parada)
    # ---------------------------------------------------------
    df = df[df['V'] > 5.0].copy()

    # ---------------------------------------------------------
    # 1. CALIBRACIÓN DE SLIP (Corrección de Schema Drift)
    # ---------------------------------------------------------
    referencia_sr = df.groupby('run_id')['SR'].transform('median')
    df['SL_calibrado'] = df['SR'] - referencia_sr
    
    if 'SL' not in df.columns or df['SL'].isna().all():
        df['SL_effective'] = df['SL_calibrado']
    else:
        df['SL_effective'] = df['SL'].fillna(df['SL_calibrado'])

    # Eliminar picos de ruido matemáticamente absurdos 
    df = df[df['SL_effective'].between(-1.5, 1.5)]

    # ---------------------------------------------------------
    # 2. Variables Nominales (Discretización)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 3. Clasificación de tipo de ensayo (Test Type) POR RUN
    # ---------------------------------------------------------
    #Se usa estadistica para determinar la run
    std_sl = df.groupby('run_id')['SL_effective'].transform('std')
    median_sa = df.groupby('run_id')['SA'].transform('median')

    df['test_type'] = 'Combined'
    
    # Si la dispersión del Slip es baja (< 0.035), no es un ensayo longitudinal
    df.loc[std_sl < 0.035, 'test_type'] = 'Pure Cornering'
    
    # Si la dispersión del Slip es alta, la máquina lo estaba barriendo a propósito.
    # Miramos la mediana del ángulo para saber si es puro o combinado.
    df.loc[(std_sl >= 0.035) & (median_sa.abs() <= 1.0), 'test_type'] = 'Pure Longitudinal'
    
    return df.sort_values(by=['run_id', 'FZ_nom', 'P_nom', 'IA_nom', 'test_type'])

def limpiar_datos(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Aplica una mediana móvil para reducir ruido por run_id."""
    df_clean = df.copy()
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
