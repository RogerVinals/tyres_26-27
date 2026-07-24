import pandas as pd
import numpy as np

def clasificar_datos_ttc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica los datos del TTC en base a las variables nominales (FZ, P, IA, V)
    y el tipo de ensayo según el manual oficial de Calspan.
    """
    print("Clasificando datos físicamente (FZ, P, IA, V, Test Type)...")
    df = df.copy()

    # 1. Variables Nominales (Discretización para agrupar datos)
    # FZ (Load) a la centena más cercana
    df['FZ_nom'] = df['FZ'].round(-2)
    # P (Pressure) al entero más cercano
    df['P_nom'] = df['P'].round(0)
    # IA (Inclination/Camber Angle) al entero más cercano
    df['IA_nom'] = df['IA'].round(0)
    # V (Velocity) a la decena más cercana (ej. 25 mph -> 30, o 45 mph -> 50)
    df['V_nom'] = df['V'].round(-1)

    # 2. Tolerancias físicas
    tol_sa = 1.0   # Grados
    tol_sl = 0.02  # Slip Ratio (SL) recomendado por Calspan en lugar de SR

    # 3. Filtrar datos inútiles (Transitorios a velocidad casi cero)
    # Asumimos que si la velocidad es muy baja, no es Steady State
    v_minima = 5.0 
    df = df[df['V'] > v_minima]

    # Por defecto
    df['test_type'] = 'Combined'

    # 4. Máscaras usando SL (Slip Ratio Tradicional) en lugar de SR
    es_recta = (df['SA'].abs() <= tol_sa)
    es_giro = (df['SA'].abs() > tol_sa)
    es_rodadura_libre = (df['SL'].abs() <= tol_sl)
    es_frenada_acel = (df['SL'].abs() > tol_sl)

    # 5. Clasificación
    df.loc[es_recta & es_rodadura_libre, 'test_type'] = 'Warmup / Straight'
    df.loc[es_giro & es_rodadura_libre, 'test_type'] = 'Pure Cornering'
    df.loc[es_recta & es_frenada_acel, 'test_type'] = 'Pure Longitudinal'

    df = df.sort_values(by=['run_id', 'FZ_nom', 'P_nom', 'IA_nom', 'test_type'])

    return df
