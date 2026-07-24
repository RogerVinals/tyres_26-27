# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

import sys
import matplotlib.pyplot as plt
from src.parsing import cargar_dat_ttc

def main():
    print("=== FSAE Tyre Data Processor ===")
    
    # IMPORTANTE: Cambia 'Run38.dat' por el nombre exacto de tu archivo en la carpeta data/
    archivo_datos = "data/B2356run4.dat"  
    
    try:
        df = cargar_dat_ttc(archivo_datos)
        print(f"[OK] Datos cargados. Filas: {len(df)}, Columnas: {len(df.columns)}")
        
        # Generar gráfica real de SA vs FY
        plt.figure(figsize=(10, 6))
        
        # Usamos scatter porque la telemetría cruda tiene mucho ruido
        plt.scatter(df['SA'], df['FY'], alpha=0.5, s=5, c='darkorange')
        
        plt.title('Raw Tyre Data: Fuerza Lateral vs Ángulo de Deslizamiento')
        plt.xlabel('Slip Angle (SA) [deg]')
        plt.ylabel('Lateral Force (FY) [N]')
        plt.grid(True)
        
        # Guardamos la gráfica
        ruta_salida = 'plots/SA_vs_FY_raw.png'
        plt.savefig(ruta_salida)
        print(f"\n[OK] Gráfica de dispersión guardada en '{ruta_salida}'")
        
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Fallo inesperado procesando los datos: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
