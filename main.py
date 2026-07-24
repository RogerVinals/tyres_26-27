
# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
	print("=== e-Tech Racing Tyre Dara Processor ===")

	#Simu de datos dummy para validar que funciona el entorno
	slip_angle = np.linspace(-12, 12, 100) #Ángulo de desliazamiento (degree)
	fy = 1500 *np.sin(np.radians(slip_angle * 8)) #Lat fuerza simulada

	#Data Frame
	df = pd.DataFrame({'Slip_Angle_deg': slip_angle, 'Fy_N': fy})
	print("\nPrimeras filas de análisis:")
	print(df.head())

	#Generador de gráficas de prueba
	plt.figure(figsize=(8, 5))
	plt.plot(df['Slip_Angle_deg'], df['Fy_N'], label='Lat Fuerza (Fy)', color='b')
	plt.title('Curva de Lat Fuerza vs Ángulo de Deslizamiento (Pacejka Dummy)')
	plt.xlabel('Angulo deslizamiento [º]')
	plt.ylabel('Lat Fuerza [N]')
	plt.grid(True)
	plt.legend()
	
	#Guardar gráficas sin abrir(no tengo GUI, todo es terminal)
	plt.savefig('data/tyre_test_plot.png')
	print("\n[OK] Gráfica guardad en 'data/tyre_test_plot.png'")

if __name__ == "__main__":
	main()

