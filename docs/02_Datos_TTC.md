# Diccionario de Datos: Canales del Calspan TTC

Todos los archivos `.dat` crudos proporcionados por el Tire Test Consortium (TTC) y generados por el Calspan Tire Research Facility (TIRF) contienen las siguientes columnas. 

**Notas Importantes:**
*   **Frecuencia de muestreo:** Todos los datos se recogen a 100 Hz.
*   **Convención de signos:** Los datos se reportan bajo el estándar [[00_Sistema_Coord|SAE J670e]] (Fuerza Vertical $F_z$ negativa en compresión).

## Lista de Canales Oficiales

| Canal | Descripción Oficial | Unidades (SI / USCS) |
| :--- | :--- | :--- |
| **AMBTMP** | Temperatura ambiente de la sala | degC / degF |
| **ET** | Tiempo transcurrido del ensayo (Elapsed Time) | sec |
| **FX** | Fuerza Longitudinal | N / lb |
| **FY** | Fuerza Lateral | N / lb |
| **FZ** | Carga Normal (Vertical) | N / lb |
| **IA** | Ángulo de Inclinación (Camber) | deg |
| **MX** | Momento de Vuelco (Overturning Moment) | N-m / lb-ft |
| **MZ** | Momento de Auto-alineamiento (Aligning Torque) | N-m / lb-ft |
| **N** | Velocidad de rotación de la rueda | rpm |
| **NFX** | Fuerza longitudinal normalizada ($F_x / F_z$) | adimensional |
| **NFY** | Fuerza lateral normalizada ($F_y / F_z$) | adimensional |
| **P** | Presión del neumático | kPa / psi |
| **RE** | Radio Efectivo | cm / in |
| **RL** | Radio Cargado (Loaded Radius) | cm / in |
| **RST** | Temperatura de la superficie de la pista | degC / degF |
| **SA** | Ángulo de Deriva (Slip Angle) | deg |
| **SL** | Ratio de Deslizamiento basado en RE (Slip Ratio Tradicional) | adimensional |
| **SR** | Ratio de Deslizamiento basado en RL (Control de la máquina) | adimensional |
| **TSTC** | Temperatura superficial del neumático - Centro | degC / degF |
| **TSTI** | Temperatura superficial del neumático - Interior (Inboard) | degC / degF |
| **TSTO** | Temperatura superficial del neumático - Exterior (Outboard) | degC / degF |
| **V** | Velocidad de la pista (Road Speed) | kph / mph |

---

## Advertencia sobre el Deslizamiento Longitudinal (SL vs SR)

El dataset incluye dos variables para el Slip Ratio: `SL` y `SR`.
*   **SR:** Es utilizado internamente por Calspan para el control de la máquina de ensayos, basado en el Radio Cargado (RL). En `SR = 0`, la fuerza longitudinal ($F_x$) no es exactamente cero.
*   **SL:** Es el ratio de deslizamiento "tradicional" de los libros de texto de dinámica vehicular, basado en el Radio Efectivo (RE). Está calibrado de forma que cuando `SL = 0`, la fuerza longitudinal es nula ($F_x = 0$).

**Decisión de Ingeniería:** Todo nuestro procesador de datos utilizará exclusivamente el canal **`SL`** por recomendación directa de Calspan.