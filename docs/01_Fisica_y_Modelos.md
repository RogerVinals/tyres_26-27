El sistema de cordenadas usado para entender los modelos es el [[00_Sistema_Coord|SAE J670e]] y los datos que se usan para interpolar el modelo se sacan del [[02_Datos_TTC|Tire Test Consortium (TTC)]]

# Modelo Avanzado: Pacejka Magic Formula 5.2 (PAC2002)

La versión 5.2 de la Fórmula Mágica de Hans Pacejka (MF-Tyre) es el modelo estándar en software multicuerpo como ADAMS/Car y simulación de tiempos por vuelta. 

A diferencia de la versión simplificada, la MF 5.2 no requiere ajustar una curva separada para cada carga vertical o camber. En su lugar, utiliza un conjunto masivo de **coeficientes empíricos** (nombrados con letras como `p`, `q`, `r`) que escalan la curva base en función de la carga dinámica y la cinemática de la suspensión.

---

## 1. Variables de Estado (Inputs)
Para calcular las fuerzas en un instante dado, el modelo requiere cuatro entradas físicas continuas:
*   **$\alpha$**: Ángulo de deslizamiento lateral (Slip Angle).
*   **$\kappa$**: Ratio de deslizamiento longitudinal (Slip Ratio).
*   **$\gamma$**: Ángulo de caída (Camber / Inclination Angle).
*   **$F_z$**: Carga vertical instantánea aplicada sobre el neumático (siempre positiva para las fórmulas de Pacejka, a pesar de usar J670e para el resto del coche).

Además, se define la **variación de carga normalizada** ($df_z$). Esto permite que los coeficientes escalen de forma adimensional respecto a la carga de diseño del neumático ($F_{z0}$):

$$df_z = \frac{F_z - F_{z0}}{F_{z0}}$$

---

## 2. Ecuaciones de Fuerza Lateral Pura (Pure Cornering)

El cálculo de $F_y$ asume rodadura libre ($\kappa = 0$). Los coeficientes a optimizar mediante *Curve Fitting* empiezan por **$p_y$** (parámetros laterales).

### A. Desplazamientos (Shifts)
Modelan las asimetrías estructurales del neumático (Ply Steer y Conicity). Hacen que la curva no pase exactamente por el origen (0,0).
*   **Desplazamiento Horizontal ($S_{Hy}$):** 
    $$S_{Hy} = p_{Hy1} + p_{Hy2} \cdot df_z + p_{Hy3} \cdot \gamma$$
*   **Desplazamiento Vertical ($S_{Vy}$):**
    $$S_{Vy} = F_z \cdot (p_{Vy1} + p_{Vy2} \cdot df_z) + (p_{Vy3} + p_{Vy4} \cdot df_z) \cdot \gamma \cdot F_z$$
*   **Ángulo de Deriva Equivalente ($\alpha_y$):**
    $$\alpha_y = \alpha + S_{Hy}$$

### B. Factores Principales (La Magia)
*   **Fricción Pico ($\mu_y$):** Disminuye a medida que aumenta la carga normal (Sensitivity) y el camber.
    $$\mu_y = (p_{Dy1} + p_{Dy2} \cdot df_z) \cdot (1 - p_{Dy3} \cdot \gamma^2)$$
*   **Factor Pico ($D_y$):** La fuerza lateral máxima absoluta.
    $$D_y = \mu_y \cdot F_z$$
*   **Factor de Forma ($C_y$):** Constante geométrica de la curva.
    $$C_y = p_{Cy1}$$
*   **Rigidez de Deriva - Cornering Stiffness ($K_{y\alpha}$):** Altamente no lineal respecto a la carga.
    $$K_{y\alpha} = p_{Ky1} \cdot F_{z0} \cdot \sin(p_{Ky2} \cdot \arctan(F_z / (p_{Ky3} \cdot F_{z0}))) \cdot (1 - p_{Ky4} \cdot |\gamma|)$$
*   **Factor de Rigidez ($B_y$):**
    $$B_y = \frac{K_{y\alpha}}{C_y \cdot D_y}$$
*   **Factor de Curvatura ($E_y$):** Controla el comportamiento post-pico (límite de adherencia).
    $$E_y = (p_{Ey1} + p_{Ey2} \cdot df_z) \cdot (1 - (p_{Ey3} + p_{Ey4} \cdot \gamma) \cdot \text{sgn}(\alpha_y))$$

### C. Ecuación Final ($F_{y0}$)
Agrupando todos los términos:
$$F_{y0} = D_y \cdot \sin(C_y \cdot \arctan(B_y \cdot \alpha_y - E_y \cdot (B_y \cdot \alpha_y - \arctan(B_y \cdot \alpha_y)))) + S_{Vy}$$

---

## 3. Ecuaciones de Fuerza Longitudinal Pura (Pure Braking/Traction)

El cálculo de $F_x$ asume ir en línea recta ($\alpha = 0$). Los coeficientes empiezan por **$p_x$**.

### A. Desplazamientos
$$S_{Hx} = p_{Hx1} + p_{Hx2} \cdot df_z$$
$$S_{Vx} = F_z \cdot (p_{Vx1} + p_{Vx2} \cdot df_z)$$
$$\kappa_x = \kappa + S_{Hx}$$

### B. Factores Principales
*   **Fricción Pico ($\mu_x$):**
    $$\mu_x = (p_{Dx1} + p_{Dx2} \cdot df_z) \cdot (1 - p_{Dx3} \cdot \gamma^2)$$
*   **Factor Pico ($D_x$):**
    $$D_x = \mu_x \cdot F_z$$
*   **Rigidez Longitudinal ($K_{x\kappa}$):**
    $$K_{x\kappa} = F_z \cdot (p_{Kx1} + p_{Kx2} \cdot df_z) \cdot \exp(p_{Kx3} \cdot df_z)$$
*   **Factores Base:**
    $$C_x = p_{Cx1}$$
    $$B_x = \frac{K_{x\kappa}}{C_x \cdot D_x}$$
    $$E_x = (p_{Ex1} + p_{Ex2} \cdot df_z + p_{Ex3} \cdot df_z^2) \cdot (1 - p_{Ex4} \cdot \text{sgn}(\kappa_x))$$

### C. Ecuación Final ($F_{x0}$)
$$F_{x0} = D_x \cdot \sin(C_x \cdot \arctan(B_x \cdot \kappa_x - E_x \cdot (B_x \cdot \kappa_x - \arctan(B_x \cdot \kappa_x)))) + S_{Vx}$$

---

## 4. Momento de Auto-Alineamiento (Aligning Torque, $M_z$)

El momento de auto-alineamiento es la fuerza de torsión que siente el piloto en el volante. Es fundamental para el *steering feel*. Físicamente, la fuerza lateral $F_y$ no se aplica en el centro exacto del parche de contacto, sino un poco más atrás. Esa distancia se llama **Pneumatic Trail** ($t$). Además, las asimetrías del neumático generan un momento residual ($M_{zr}$).

La ecuación general en el sistema coordenado SAE es:
$$M_z = -t \cdot F_{y0} + M_{zr}$$

### A. Pneumatic Trail ($t$)
Se modela con otra curva de Pacejka, pero usando la función coseno (ya que el trail es máximo cuando se va recto y disminuye a medida que aumenta el ángulo de deriva). Los coeficientes empíricos a ajustar empiezan por **$q_z$**.
$$t = D_t \cdot \cos(C_t \cdot \arctan(B_t \cdot \alpha_t - E_t \cdot (B_t \cdot \alpha_t - \arctan(B_t \cdot \alpha_t)))) \cdot \cos(\alpha)$$

### B. Momento Residual ($M_{zr}$)
El momento que existe incluso a cero grados de deriva debido a la caída (camber) y la conicidad.
$$M_{zr} = D_r \cdot \cos(C_r \cdot \arctan(B_r \cdot \alpha_r))$$

---

## 5. Deslizamiento Combinado (Combined Slip)

En un circuito de carreras, rara vez estamos en "Pure Cornering" o "Pure Braking". El piloto suele frenar mientras gira el volante (Trail Braking) o acelerar saliendo de la curva. La MF 5.2 utiliza la elipse de adherencia multiplicando las fuerzas puras ($F_{y0}$ y $F_{x0}$) por unas funciones de ponderación ($G_{x\alpha}$ y $G_{y\kappa}$).

### A. Fuerza Longitudinal bajo Deriva Lateral ($F_x$)
Si el neumático está girado, pierde capacidad de frenado. Los coeficientes para esta interacción empiezan por **$r_x$**.
$$F_x = G_{x\alpha} \cdot F_{x0}$$

La función $G_{x\alpha}$ también tiene forma de campana (coseno de Pacejka):
$$G_{x\alpha} = \cos(C_{x\alpha} \cdot \arctan(B_{x\alpha} \cdot \alpha_s - E_{x\alpha} \cdot (B_{x\alpha} \cdot \alpha_s - \arctan(B_{x\alpha} \cdot \alpha_s))))$$

### B. Fuerza Lateral bajo Deslizamiento Longitudinal ($F_y$)
Si bloqueamos frenos o patinamos acelerando, perdemos agarre lateral. Los coeficientes empiezan por **$r_y$**.
$$F_y = G_{y\kappa} \cdot F_{y0}$$

$$G_{y\kappa} = \cos(C_{y\kappa} \cdot \arctan(B_{y\kappa} \cdot \kappa_s - E_{y\kappa} \cdot (B_{y\kappa} \cdot \kappa_s - \arctan(B_{y\kappa} \cdot \kappa_s))))$$

---

## 6. Modelo Transitorio (Relaxation Length)

La Fórmula Mágica calcula fuerzas en **estado estacionario (Steady-State)**. Sin embargo, en la realidad (y en las pruebas de Calspan que empiezan desde velocidad cero), si giras el volante de golpe, el neumático no genera la fuerza lateral instantáneamente; la carcasa tiene que deformarse a medida que rueda.

Esa distancia que necesita rodar para alcanzar el 63% de su fuerza lateral máxima se llama **Longitud de Relajación** ($\sigma$). Para simulaciones dinámicas multicuerpo, la fuerza lateral real ($F_{y, transient}$) se calcula resolviendo esta ecuación diferencial de primer orden respecto a la fuerza teórica estacionaria de Pacejka ($F_{y, steady}$):

$$\frac{\sigma_y}{V} \cdot \frac{d F_{y, transient}}{dt} + F_{y, transient} = F_{y, steady}$$

Donde $V$ es la velocidad de avance del vehículo. Por eso, en nuestro procesador de datos, descartamos los puntos donde $V < 5.0$ mph: para evitar que el transitorio arruine nuestro ajuste de la curva estacionaria.