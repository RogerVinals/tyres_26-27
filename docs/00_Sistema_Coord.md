![[SAE 670e.png]]

# Sistema de Coordenadas (SAE J670e)

Para garantizar la coherencia en todo el modelo del vehículo y la interpretación de los datos del Tyre Test Consortium (TTC) de Calspan, utilizamos el sistema de coordenadas estándar **SAE J670e**. 

Es vital que todo el equipo de Dinámica Vehicular y Suspensión respete esta convención de signos para evitar fuerzas invertidas en las simulaciones.

## Ejes del Vehículo / Neumático
*   **Eje X (Longitudinal):** Positivo hacia ADELANTE (Dirección de avance del vehículo).
*   **Eje Y (Lateral):** Positivo hacia la DERECHA del piloto.
*   **Eje Z (Vertical):** Positivo hacia ABAJO (hacia el centro de la tierra).

## Fuerzas y Momentos
*   **$F_x$ (Longitudinal Force):** Positiva en aceleración, negativa en frenada.
*   **$F_y$ (Lateral Force):** Fuerza de agarre lateral. 
*   **$F_z$ (Normal Load):** Carga vertical. Debido a que el eje Z es hacia abajo, **$F_z$ es siempre un valor negativo** en compresión contra el asfalto. (Ej: -1100 N).
*   **$M_z$ (Aligning Torque):** Momento de auto-alineamiento.

## Ángulos
*   **$\alpha$ (Slip Angle / Ángulo de Deriva):** Ángulo entre el plano medio del neumático y su vector de velocidad real. 
*   **$\gamma$ (Camber / Inclination Angle):** Ángulo de caída. Positivo cuando la parte superior del neumático se inclina hacia afuera del vehículo (caída positiva), negativo hacia adentro.