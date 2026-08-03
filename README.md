# FSAE Tyre Data Processor - e-Tech Racing

Software desarrollado in-house para e-Tech Racing para procesar, limpiar y visualizar datos de neumáticos procedentes del Tyre Test Consortium (TTC). Diseñado para optimizar el rendimiento dinámico del vehículo.

## Estructura del Repositorio

- `data/`: Directorio para los archivos `.dat` crudos del TTC (no incluido en git).
- `plots/`: Gráficas generadas automáticamente.
- `results/`: Datasets procesados en formato `.parquet`.
- `src/`: Lógica modular del proyecto.
- `notebooks/`: Interfaces interactivas (Jupyter).
- `main.py`: Orquestador principal del pipeline.

## Instalación

1.  **Dependencias del Sistema (Rocky Linux):**
    ```bash
    sudo dnf install python3-tkinter
    ```

2.  **Entorno Python:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Flujo de Trabajo

### 1. Procesamiento de Datos
Para cargar los archivos crudos de `data/`, clasificarlos físicamente y generar el dataset base:
```bash
python main.py
```
Este comando crea `results/dataset_clasificado.parquet`.

### 2. Exploración y Visualización
Para inspeccionar los datos de forma gráfica e interactiva:
```bash
python -m src.gui_explorer
```
- **Interfaz:** Permite seleccionar `FZ_nom`, `P_nom` e `IA_nom` mediante menús desplegables dependientes.
- **Visualización:** Al actualizar, plotea automáticamente las curvas de `FX` o `FY` diferenciadas por niveles de carga vertical (`FZ_nom`).

## Desarrollo y Contribución
- Mantener la modularidad: Toda lógica matemática debe residir en `src/`.
- `main.py` solo debe orquestar funciones importadas.
- Usar siempre `pathlib` para manejo de rutas.
- Asegurar tipado (`type hints`) en nuevas funciones.
