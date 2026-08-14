# FSAE Tyre Data Processor - e-Tech Racing

*Lee esto en otro idioma: [English](README.md)*

Software desarrollado in-house para e-Tech Racing para procesar, limpiar y visualizar datos de neumáticos procedentes del Tyre Test Consortium (TTC). Diseñado para optimizar el rendimiento dinámico del vehículo.

## Estructura del Repositorio

- `data/`: Directorio para los archivos `.dat` crudos del TTC (no incluido en git).
- `plots/`: Gráficas generadas automáticamente.
- `results/`: Datasets procesados en formato `.parquet`.
- `src/`: Lógica modular del proyecto.
- `notebooks/`: Interfaces interactivas (Jupyter).
- `main.py`: Orquestador principal del pipeline.
- `matlab/`: Matlab directorio
## Instalación

A continuación se describen pasos de instalación por sistema operativo. Requiere Python 3.10+.

### Linux (Rocky Linux / Fedora)

1. Instalar dependencias del sistema (ej. tkinter):
```bash
sudo dnf install python3-tkinter
```
2. Crear y activar un entorno virtual, luego instalar dependencias Python:
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Debian / Ubuntu

```bash
sudo apt update
sudo apt install python3-tk python3-venv
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS

1. Instalar Python 3 (recomendado con Homebrew):
```bash
brew install python
```
2. Crear y activar entorno virtual e instalar dependencias:
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Nota: en macOS, si hay problemas con tkinter, instalar tcl-tk con Homebrew (`brew install tcl-tk`) y seguir las instrucciones de Homebrew para vincularlo al Python usado.

### Windows

1. Instalar Python 3 desde python.org (marcar "Add Python to PATH").
2. Abrir PowerShell o CMD y crear/activar el entorno virtual:
```powershell
python -m venv venv
# PowerShell
venv\Scripts\Activate.ps1
# CMD
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Nota sobre los archivos .dat

Los archivos `.dat` del Tyre Test Consortium (TTC) no están incluidos en este repositorio por motivos legales y de licencia. Si necesitas los datos crudos, solicita acceso al organizador del ensayo o al TTC. Coloca los archivos recibidos dentro del directorio `data/` (no subirlos al repositorio) antes de ejecutar el pipeline.


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
