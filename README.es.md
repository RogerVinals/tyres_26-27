# FSAE Tyre Data Processor - e-Tech Racing

*Lee esto en otro idioma: [English](README.md)*

Software desarrollado in-house para e-Tech Racing para procesar, limpiar y visualizar datos de neumáticos procedentes del Tyre Test Consortium (TTC). Diseñado para optimizar el rendimiento dinámico del vehículo.

## Estructura del Repositorio

```
tyres_26-27/
├── config_ttc.yaml           # Configuración (umbrales, rutas)
├── main.py                   # Wrapper → ejecuta scripts/main.py
├── pyproject.toml            # Configuración del paquete (pip install -e .)
├── requirements.txt
├── README.md
├── README.es.md
├── LICENSE
├── data/                     # Archivos .dat crudos del TTC (no incluido en git)
├── docs/                     # Documentación
├── img/
├── matlab/
├── plots/                    # Gráficas generadas automáticamente (en .gitignore)
├── results/                  # Datasets procesados (.parquet, .tir) (en .gitignore)
├── logs/                     # Logs (en .gitignore)
├── notebooks/                # Cuadernos de Jupyter (vacío)
├── tests/                    # Pruebas unitarias (vacío)
└── src/
    └── tyres_ttc/            # Paquete Python instalable
        ├── __init__.py
        ├── parsing.py        # Carga de archivos .dat TTC
        ├── processing.py     # Detección de warmup, clasificación, nominales
        ├── plotting.py       # Visualización
        ├── pacejka_mf52.py   # Cálculo de Magic Formula 5.2
        ├── fit_ttc.py        # Ajuste de coeficientes MF5.2 + salida .tir
        └── gui_fitting.py    # Herramientas de ajuste GUI
├── scripts/
│   ├── __init__.py
│   ├── main.py               # Orquestador principal del pipeline
│   └── fit_mf.py             # Punto de entrada solo para ajuste MF
```

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

## Instalación como Paquete Python (Recomendado)

Para instalar el paquete en modo editable (de modo que los cambios en `src/` se reflejen inmediatamente):

```bash
cd /home/rvinyals/Escriptori/mis_repos/tyres_26-27
source venv/bin/activate
pip install -e .
```

Esto hace que el paquete `tyres_ttc` sea importable desde cualquier lugar e instala dos scripts de consola:
- `ttc-process` → ejecuta el pipeline completo de procesamiento
- `ttc-fit` → ejecuta el ajuste de coeficientes MF5.2

## Nota sobre los archivos .dat

Los archivos `.dat` del Tyre Test Consortium (TTC) no están incluidos en este repositorio por motivos legales y de licencia. Si necesitas los datos crudos, solicita acceso al organizador del ensayo o al TTC. Coloca los archivos recibidos dentro del directorio `data/` (no subirlos al repositorio) antes de ejecutar el pipeline.

## Flujo de Trabajo

### 1. Procesamiento de Datos

Para cargar los archivos crudos de `data/`, clasificarlos físicamente y generar el dataset base:

```bash
# Opción A: Usando el wrapper (modo desarrollo)
source venv/bin/activate
export PYTHONPATH=src
python main.py

# Opción B: Usando los scripts directamente
source venv/bin/activate
export PYTHONPATH=src
python scripts/main.py

# Opción C: Si está instalado como paquete
ttc-process
```

Este comando crea `results/dataset_clasificado.parquet`.

### 2. Ajuste de Magic Formula 5.2

Para ajustar los coeficientes MF5.2 y generar el archivo `.tir`:

```bash
# Opción A: Interactivo (pregunta si se quiere ajustar)
source venv/bin/activate
export PYTHONPATH=src
python main.py
# → Responder 'si' cuando se pregunte

# Opción B: No interactivo (forzar ajuste sin preguntar)
source venv/bin/activate
export PYTHONPATH=src
export MF_FIT=si
python main.py

# Opción C: Script directo de ajuste
source venv/bin/activate
export PYTHONPATH=src
python scripts/fit_mf.py

# Opción D: Si está instalado como paquete
ttc-fit
```

El archivo `.tir` se guarda en `results/coeficientes_mf52.tir`.

### 3. Exploración y Visualización

El pipeline genera automáticamente gráficas en `plots/`:
- `plot_raw_*_*.png` — Datos crudos por run
- `analysis_FY_vs_SA.png` — Análisis lateral global por FZ
- `analysis_FX_vs_SL_todos.png` — Todo-en-uno FX vs SL
- `analysis_FY_vs_SA_todos.png` — Todo-en-uno FY vs SA

## Variables de Entorno

| Variable | Valores | Efecto |
|----------|---------|--------|
| `MF_FIT` | `si`, `sí`, `s`, `yes`, `y` | Ejecutar ajuste MF automáticamente sin preguntar |
| `MF_FIT` | `no`, `n`, `false`, `0` | Saltar ajuste MF automáticamente |

Si `MF_FIT` no está definida y no hay terminal interactivo, el paso de ajuste se salta automáticamente.

## Desarrollo y Contribución

- Mantener la modularidad: Toda lógica matemática debe residir en `src/tyres_ttc/`.
- `main.py` solo debe orquestar funciones importadas.
- Usar siempre `pathlib` para manejo de rutas.
- Asegurar tipado (`type hints`) en nuevas funciones.
- Ejecutar desde la raíz del repositorio con `export PYTHONPATH=src` o instalar con `pip install -e .`.
