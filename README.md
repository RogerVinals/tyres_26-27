# FSAE Tyre Data Processor - e-Tech Racing

*Read this in other languages: [Español](README.es.md)*

In-house software developed for e-Tech Racing to process, clean, and visualize tire data from the Tyre Test Consortium (TTC). Designed to optimize vehicle dynamic performance.

## Repository Structure

```
tyres_26-27/
├── config_ttc.yaml           # Configuration (thresholds, paths)
├── main.py                   # Wrapper → executes scripts/main.py
├── pyproject.toml            # Package configuration (pip install -e .)
├── requirements.txt
├── README.md
├── README.es.md
├── LICENSE
├── data/                     # Raw .dat TTC files (not included in git)
├── docs/                     # Documentation
├── img/
├── matlab/
├── plots/                    # Generated plots (in .gitignore)
├── results/                  # Processed datasets (.parquet, .tir) (in .gitignore)
├── logs/                     # Logs (in .gitignore)
├── notebooks/                # Jupyter notebooks (empty)
├── tests/                    # Unit tests (empty)
└── src/
    └── tyres_ttc/            # Installable Python package
        ├── __init__.py
        ├── parsing.py        # Loading .dat TTC files
        ├── processing.py     # Warmup detection, classification, nominals
        ├── plotting.py       # Visualization
        ├── pacejka_mf52.py   # Magic Formula 5.2 calculation
        ├── fit_ttc.py        # MF5.2 coefficient fitting + .tir output
        └── gui_fitting.py    # GUI fitting tools
├── scripts/
│   ├── __init__.py
│   ├── main.py               # Main pipeline orchestrator
│   └── fit_mf.py             # Entry point for MF fitting only
```

## Installation

Installation steps per operating system are described below. Requires Python 3.10+.

### Linux (Rocky Linux / Fedora)

```bash
sudo dnf install python3-tkinter
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

1. Install Python 3 (recommended with Homebrew):
```bash
brew install python
```
2. Create and activate a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Note: on macOS, if there are issues with tkinter, install tcl-tk with Homebrew (`brew install tcl-tk`) and follow Homebrew's instructions to link it to the used Python.

### Windows

1. Install Python 3 from python.org (check "Add Python to PATH").
2. Open PowerShell or CMD and create/activate the virtual environment:
```powershell
python -m venv venv
# PowerShell
venv\Scripts\Activate.ps1
# CMD
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Installing as a Python Package (Recommended)

To install the package in editable mode (so changes to `src/` are reflected immediately):

```bash
cd /home/rvinyals/Escriptori/mis_repos/tyres_26-27
source venv/bin/activate
pip install -e .
```

This makes the `tyres_ttc` package importable from anywhere and installs two console scripts:
- `ttc-process` → runs the full processing pipeline
- `ttc-fit` → runs the Magic Formula 5.2 coefficient fitting

## Note on .dat Files

The `.dat` files from the Tyre Test Consortium (TTC) are not included in this repository for legal and licensing reasons. If you need the raw data, request access from the test organizer or the TTC. Place the received files inside the `data/` directory (do not upload them to the repository) before running the pipeline.

## Workflow

### 1. Data Processing

To load the raw files from `data/`, classify them physically, and generate the base dataset:

```bash
# Option A: Using the wrapper (development mode)
source venv/bin/activate
export PYTHONPATH=src
python main.py

# Option B: Using the scripts directly
source venv/bin/activate
export PYTHONPATH=src
python scripts/main.py

# Option C: If installed as package
ttc-process
```

This command creates `results/dataset_clasificado.parquet`.

### 2. Magic Formula 5.2 Fitting

To fit Pacejka MF5.2 coefficients and generate the `.tir` file:

```bash
# Option A: Interactive (asks if you want to fit)
source venv/bin/activate
export PYTHONPATH=src
python main.py
# → Answer 'si' when asked

# Option B: Non-interactive (force fitting)
source venv/bin/activate
export PYTHONPATH=src
export MF_FIT=si
python main.py

# Option C: Direct fitting script
source venv/bin/activate
export PYTHONPATH=src
python scripts/fit_mf.py

# Option D: If installed as package
ttc-fit
```

The `.tir` file is saved to `results/coeficientes_mf52.tir`.

### 3. Exploration and Visualization

The pipeline automatically generates plots in `plots/`:
- `plot_raw_*_*.png` — Raw data per run
- `analysis_FY_vs_SA.png` — Global lateral analysis by FZ
- `analysis_FX_vs_SL_todos.png` — All-in-one FX vs SL
- `analysis_FY_vs_SA_todos.png` — All-in-one FY vs SA

## Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `MF_FIT` | `si`, `sí`, `s`, `yes`, `y` | Run MF fitting automatically without asking |
| `MF_FIT` | `no`, `n`, `false`, `0` | Skip MF fitting automatically |

If `MF_FIT` is not set and there is no interactive terminal, the fitting step is skipped automatically.

## Development and Contribution

- Maintain modularity: All mathematical logic must reside in `src/tyres_ttc/`.
- `main.py` should only orchestrate imported functions.
- Always use `pathlib` for path management.
- Ensure type hints in new functions.
- Run from the repository root with `export PYTHONPATH=src` or install with `pip install -e .`.
