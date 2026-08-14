# FSAE Tyre Data Processor - e-Tech Racing

 *Read this in other languages: [Español](README.es.md)*

In-house software developed for e-Tech Racing to process, clean, and visualize tire data from the Tyre Test Consortium (TTC). Designed to optimize vehicle dynamic performance.

## Repository Structure

- `data/`: Directory for raw `.dat` TTC files (not included in git).
- `plots/`: Automatically generated plots.
- `results/`: Processed datasets in `.parquet` format.
- `src/`: Modular logic of the project.
- `notebooks/`: Interactive interfaces (Jupyter).
- `main.py`: Main pipeline orchestrator.
- `matlab/`: Matlab directory

## Installation

Installation steps per operating system are described below. Requires Python 3.10+.

### Linux (Rocky Linux / Fedora)

1. Install system dependencies (e.g., tkinter):
```bash
sudo dnf install python3-tkinter
```
2. Create and activate a virtual environment, then install Python dependencies:
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

## Note on .dat files

The `.dat` files from the Tyre Test Consortium (TTC) are not included in this repository for legal and licensing reasons. If you need the raw data, request access from the test organizer or the TTC. Place the received files inside the `data/` directory (do not upload them to the repository) before running the pipeline.

## Workflow

### 1. Data Processing
To load the raw files from `data/`, classify them physically, and generate the base dataset:
```bash
python main.py
```
This command creates `results/dataset_clasificado.parquet`.

### 2. Exploration and Visualization
To graphically and interactively inspect the data:
```bash
python -m src.gui_explorer
```
- **Interface:** Allows selecting `FZ_nom`, `P_nom`, and `IA_nom` through dependent dropdown menus.
- **Visualization:** Upon updating, automatically plots the `FX` or `FY` curves differentiated by vertical load levels (`FZ_nom`).

## Development and Contribution
- Maintain modularity: All mathematical logic must reside in `src/`.
- `main.py` should only orchestrate imported functions.
- Always use `pathlib` for path management.
- Ensure type hints in new functions.
