# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

"""Wrapper hacia el pipeline principal en scripts/main.py.

Mantiene compatibilidad con flujos existentes que ejecutan `python main.py`
desde la raíz del repositorio.
"""

from scripts.main import main

if __name__ == "__main__":
    main()
