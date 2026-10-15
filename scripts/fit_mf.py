# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

"""Punto de entrada para ajustar coeficientes Magic Formula 5.2.

Uso:
    python scripts/fit_mf.py
    MF_FIT=si python scripts/fit_mf.py
    ttc-fit  (después de instalar el paquete con pip install -e .)
"""

from tyres_ttc.fit_ttc import main

if __name__ == "__main__":
    main()
