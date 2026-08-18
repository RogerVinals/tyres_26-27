# Copyright (C) 2026 Roger Viñals
# Distributed under the terms of the GNU General Public License v3 (GPLv3)

"""
pacejka_mf52.py
================
Implementación de la Pacejka "Magic Formula 5.2" (MF5.2), siguiendo la estructura de coef. y ecuaciones
desctiras en:

	http://www.racer.nl/reference/pacejka.htm (documentación de Racer)

Este codigo incluye: 
	-Slip puro (pure slip)
	-Slip combinado(combined)
	-Ajuste de coeficientes (fitting) a datos reales usando mínimos cuadrados no lineales 
	(scipy.optimize.least_squares)

CONVENIO DE UNIDADES usado en este módulo (igual que el codigo fuente de Racer, pero en SI)
	-Fz     : Newton (N), positivo tiene sentido hacia abajo
	-kappa  : Slip ratio adimensional (sin %), 0 = free slip
	-alpha  : slip angle en RADIANES
	-gamma  : camber en RADIANES
	-Fx, Fy : Newtons
	-Mz     : N*m

NOTA IMPORTANTE sobre ejes:
    El texto original usa el eje de Racer, al final del cálculo hace 'Fy = -Fy' y 'Mz = -Mz'
    para su convenio interno. Este modulo NO aplica esa negación al final: devuelve Fy y Mz
    en el convenio "estándar" SAE que usan la mayoria de datasets del TTC (Fy positiva cuando el SA es positivo)
    Si tus datos vienen en otro convenio de ejes, puede que necessites invertir el signo de alpha, gamma, Fy o MZ
    antes de comparar.
"""

from dataclasses import dataclass, fields
import numpy as np

#========================================================================
# 1. Coeficientes
#========================================================================

@dataclass
class MF52Coeffs:
    """Coeficiente MF5.2. Todos los valores por defecto = 0 (o 1 para las lambdas de scaling factors), salvo
    unos pocos ejemplos razonables tomados de la página de referencia (pcx1, pdx1, pcy1, pdy1, pky1-3). """

    #Referencia
    Fz0: float = 4500  #N, carga nominal de referencia
    R0: float = 0.3    #m, radio del neumático sin carga

     # ---- Longitudinal puro (Fx0) ----
    pCx1: float = 1.3
    pDx1: float = 1.2
    pDx2: float = 0.0
    pDx3: float = 0.0
    pEx1: float = 0.0
    pEx2: float = 0.0
    pEx3: float = 0.0
    pEx4: float = 0.0
    pKx1: float = 30.0
    pKx2: float = 0.0
    pKx3: float = 0.0
    pHx1: float = 0.0
    pHx2: float = 0.0
    pVx1: float = 0.0
    pVx2: float = 0.0
 
    # ---- Lateral puro (Fy0) ----
    pCy1: float = 1.34
    pDy1: float = 1.04
    pDy2: float = 0.0
    pDy3: float = 0.0
    pEy1: float = 0.0
    pEy2: float = 0.0
    pEy3: float = 0.0
    pEy4: float = 0.0
    pKy1: float = -25.0
    pKy2: float = 2.71
    pKy3: float = 0.39
    pHy1: float = 0.0
    pHy2: float = 0.0
    pHy3: float = 0.0
    pVy1: float = 0.0
    pVy2: float = 0.0
    pVy3: float = 0.0
    pVy4: float = 0.0
 
    # ---- Momento autoalineante puro (Mz0) ----
    qBz1: float = 0.0
    qBz2: float = 0.0
    qBz3: float = 0.0
    qBz4: float = 0.0
    qBz5: float = 0.0   # (no se usa en Bt del MF5.2 "clásico" del texto, pero se deja disponible)
    qBz9: float = 0.0
    qBz10: float = 0.0
    qCz1: float = 1.0
    qDz1: float = 0.0
    qDz2: float = 0.0
    qDz3: float = 0.0
    qDz4: float = 0.0
    qDz6: float = 0.0
    qDz7: float = 0.0
    qDz8: float = 0.0
    qDz9: float = 0.0
    qEz1: float = 0.0
    qEz2: float = 0.0
    qEz3: float = 0.0
    qEz4: float = 0.0
    qEz5: float = 0.0
    qHz1: float = 0.0
    qHz2: float = 0.0
    qHz3: float = 0.0
    qHz4: float = 0.0
    ssz1: float = 0.0
    ssz2: float = 0.0
    ssz3: float = 0.0
    ssz4: float = 0.0
  

    # ---- Slip combinado: peso longitudinal Gxa ----
    rBx1: float = 1.0
    rBx2: float = 1.0
    rCx1: float = 1.0
    rHx1: float = 0.0
    rEx1: float = 0.0
    rEx2: float = 0.0
 
    # ---- Slip combinado: peso lateral Gyk + Svyk ----
    rBy1: float = 1.0
    rBy2: float = 1.0
    rBy3: float = 0.0
    rCy1: float = 1.0
    rEy1: float = 0.0
    rEy2: float = 0.0
    rHy1: float = 0.0
    rHy2: float = 0.0
    rVy1: float = 0.0
    rVy2: float = 0.0
    rVy3: float = 0.0
    rVy4: float = 1.0
    rVy5: float = 1.0
    rVy6: float = 1.0
 
    # ---- Factores de escala lambda (1.0 = sin escalado) ----
    lCx: float = 1.0
    lMUx: float = 1.0
    lEx: float = 1.0
    lKx: float = 1.0
    lHx: float = 1.0
    lVx: float = 1.0
    lGammax: float = 1.0
 
    lCy: float = 1.0
    lMUy: float = 1.0
    lEy: float = 1.0
    lKy: float = 1.0
    lHy: float = 1.0
    lVy: float = 1.0
    lGammay: float = 1.0
 
    lFz0: float = 1.0
    lGammaz: float = 1.0
    lt: float = 1.0
    lr: float = 1.0
    lXal: float = 1.0
    lYka: float = 1.0
    lVyka: float = 1.0
    ls: float = 1.0

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}



EPS = 1e-6


def _safe_div(a, b, eps=1e-3):
    b = np.asarray(b, dtype=float)
    small = np.abs(b) < eps
    b_safe = np.where(small, b + eps, b)
    return a / b_safe
 
 
def _sign_nonzero(x):
    s = np.sign(x)
    return np.where(s == 0, 1.0, s)
 
 
def _dfz(Fz, c: MF52Coeffs):
    Fz0_prime = c.lFz0 * c.Fz0
    dfz = (Fz - Fz0_prime) / Fz0_prime
    return dfz, Fz0_prime


 
# ==========================================================================
# 2. Slip puro
# ==========================================================================
 
def pure_slip_fx(kappa, Fz, gamma, c: MF52Coeffs):
    """Fx0: fuerza longitudinal en slip puro (alpha = 0).
 
    Devuelve (Fx0, extra) donde extra contiene Bx, Cx, Dx, Ex, Kx
    (necesarios luego para el momento autoalineante y el slip combinado).
    """
    kappa = np.asarray(kappa, dtype=float)
    Fz = np.asarray(Fz, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
 
    dfz, _ = _dfz(Fz, c)
    gamma_x = gamma * c.lGammax
 
    Shx = (c.pHx1 + c.pHx2 * dfz) * c.lHx
    Svx = Fz * (c.pVx1 + c.pVx2 * dfz) * c.lVx * c.lMUx
 
    kx = kappa + Shx
    sign_kx = _sign_nonzero(kx)
 
    Cx = c.pCx1 * c.lCx
    mux = (c.pDx1 + c.pDx2 * dfz) * (1.0 - c.pDx3 * gamma_x ** 2) * c.lMUx
    Ex = (c.pEx1 + c.pEx2 * dfz + c.pEx3 * dfz ** 2) * (1.0 - c.pEx4 * sign_kx) * c.lEx
    Ex = np.minimum(Ex, 1.0)
 
    Dx = mux * Fz
    Kx = Fz * (c.pKx1 + c.pKx2 * dfz) * np.exp(c.pKx3 * dfz) * c.lKx
 
    Bx = _safe_div(Kx, Cx * Dx)
 
    t = Bx * kx
    Fx0 = Dx * np.sin(Cx * np.arctan(t - Ex * (t - np.arctan(t)))) + Svx
 
    extra = dict(Bx=Bx, Cx=Cx, Dx=Dx, Ex=Ex, Kx=Kx, Shx=Shx, Svx=Svx, dfz=dfz)
    return Fx0, extra
 
 
def pure_slip_fy(alpha, Fz, gamma, c: MF52Coeffs):
    """Fy0: fuerza lateral en slip puro (kappa = 0).
 
    Devuelve (Fy0, extra) con By, Cy, Dy, Ey, Ky, muy, Shy, Svy, say
    (necesarios para Mz0 y para el slip combinado).
    """
    alpha = np.asarray(alpha, dtype=float)
    Fz = np.asarray(Fz, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
 
    dfz, Fz0p = _dfz(Fz, c)
    gamma_y = gamma * c.lGammay
 
    Cy = c.pCy1 * c.lCy
    muy = (c.pDy1 + c.pDy2 * dfz) * (1.0 - c.pDy3 * gamma_y ** 2) * c.lMUy
    Dy = muy * Fz
 
    Ky = (c.pKy1 * Fz0p
          * np.sin(2.0 * np.arctan(_safe_div(Fz, c.pKy2 * Fz0p)))
          * (1.0 - c.pKy3 * np.abs(gamma_y)) * c.lFz0 * c.lKy)
 
    By = _safe_div(Ky, Cy * Dy)
 
    Shy = (c.pHy1 + c.pHy2 * dfz) * c.lHy + c.pHy3 * gamma_y
    Svy = Fz * ((c.pVy1 + c.pVy2 * dfz) * c.lVy
                + (c.pVy3 + c.pVy4 * dfz) * gamma_y) * c.lMUy
 
    say = alpha + Shy
    sign_ay = _sign_nonzero(say)
    Ey = (c.pEy1 + c.pEy2 * dfz) * (1.0 - (c.pEy3 + c.pEy4 * gamma_y) * sign_ay) * c.lEy
    Ey = np.minimum(Ey, 1.0)
 
    t = By * say
    Fy0 = Dy * np.sin(Cy * np.arctan(t - Ey * (t - np.arctan(t)))) + Svy
 
    extra = dict(By=By, Cy=Cy, Dy=Dy, Ey=Ey, Ky=Ky, muy=muy,
                 Shy=Shy, Svy=Svy, say=say, dfz=dfz, Fz0p=Fz0p)
    return Fy0, extra
 
 
def pure_slip_mz(alpha, Fz, gamma, c: MF52Coeffs, fy_extra=None):
    """Mz0: momento autoalineante en slip puro.
 
    Si ya calculaste pure_slip_fy(...), pásalo en fy_extra para no
    recalcular By/Cy/Dy/Ky/Shy/Svy.
    """
    alpha = np.asarray(alpha, dtype=float)
    Fz = np.asarray(Fz, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
 
    if fy_extra is None:
        Fy0, fy_extra = pure_slip_fy(alpha, Fz, gamma, c)
    else:
        Fy0, _ = pure_slip_fy(alpha, Fz, gamma, c)
 
    dfz, Fz0p = _dfz(Fz, c)
    gamma_z = gamma * c.lGammaz
    By, Cy, Ky = fy_extra["By"], fy_extra["Cy"], fy_extra["Ky"]
    Shy, Svy = fy_extra["Shy"], fy_extra["Svy"]
 
    Sht = c.qHz1 + c.qHz2 * dfz + (c.qHz3 + c.qHz4 * dfz) * gamma_z
    alpha_t = alpha + Sht
 
    Ky_safe = np.where(np.abs(Ky) < 1e-3, np.where(Ky < 0, -1e-3, 1e-3), Ky)
    Shf = Shy + Svy / Ky_safe
    alpha_r = alpha + Shf
 
    Bt = ((c.qBz1 + c.qBz2 * dfz + c.qBz3 * dfz ** 2)
          * (1.0 + c.qBz4 * gamma_z + c.qBz5 * np.abs(gamma_z))
          * c.lKy / c.lMUy)
    Ct = c.qCz1
    Dt = (Fz * (c.qDz1 + c.qDz2 * dfz)
          * (1.0 + c.qDz3 * gamma_z + c.qDz4 * gamma_z ** 2)
          * (c.R0 / Fz0p) * c.lt)
    Et = ((c.qEz1 + c.qEz2 * dfz + c.qEz3 * dfz ** 2)
          * (1.0 + (c.qEz4 + c.qEz5 * gamma_z) * (2.0 / np.pi)
             * np.arctan(Bt * Ct * alpha_t)))
    Et = np.minimum(Et, 1.0)
 
    Br = c.qBz9 * c.lKy / c.lMUy + c.qBz10 * By * Cy
 
    t_bt = Bt * alpha_t
    t0 = Dt * np.cos(Ct * np.arctan(t_bt - Et * (t_bt - np.arctan(t_bt)))) * np.cos(alpha)
 
    Dr = Fz * ((c.qDz6 + c.qDz7 * dfz) * c.lr
               + (c.qDz8 + c.qDz9 * dfz) * gamma_z) * c.R0 * c.lMUy
 
    Mzr = Dr * np.cos(np.arctan(Br * alpha_r)) * np.cos(alpha)
 
    Mz0 = -t0 * Fy0 + Mzr
 
    extra = dict(Bt=Bt, Ct=Ct, Dt=Dt, Et=Et, Br=Br, Dr=Dr,
                  alpha_t=alpha_t, alpha_r=alpha_r, t0=t0, Mzr=Mzr, dfz=dfz)
    return Mz0, extra
 
 
# ==========================================================================
# 3. Slip combinado
# ==========================================================================
 
def combined_slip(alpha, kappa, Fz, gamma, c: MF52Coeffs):
    """Fx, Fy, Mz en slip combinado (alpha y kappa simultáneos).
 
    alpha en radianes, kappa adimensional, Fz en N, gamma en radianes.
    Devuelve (Fx, Fy, Mz).
    """
    alpha = np.asarray(alpha, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    Fz = np.asarray(Fz, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
 
    dfz, Fz0p = _dfz(Fz, c)
 
    Fx0, x_ex = pure_slip_fx(kappa, Fz, gamma, c)
    Fy0, y_ex = pure_slip_fy(alpha, Fz, gamma, c)
    Mz0, z_ex = pure_slip_mz(alpha, Fz, gamma, c, fy_extra=y_ex)
 
    Bx, Cx, Kx = x_ex["Bx"], x_ex["Cx"], x_ex["Kx"]
    By, Cy, Ky, muy = y_ex["By"], y_ex["Cy"], y_ex["Ky"], y_ex["muy"]
 
    # ---- Peso longitudinal Gxa ----
    Shxa = c.rHx1
    Exa = c.rEx1 + c.rEx2 * dfz
    Cxa = c.rCx1
    Bxa = c.rBx1 * np.cos(np.arctan(c.rBx2 * kappa)) * c.lXal
 
    alpha_s = alpha + Shxa
    t_h = Bxa * Shxa
    Gxa0 = np.cos(Cxa * np.arctan(t_h - Exa * (t_h - np.arctan(t_h))))
    t_s = Bxa * alpha_s
    Gxa_num = np.cos(Cxa * np.arctan(t_s - Exa * (t_s - np.arctan(t_s))))
    Gxa = np.where(np.abs(Gxa0) > EPS, Gxa_num / Gxa0, 0.0)
 
    Fx = Gxa * Fx0
 
    # ---- Peso lateral Gyk + Svyk ----
    Dvyk = muy * Fz * (c.rVy1 + c.rVy2 * dfz + c.rVy3 * gamma) * np.cos(np.arctan(c.rVy4 * alpha))
    Svyk = Dvyk * np.sin(c.rVy5 * np.arctan(c.rVy6 * kappa)) * c.lVyka
    Shyk = c.rHy1 + c.rHy2 * dfz
    Eyk = c.rEy1 + c.rEy2 * dfz
    Cyk = c.rCy1
    Byk = c.rBy1 * np.cos(np.arctan(c.rBy2 * (alpha - c.rBy3))) * c.lYka
 
    ks = kappa + Shyk
    t_hy = Byk * Shyk
    Gyk0 = np.cos(Cyk * np.arctan(t_hy - Eyk * (t_hy - np.arctan(t_hy))))
    t_ky = Byk * ks
    Gyk_num = np.cos(Cyk * np.arctan(t_ky - Eyk * (t_ky - np.arctan(t_ky))))
    Gyk = np.where(np.abs(Gyk0) > EPS, Gyk_num / Gyk0, 0.0)
 
    Fy = Gyk * Fy0 + Svyk
 
    # ---- Momento autoalineante combinado ----
    Ky_safe = np.where(np.abs(Ky) < 1e-3, np.where(Ky < 0, -1e-3, 1e-3), Ky)
    kk = (Kx / Ky_safe) ** 2 * kappa ** 2
 
    alpha_t, alpha_r = z_ex["alpha_t"], z_ex["alpha_r"]
    sign_at = _sign_nonzero(alpha_t)
    sign_ar = _sign_nonzero(alpha_r)
    alpha_t_eq = np.sqrt(alpha_t ** 2 + kk) * sign_at
    alpha_r_eq = np.sqrt(alpha_r ** 2 + kk) * sign_ar
 
    s = (c.ssz1 + c.ssz2 * (Fy / Fz0p) + (c.ssz3 + c.ssz4 * dfz) * gamma) * c.R0 * c.ls
 
    Br, Dt, Ct, Et = z_ex["Br"], z_ex["Dt"], z_ex["Ct"], z_ex["Et"]
    Dr = z_ex["Dr"]
 
    Mzr = Dr * np.cos(np.arctan(Br * alpha_r_eq)) * np.cos(alpha)
    Fy_der = Fy - Svyk
 
    Bt = z_ex["Bt"]
    tt = Bt * alpha_t_eq
    t_ = Dt * np.cos(Ct * np.arctan(tt - Et * (tt - np.arctan(tt)))) * np.cos(alpha)
 
    Mz = -t_ * Fy_der + Mzr + s * Fx
 
    return Fx, Fy, Mz
