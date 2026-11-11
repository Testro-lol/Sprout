#!/usr/bin/env python3
"""Linear normal modes of a viscous 2D drop (a cylinder) in a viscous ambient fluid.

Reference values for `sprout validate surface` (oil drop in water). The inviscid formula
omega^2 = n (n^2 - 1) sigma / ((rho_i + rho_o) R^3) ignores the boundary layers on both
sides of the interface, which are not thin here: with olive-oil viscosity on millimetre
cells, sqrt(2 nu / omega) is about a third of the drop radius.

Shape r = R + eta exp(i n theta + s t), stream function psi(r) exp(i n theta + s t):
    inside   psi = a r^n  + b I_n(q_i r),   q_i^2 = s / nu_i
    outside  psi = c r^-n + d K_n(q_o r),   q_o^2 = s / nu_o
(the power-law parts are irrotational and carry the pressure, p = -/+ i rho s a r^n).
Four conditions at r = R: continuous u_r and u_theta, continuous shear stress, and the
normal-stress jump sigma (n^2 - 1) eta / R^2 with s eta = u_r. The determinant of the 4 x 4
system vanishes at the complex growth rate s = -gamma + i omega.

Usage: drop_modes.py [R_cells] [cell_size_m]   (defaults: 20 cells of 1 mm)
"""
import cmath
import math
import sys


def bessel_i(n, z, steps=2000):
    """I_n(z) for integer n: (1/pi) * integral over [0, pi] of exp(z cos t) cos(n t)."""
    h = math.pi / steps
    total = 0.5 * (cmath.exp(z) + cmath.exp(-z) * math.cos(n * math.pi))
    for k in range(1, steps):
        t = k * h
        total += cmath.exp(z * math.cos(t)) * math.cos(n * t)
    return total * h / math.pi


def bessel_k(n, z, steps=4000, upper=12.0):
    """K_n(z) for Re z > 0: integral over [0, inf) of exp(-z cosh t) cosh(n t)."""
    h = upper / steps
    total = 0.5 * cmath.exp(-z)
    for k in range(1, steps + 1):
        t = k * h
        total += cmath.exp(-z * math.cosh(t)) * math.cosh(n * t)
    return total * h


def with_derivatives(f, sign, n, z):
    """(f, f', f'') of I_n (sign +1) or K_n (sign -1) at z."""
    v = f(n, z)
    d1 = sign * 0.5 * (f(n - 1, z) + f(n + 1, z))
    d2 = -d1 / z + (1 + n * n / (z * z)) * v
    return v, d1, d2


def det4(m):
    m = [row[:] for row in m]
    d = 1
    for c in range(4):
        piv = max(range(c, 4), key=lambda r: abs(m[r][c]))
        if piv != c:
            m[c], m[piv] = m[piv], m[c]
            d = -d
        d *= m[c][c]
        for r in range(c + 1, 4):
            f = m[r][c] / m[c][c]
            for k in range(c, 4):
                m[r][k] -= f * m[c][k]
    return d


def growth_rate(n, R, sigma, rho_i, rho_o, nu_i, nu_o):
    mu_i, mu_o = rho_i * nu_i, rho_o * nu_o

    def det(s):
        qi, qo = cmath.sqrt(s / nu_i), cmath.sqrt(s / nu_o)
        iv, i1, i2 = with_derivatives(bessel_i, 1, n, qi * R)
        kv, k1, k2 = with_derivatives(bessel_k, -1, n, qo * R)
        # psi, psi', psi'' of each basis function at r = R, normalised to psi = 1.
        f = [
            (1.0, n / R, n * (n - 1) / R**2),
            (1.0, qi * i1 / iv, qi**2 * i2 / iv),
            (1.0, -n / R, n * (n + 1) / R**2),
            (1.0, qo * k1 / kv, qo**2 * k2 / kv),
        ]
        side = [1, 1, -1, -1]            # inside minus outside
        mu = [mu_i, mu_i, mu_o, mu_o]
        # Pressure of the irrotational parts: inside -i rho_i s psi, outside +i rho_o s psi.
        p = [-1j * rho_i * s, 0.0, 1j * rho_o * s, 0.0]
        # Scale the rows to comparable magnitudes.
        m = [[0j] * 4 for _ in range(4)]
        for k, (psi, d1, d2) in enumerate(f):
            m[0][k] = side[k] * psi                                              # u_r
            m[1][k] = side[k] * d1 * R                                           # u_theta
            m[2][k] = side[k] * mu[k] * (-d2 + d1 / R - n**2 / R**2 * psi) * R**2 / mu_i   # shear stress
            durdr = 1j * n * (d1 / R - psi / R**2)
            m[3][k] = side[k] * (p[k] - 2 * mu[k] * durdr) / (rho_i * abs(s))    # normal stress
        # sigma (n^2 - 1) eta / R^2 with eta = u_r / s = i n psi / (R s); u_r is continuous: use the inside.
        cap = sigma * (n**2 - 1) / R**2 * 1j * n / (R * s) / (rho_i * abs(s))
        m[3][0] -= cap
        m[3][1] -= cap
        return det4(m)

    w0 = math.sqrt(n * (n**2 - 1) * sigma / ((rho_i + rho_o) * R**3))
    s0, s1 = complex(-0.05 * w0, 0.95 * w0), complex(-0.1 * w0, 0.9 * w0)
    d0, d1 = det(s0), det(s1)
    for _ in range(60):                        # secant iteration in the complex plane
        s2 = s1 - d1 * (s1 - s0) / (d1 - d0)
        s0, d0, s1, d1 = s1, d1, s2, det(s2)
        if abs(s1 - s0) < 1e-10 * abs(s1):
            break
    return s1, w0


if __name__ == "__main__":
    R_cells = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    dx = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    rho_w = 998.0
    cases = {
        "oil drop in water (defaults)": dict(sigma=0.025, rho_i=0.9 * rho_w, rho_o=rho_w, nu_i=6.0e-5, nu_o=1.0e-6),
        "both fluids with water's viscosity": dict(sigma=0.025, rho_i=0.9 * rho_w, rho_o=rho_w, nu_i=1.0e-6, nu_o=1.0e-6),
    }
    for name, c in cases.items():
        s, w0 = growth_rate(2, R_cells * dx, **c)
        print(f"{name}: R = {R_cells:g} cells of {dx:g} m")
        print(f"  inviscid period {2 * math.pi / w0:.4f} s")
        print(f"  viscous  period {2 * math.pi / s.imag:.4f} s ({(w0 / s.imag - 1) * 100:+.2f} %), decay rate {-s.real:.4f} /s,"
              f" amplitude ratio per period {math.exp(s.real * 2 * math.pi / s.imag):.4f}")
