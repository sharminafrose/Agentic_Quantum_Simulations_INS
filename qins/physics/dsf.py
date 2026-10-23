"""Retarded Green's function G(j, t)  ->  dynamical structure factor S(q, w).

Same estimator as the IBM tutorial (``get_dsf``), with optional time window
and Gaussian energy broadening (to mimic the INS instrument resolution).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator

__all__ = ["rgf_from_z", "rgf_to_dsf", "regrid", "normalize"]


def rgf_from_z(z: np.ndarray) -> np.ndarray:
    """Kicked <Z_j(t)> -> G(j, t) for t = 1..T.

    ``z`` has shape (n_signs, T+1, n): one kick sign (+pi/2) or two (+pi/2, -pi/2);
    with two signs G = (z+ - z-)/2, which cancels the non-retarded part exactly.
    """
    z = np.asarray(z, dtype=float)
    g = z[0] if z.shape[0] == 1 else (z[0] - z[1]) / 2
    return g[1:]


def _window(n_t: int, kind: str) -> np.ndarray:
    if kind in (None, "none"):
        return np.ones(n_t)
    if kind == "hann":  # half Hann: 1 at t=0, 0 at t_max
        return 0.5 * (1 + np.cos(np.pi * np.arange(1, n_t + 1) / (n_t + 1)))
    if kind == "gaussian":
        t = np.arange(1, n_t + 1) / n_t
        return np.exp(-0.5 * (t / 0.5) ** 2)
    raise ValueError(f"unknown window {kind!r}")


def rgf_to_dsf(rgf: np.ndarray, dt: float, center: int, n_omega: int = 100, n_q: int = 100,
               window: str = "none", broadening: float = 0.0) -> dict[str, Any]:
    """Fourier transform in time and space (centre-site approximation).

    Args:
        rgf: G(j, t) with shape (n_t, n_sites) for t = dt, 2 dt, ..., n_t dt.
        dt: Trotter time step (units of 1/J).
        center: kicked site index.
        broadening: Gaussian sigma in units of J applied along omega.
    """
    n_t, n_sites = rgf.shape
    times = dt * np.arange(1, n_t + 1)
    omegas = np.linspace(0.0, np.pi / dt, n_omega)
    qs = np.linspace(0.0, 2 * np.pi, n_q)
    g = rgf * _window(n_t, window)[:, None]
    g_w = (np.exp(1j * np.outer(omegas, times)) @ g) * dt / 4          # (n_omega, n_sites)
    m = np.arange(n_sites) - center
    phase = np.exp(-1j * np.outer(m, qs))                               # (n_sites, n_q)
    s = np.imag(g_w @ phase)
    s = -(s + s[:, ::-1]) / 2
    if broadening > 0:
        s = gaussian_filter1d(s, sigma=broadening / (omegas[1] - omegas[0]), axis=0, mode="nearest")
    s = np.clip(s, 0.0, None)
    return {"q": qs, "omega": omegas, "S": s}


def regrid(q_src: np.ndarray, w_src: np.ndarray, s_src: np.ndarray,
           q_dst: np.ndarray, w_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation of S(w, q) onto another grid (zero outside)."""
    interp = RegularGridInterpolator((w_src, q_src), s_src, bounds_error=False, fill_value=0.0)
    ww, qq = np.meshgrid(w_dst, q_dst, indexing="ij")
    return interp(np.stack([ww.ravel(), qq.ravel()], axis=1)).reshape(ww.shape)


def normalize(s: np.ndarray, how: str = "max") -> np.ndarray:
    if how == "max":
        return s / s.max() if s.max() > 0 else s
    if how == "sum":
        return s / s.sum() if s.sum() > 0 else s
    return s
