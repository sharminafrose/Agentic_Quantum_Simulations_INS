"""Spectral-comparison and entanglement metrics.

All map metrics take two arrays ``A, B`` of shape (n_omega, n_q) on the same
grid. They are unit-tested against synthetic spectra (agent a06, tests/).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance

__all__ = ["mse", "wasserstein", "ssim", "nqfi", "sum_rule_normalize", "static_correlation",
           "concurrence", "two_tangle", "compare_maps"]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def wasserstein(a: np.ndarray, b: np.ndarray, omega: np.ndarray, rel_weight: float = 1e-3) -> float:
    """Mean over q of the 1D earth-mover distance between energy line-cuts.

    Line-cuts are normalised to unit weight, so q-columns that carry
    (numerically) no spectral weight -- e.g. q ~ 0 where S ~ 1e-16 -- would
    turn rounding noise into arbitrary distributions. Only columns holding at
    least ``rel_weight`` of the heaviest column in *both* maps are averaged.
    """
    wa, wb = a.sum(axis=0), b.sum(axis=0)
    keep = (wa > rel_weight * wa.max()) & (wb > rel_weight * wb.max()) if wa.max() > 0 and wb.max() > 0 else []
    vals = []
    for k in np.flatnonzero(keep):
        vals.append(wasserstein_distance(omega, omega, a[:, k], b[:, k]))
    return float(np.mean(vals)) if vals else float("nan")


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    """Structural similarity index (Wang et al. 2004) with a Gaussian window."""
    data_range = max(a.max(), b.max()) - min(a.min(), b.min()) or 1.0
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    var_a = gaussian_filter(a * a, sigma) - mu_a ** 2
    var_b = gaussian_filter(b * b, sigma) - mu_b ** 2
    cov = gaussian_filter(a * b, sigma) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def sum_rule_normalize(s: np.ndarray, q: np.ndarray, omega: np.ndarray, total: float = 0.25) -> np.ndarray:
    """Scale S so that  (1/2pi) int dq int dw S^{zz}(q, w) = <(S^z)^2> = 1/4."""
    integral = np.trapezoid(np.trapezoid(s, omega, axis=0), q) / (2 * np.pi)
    return s * (total / integral) if integral > 0 else s


def nqfi(s: np.ndarray, omega: np.ndarray, temperature: float = 0.0) -> np.ndarray:
    """Normalised quantum Fisher information per q (Hauke et al., Nat. Phys. 2016).

    f_Q(q) = 4 int_0^inf dw tanh(w / 2T) (1 - exp(-w/T)) S(q, w); for T = 0 the
    thermal factors are 1. Use on sum-rule normalised spectra.
    """
    if temperature > 0:
        w = np.maximum(omega, 1e-12)
        factor = np.tanh(w / (2 * temperature)) * (1 - np.exp(-w / temperature))
    else:
        factor = np.ones_like(omega)
    return 4 * np.trapezoid(factor[:, None] * s, omega, axis=0)


def static_correlation(s: np.ndarray, q: np.ndarray, omega: np.ndarray, r_max: int = 6) -> np.ndarray:
    """g(r) = <S^z_0 S^z_r> = (1/2pi) int dq cos(q r) int dw S(q, w), r = 0..r_max."""
    sq = np.trapezoid(s, omega, axis=0)
    return np.array([np.trapezoid(np.cos(q * r) * sq, q) / (2 * np.pi) for r in range(r_max + 1)])


def concurrence(gxx: float, gyy: float, gzz: float) -> float:
    """Pairwise concurrence of an X-state with zero magnetisation (spin units)."""
    c = max(0.0, abs(gxx + gyy) - abs(0.25 + gzz), abs(gxx - gyy) - abs(0.25 - gzz))
    return 2 * c


def two_tangle(g: np.ndarray, gxx: np.ndarray | None = None, gyy: np.ndarray | None = None) -> dict[str, Any]:
    """tau_2 = sum_{j != i} C_ij^2 = 2 sum_{r >= 1} C_r^2.

    Only ``g_zz`` is measured; transverse correlations default to ``g_zz``
    (SU(2) symmetry, exact for KCuF3; an approximation for anisotropic chains).
    """
    gxx = g if gxx is None else gxx
    gyy = g if gyy is None else gyy
    c = np.array([concurrence(gxx[r], gyy[r], g[r]) for r in range(1, len(g))])
    return {"concurrence": c, "two_tangle": float(2 * np.sum(c ** 2)), "assumed_isotropic": True}


def compare_maps(a: np.ndarray, b: np.ndarray, omega: np.ndarray) -> dict[str, float]:
    """All three map metrics on max-normalised spectra."""
    an = a / a.max() if a.max() > 0 else a
    bn = b / b.max() if b.max() > 0 else b
    return {"mse": mse(an, bn), "wasserstein": wasserstein(an, bn, omega), "ssim": ssim(an, bn)}
