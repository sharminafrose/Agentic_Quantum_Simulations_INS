import numpy as np

from qins.agents.a06_metrics import self_test, synthetic_dsf
from qins.physics import metrics as M


def test_self_test_passes():
    checks = self_test(np.random.default_rng(1))
    assert all(c["pass"] for c in checks), [c for c in checks if not c["pass"]]


def test_ssim_symmetric_and_bounded():
    q, w = np.linspace(0, 2 * np.pi, 40), np.linspace(0, 4, 50)
    a, b = synthetic_dsf(q, w), synthetic_dsf(q, w, shift=0.2)
    assert abs(M.ssim(a, b) - M.ssim(b, a)) < 1e-12
    assert -1 <= M.ssim(a, b) <= 1


def test_two_tangle_of_singlet_pair_correlations():
    # perfect nearest-neighbour singlet: g(1) = -1/4 in every component -> C = 1
    g = np.array([0.25, -0.25])
    assert abs(M.two_tangle(g)["concurrence"][0] - 1.0) < 1e-12


def test_wasserstein_ignores_weightless_columns():
    q, w = np.linspace(0, 2 * np.pi, 40), np.linspace(0, 4, 50)
    a = synthetic_dsf(q, w)
    b = a.copy()
    rng = np.random.default_rng(0)
    a[:, 0], b[:, 0] = rng.random(50) * 1e-16, rng.random(50) * 1e-16  # rounding noise at q = 0
    assert M.wasserstein(a, b, w) < 1e-12
