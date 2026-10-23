"""Agent a06 -- metric suite validation on synthetic spectra.

Runs before any real data exists: builds dummy dispersive spectra and checks
that MSE / Wasserstein / SSIM / nQFI / concurrence behave as expected. If any
check fails the task fails, and benchmarking (a10) is skipped -- so no paper
number is ever produced by an untested metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..physics import metrics as M
from .base import Agent, TaskContext, register_agent


def synthetic_dsf(q: np.ndarray, w: np.ndarray, shift: float = 0.0, width: float = 0.1) -> np.ndarray:
    """Gaussian ridge along the des Cloizeaux-Pearson dispersion (pi/2)|sin q|."""
    disp = np.pi / 2 * np.abs(np.sin(q)) + shift
    return np.exp(-0.5 * ((w[:, None] - disp[None, :]) / width) ** 2)


def self_test(rng: np.random.Generator) -> list[dict[str, Any]]:
    q, w = np.linspace(0, 2 * np.pi, 80), np.linspace(0, 4, 120)
    a = synthetic_dsf(q, w)
    shifted = synthetic_dsf(q, w, shift=0.3)
    noisy = np.clip(a + rng.normal(scale=0.05, size=a.shape), 0, None)
    checks = []

    def add(name: str, value: float, ok: bool, expect: str) -> None:
        checks.append({"check": name, "value": float(value), "pass": bool(ok), "expect": expect})

    add("mse_identical", M.mse(a, a), M.mse(a, a) == 0, "0")
    add("ssim_identical", M.ssim(a, a), abs(M.ssim(a, a) - 1) < 1e-9, "1")
    add("wasserstein_identical", M.wasserstein(a, a, w), M.wasserstein(a, a, w) < 1e-9, "0")
    wd = M.wasserstein(a, shifted, w)
    add("wasserstein_shift_0.3", wd, abs(wd - 0.3) < 0.03, "~0.3 (energy shift)")
    add("ssim_noisy_lt_1", M.ssim(a, noisy), M.ssim(a, noisy) < 0.99, "< 1")
    add("mse_order", M.mse(a, shifted) - M.mse(a, noisy), M.mse(a, shifted) > M.mse(a, noisy), "shift worse than noise")
    s_norm = M.sum_rule_normalize(a, q, w)
    total = np.trapezoid(np.trapezoid(s_norm, w, axis=0), q) / (2 * np.pi)
    add("sum_rule", total, abs(total - 0.25) < 1e-9, "0.25")
    fq = M.nqfi(s_norm, w)
    add("nqfi_positive", fq.min(), fq.min() >= 0, ">= 0")
    c_heis = M.concurrence(-0.1477, -0.1477, -0.1477)
    add("concurrence_heisenberg_nn", c_heis, abs(c_heis - 0.386) < 0.01, "~0.386 (Bethe-ansatz NN correlations)")
    add("concurrence_product", M.concurrence(0.0, 0.0, 0.0), M.concurrence(0.0, 0.0, 0.0) == 0, "0")
    return checks


@register_agent
class MetricsAgent(Agent):
    name = "a06_metrics"
    title = "Metric suite (MSE, Wasserstein, SSIM, nQFI, two-tangle) self-test"
    fanout = None
    deps = ()

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        checks = self_test(ctx.rng)
        ctx.save_json("self_test", checks)
        failed = [c["check"] for c in checks if not c["pass"]]
        for c in checks:
            ctx.metric(c["check"], c["value"], passed=c["pass"])
        if failed:
            raise AssertionError(f"metric self-test failed: {failed}")
        return {"n_checks": len(checks), "all_passed": True,
                "metrics": ["mse", "wasserstein", "ssim", "nqfi", "two_tangle"]}
