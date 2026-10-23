"""Agent a09 -- DSF reconstruction.

Fourier-transforms every available Green's function of a model with the same
estimator, so all spectra are directly comparable:

* ``mps_reference``     (a01, fine-step DMRG + TEBD-like evolution)
* ``noiseless_circuit`` (a07, assembled circuits without noise)
* ``noisy_p=<p>``       (a05, depolarising sweep)
* ``hardware``          (a08, emulator or real device)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..physics.dsf import rgf_to_dsf
from .base import Agent, TaskContext, load_model, register_agent


def plot_dsf_grid(spectra: dict[str, np.ndarray], q: np.ndarray, w_meV: np.ndarray, title: str) -> Any:
    from matplotlib.figure import Figure  # no pyplot: thread-safe
    n = len(spectra)
    fig = Figure(figsize=(3.2 * n, 3.0))
    axes = fig.subplots(1, n, squeeze=False)
    for ax, (name, s) in zip(axes[0], spectra.items()):
        ax.pcolormesh(q / np.pi, w_meV, s / max(s.max(), 1e-12), shading="auto", cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("q / pi")
    axes[0][0].set_ylabel("energy (meV)")
    fig.suptitle(title)
    return fig


@register_agent
class DSFAgent(Agent):
    name = "a09_dsf"
    title = "DSF reconstruction (Fourier transform of RGF)"
    fanout = "model"
    deps = ("a00_model_spec", "a01_mps_dmrg", "a05_noise_model", "a07_assembly", "a08_hw_execution")

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        model = load_model(ctx)
        dcfg, dt = ctx.cfg["dsf"], float(ctx.cfg["dynamics"]["dt"])
        broadening = float(dcfg.get("broadening_meV", 0.0)) / model.energy_scale_meV

        rgfs: dict[str, np.ndarray] = {"mps_reference": ctx.load_npz(ctx.up("a01_mps_dmrg")["artifacts"]["reference_rgf"])["rgf"]}
        a07 = ctx.up("a07_assembly")["artifacts"]
        if "noiseless_rgf" in a07:
            rgfs["noiseless_circuit"] = ctx.load_npz(a07["noiseless_rgf"])["rgf"]
        noisy = ctx.load_npz(ctx.up("a05_noise_model")["artifacts"]["noisy_rgf"])
        for p, rgf in zip(noisy["rates"], noisy["rgf"]):
            rgfs[f"noisy_p={p:g}"] = rgf
        hw = ctx.up("a08_hw_execution")
        if "hardware_rgf" in hw["artifacts"]:
            rgfs[f"hardware({hw['summary']['backend']})"] = ctx.load_npz(hw["artifacts"]["hardware_rgf"])["rgf"]

        ref_rms = float(np.sqrt(np.mean(rgfs["mps_reference"] ** 2)))
        diagnostics = {}
        for name, rgf in rgfs.items():  # signal size vs. deviation from the reference (SNR-style check)
            diagnostics[name] = {"rgf_rms": float(np.sqrt(np.mean(rgf ** 2))),
                                 "rms_dev_from_reference": float(np.sqrt(np.mean((rgf - rgfs["mps_reference"]) ** 2))),
                                 "reference_rms": ref_rms}
            ctx.metric("rgf_signal", diagnostics[name]["rgf_rms"], source=name,
                       rms_dev_from_reference=diagnostics[name]["rms_dev_from_reference"])
        ctx.save_json("rgf_diagnostics", diagnostics)

        spectra, q, w = {}, None, None
        for name, rgf in rgfs.items():
            out = rgf_to_dsf(rgf, dt, model.center, int(dcfg["n_omega"]), int(dcfg["n_q"]),
                             dcfg.get("window", "none"), broadening)
            spectra[name], q, w = out["S"], out["q"], out["omega"]
        w_meV = w * model.energy_scale_meV
        ctx.save_npz("dsf", q=q, omega=w, omega_meV=w_meV, sources=np.array(list(spectra)),
                     S=np.array(list(spectra.values())))
        ctx.save_figure("dsf_panels", plot_dsf_grid(spectra, q, w_meV, f"{model.material} (N={model.n_sites})"))
        ctx.decision("dsf_estimator", "centre-site FT", reasons={"window": dcfg.get("window", "none"),
                                                                 "broadening_meV": dcfg.get("broadening_meV", 0.0)})
        return {"sources": list(spectra), "n_q": len(q), "n_omega": len(w), "omega_max_meV": float(w_meV[-1])}
