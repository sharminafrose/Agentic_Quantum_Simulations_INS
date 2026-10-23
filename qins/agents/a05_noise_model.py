"""Agent a05 -- depolarising-noise MPS simulations (error-rate sweep).

Runs the *same* assembled circuits as the hardware (from a07), transpiled to
the Heron basis on a line, with a uniform depolarising channel of strength p
on every CZ (and ``one_qubit_ratio * p`` on 1q gates), for each p in
``noise.depolarizing_rates`` (e.g. 7.5e-3 -> 1e-3). Uses Aer's MPS method, so
it runs in parallel with -- and independently of -- the hardware job (a08).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..physics.dsf import rgf_from_z
from ..physics.noise import depolarizing_noise_model, run_z, to_line_isa
from .base import Agent, TaskContext, load_model, register_agent


@register_agent
class NoiseModelAgent(Agent):
    name = "a05_noise_model"
    title = "Depolarising-channel MPS simulation sweep"
    fanout = "model"
    deps = ("a00_model_spec", "a07_assembly")

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        model = load_model(ctx)
        ncfg = ctx.cfg["noise"]
        isa = to_line_isa(ctx.load_circuits(ctx.up("a07_assembly")["artifacts"]["circuits"]), seed=ctx.seed)
        shots, method = int(ncfg["shots"]), ncfg.get("method", "matrix_product_state")
        n_signs = len(ctx.up("a07_assembly")["summary"]["kick_signs"])

        rates = [float(p) for p in ncfg["depolarizing_rates"]]
        rgfs, zs = [], []
        for p in rates:
            nm = depolarizing_noise_model(p, float(ncfg["one_qubit_ratio"]))
            z = run_z(isa, nm, shots, ctx.seed, method).reshape(n_signs, -1, model.n_sites)
            rgfs.append(rgf_from_z(z))
            zs.append(z)
            ctx.metric("noisy_center_signal", float(np.abs(rgfs[-1][:, model.center]).mean()), p=p)
        ctx.save_npz("noisy_rgf", rgf=np.array(rgfs), z=np.array(zs), rates=np.array(rates),
                     dt=float(ctx.cfg["dynamics"]["dt"]), center=model.center)
        return {"rates": rates, "shots": shots, "n_circuits": len(isa),
                "max_n_2q": int(max(c.count_ops().get("cz", 0) for c in isa))}
