"""Agent a03 -- approximate quantum compilation (qiskit-addon-aqc-tensor).

For each kick sign and early time step t = 1..aqc_steps the deep target circuit
``ground state + kick + t Trotter steps`` is compressed into a fixed, shallow
ansatz by maximising the MPS state fidelity (L-BFGS-B with JAX gradients,
warm-started from the previous step) -- as in the IBM tutorial.

Note: AQC compresses *states*, so it needs the ground-state circuit from
agent a02; it runs per model, so different models still compile in parallel.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..physics.circuits import circuit_stats, kick, trotter_circuit
from ..physics.mps import decompose
from .base import Agent, TaskContext, load_model, register_agent


@register_agent
class AQCAgent(Agent):
    name = "a03_aqc"
    title = "Approximate quantum compilation (tensor network)"
    fanout = "model"
    deps = ("a00_model_spec", "a02_gs_ansatz")

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        acfg, dyn = ctx.cfg["aqc"], ctx.cfg["dynamics"]
        if not acfg.get("enabled", True) or int(acfg["aqc_steps"]) == 0:
            ctx.decision("aqc", "disabled", reasons="aqc.enabled=false or aqc_steps=0")
            return {"enabled": False}

        import quimb.tensor as qtn
        from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
        from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
        from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
        from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator

        model = load_model(ctx)
        gs = ctx.load_circuits(ctx.up("a02_gs_ansatz")["artifacts"]["ground_state_circuit"])[0]
        dt, order = float(dyn["dt"]), int(dyn["trotter_order"])
        sim = QuimbSimulator(partial(qtn.CircuitMPS, gate_opts={"max_bond": int(acfg["max_bond"]),
                                                                "cutoff": float(acfg["cutoff"])}),
                             autodiff_backend=acfg.get("autodiff_backend", "jax"))

        signs = ctx.up("a02_gs_ansatz")["summary"]["kick_signs"]
        circuits, rows, n_params = [], [], 0
        for sign in signs:
            prefix = gs.compose(kick(model, sign))
            template = decompose(prefix.compose(trotter_circuit(model, dt, int(acfg["ansatz_steps"]), order)))
            ansatz, x0 = generate_ansatz_from_circuit(template, qubits_initially_zero=True)
            x, n_params = np.asarray(x0, dtype=float), len(x0)
            ctx.info("AQC ansatz generated", sign=sign, n_params=n_params, ansatz_depth=ansatz.depth())
            for t in range(1, int(acfg["aqc_steps"]) + 1):
                target_qc = decompose(prefix.compose(trotter_circuit(model, dt, t, order)))
                target = tensornetwork_from_circuit(target_qc, sim)
                objective = MaximizeStateFidelity(target, ansatz, sim)
                t0 = time.time()
                res = minimize(objective.loss_function, x, method="L-BFGS-B", jac=True,  # warm start
                               options={"maxiter": int(acfg["maxiter"])})
                x = res.x
                fidelity = float(1 - res.fun)
                circuits.append(ansatz.assign_parameters(x))
                rows.append({"sign": sign, "step": t, "fidelity": fidelity, "n_iter": int(res.nit),
                             "wall_s": round(time.time() - t0, 3), "target_stats": circuit_stats(target_qc),
                             "message": str(res.message), "params": x.tolist()})
                ctx.metric("aqc_fidelity", fidelity, sign=sign, step=t, n_iter=int(res.nit))

        ctx.save_circuits("aqc_circuits", circuits)
        ctx.save_json("aqc_steps", rows)
        return {"enabled": True, "aqc_steps": int(acfg["aqc_steps"]), "kick_signs": signs,
                "ansatz_steps": int(acfg["ansatz_steps"]), "n_params": n_params,
                "fidelities": [r["fidelity"] for r in rows],
                "aqc_stats": circuit_stats(circuits[-1]), "target_stats_last": rows[-1]["target_stats"]}
