"""Agent a07 -- circuit assembly.

For every kick sign (decided by a02, see ``kick_signs``) and time step t = 0..n_steps build
the circuit that is executed:

* t <= aqc_steps : the AQC-compressed circuit from a03 (already contains
  ground state + kick + t Trotter steps)
* t >  aqc_steps : last AQC circuit + (t - aqc_steps) exact Trotter steps ("tail")
* AQC disabled   : ground state (a02) + kick + t Trotter steps

Circuits are stored flat, sign-major: index = sign_index * (n_steps+1) + t.
Also records gate counts / depths (for the paper's resource table) and a
noiseless MPS simulation of the assembled circuits, which isolates the
algorithmic (VQE + AQC + Trotter) error from hardware noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..physics.circuits import circuit_stats, kick, trotter_circuit
from ..physics.dsf import rgf_from_z
from ..physics.mps import circuit_to_mps, z_profile
from .base import Agent, TaskContext, load_model, register_agent


@register_agent
class CircuitAssemblyAgent(Agent):
    name = "a07_assembly"
    title = "Circuit assembly (GS + AQC + Trotter tail)"
    fanout = "model"
    deps = ("a00_model_spec", "a02_gs_ansatz", "a03_aqc")

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        model = load_model(ctx)
        dyn = ctx.cfg["dynamics"]
        dt, order, n_steps = float(dyn["dt"]), int(dyn["trotter_order"]), int(dyn["n_steps"])
        gs = ctx.load_circuits(ctx.up("a02_gs_ansatz")["artifacts"]["ground_state_circuit"])[0]
        aqc_res = ctx.up("a03_aqc")
        aqc_all = ctx.load_circuits(aqc_res["artifacts"]["aqc_circuits"]) if aqc_res["summary"].get("enabled") else []
        signs = ctx.up("a02_gs_ansatz")["summary"]["kick_signs"]  # decided once, from the prepared state
        k = len(aqc_all) // len(signs)

        circuits, stats = [], []
        for si, sign in enumerate(signs):
            prefix = gs.compose(kick(model, sign))
            aqc = aqc_all[si * k:(si + 1) * k]
            for t in range(n_steps + 1):
                if t == 0:
                    qc, how = prefix, "gs+kick"
                elif t <= k:
                    qc, how = aqc[t - 1], f"aqc[{t}]"
                elif k:
                    qc, how = aqc[-1].compose(trotter_circuit(model, dt, t - k, order)), f"aqc[{k}]+trotter[{t - k}]"
                else:
                    qc, how = prefix.compose(trotter_circuit(model, dt, t, order)), f"gs+kick+trotter[{t}]"
                qc.name = f"{model.name}_s{'+' if sign > 0 else '-'}_t{t}"
                circuits.append(qc)
                stats.append(circuit_stats(qc) | {"sign": sign, "t": t, "plan": how})

        full_trotter = circuit_stats(gs.compose(kick(model)).compose(trotter_circuit(model, dt, n_steps, order)))
        ctx.save_circuits("circuits", circuits)
        ctx.save_json("circuit_stats", {"per_circuit": stats, "full_trotter_last_step": full_trotter,
                                        "signs": signs, "n_steps": n_steps, "order": "sign-major"})
        ctx.decision("assembly_plan", [s["plan"] for s in stats[: n_steps + 1]], kick_signs=signs,
                     reasons=f"{k} AQC steps, {n_steps - k} Trotter-tail steps, {len(signs)} kick sign(s)")

        out: dict[str, Any] = {"n_circuits": len(circuits), "kick_signs": signs, "aqc_steps": k,
                               "last_step_n_2q": stats[-1]["n_2q"], "last_step_depth_2q": stats[-1]["depth_2q"],
                               "full_trotter_n_2q": full_trotter["n_2q"]}
        if ctx.cfg["assembly"].get("noiseless_sim", True):
            mb = int(ctx.cfg["assembly"].get("max_bond", 128))
            z = np.array([z_profile(circuit_to_mps(qc, max_bond=mb)) for qc in circuits])
            z = z.reshape(len(signs), n_steps + 1, model.n_sites)
            ctx.save_npz("noiseless_rgf", rgf=rgf_from_z(z), z=z, dt=dt, center=model.center)
            out["noiseless_sim"] = True
        return out
