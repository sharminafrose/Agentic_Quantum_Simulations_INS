"""Agent a01 -- classical MPS/DMRG reference.

* DMRG ground state for every bond dimension in ``dmrg.chi_list`` (convergence test)
* exact-diagonalisation cross-check for small chains
* static correlations <S^a_c S^a_{c+r}> (ground truth for the two-tangle)
* reference kicked dynamics G(j, t) with a fine Trotter step on the
  converged DMRG state (the classical "reference RGF/DSF")
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..physics.circuits import kick, trotter_circuit
from ..physics.dsf import rgf_from_z
from ..physics.mps import dmrg, kicked_dynamics
from .base import Agent, TaskContext, load_model, register_agent


def _static_correlations(psi: Any, center: int, r_max: int) -> dict[str, list[float]]:
    """<S^a_i S^a_{i+r}> averaged over i in {center-1, center}.

    Averaging the two central bonds removes most of the open-boundary
    dimerisation (alternating strong/weak bonds) of finite chains.
    """
    import quimb as qu
    out: dict[str, list[float]] = {}
    for comp in ("x", "y", "z"):
        op = qu.spin_operator(comp, S=1 / 2)
        vals = [0.25]
        for r in range(1, r_max + 1):
            starts = [i for i in (center - 1, center) if 0 <= i and i + r < psi.L]
            vals.append(float(np.mean([np.real(psi.correlation(op, i, i + r)) for i in starts])))
        out[f"g_{comp}{comp}"] = vals
    return out


@register_agent
class MPSDMRGAgent(Agent):
    name = "a01_mps_dmrg"
    title = "Classical MPS/DMRG ground state + reference dynamics"
    fanout = "model"
    deps = ("a00_model_spec",)

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        model = load_model(ctx)
        dcfg, dyn = ctx.cfg["dmrg"], ctx.cfg["dynamics"]

        sweep, best_psi = [], None
        for chi in dcfg["chi_list"]:
            e, psi, info = dmrg(model, int(chi), float(dcfg["cutoff"]), int(dcfg["max_sweeps"]), float(dcfg["tol"]))
            info["energy_per_site"] = e / model.n_sites
            sweep.append(info)
            best_psi = psi
            ctx.metric("dmrg_energy", e, chi=chi, max_bond_reached=info["max_bond_reached"], wall_s=info["wall_s"])
        e_best = sweep[-1]["energy"]
        for row in sweep:
            row["delta_E_vs_max_chi"] = row["energy"] - e_best
        chi_conv = next((r["chi"] for r in sweep if abs(r["delta_E_vs_max_chi"]) < float(dcfg["conv_tol"])),
                        sweep[-1]["chi"])
        ctx.decision("converged_chi", chi_conv, reasons=f"|E(chi) - E(chi_max)| < {dcfg['conv_tol']}")
        ctx.save_json("chi_convergence", sweep)

        exact_e = None
        if model.n_sites <= 14:
            exact_e, _ = model.exact_ground_state()
            ctx.metric("dmrg_minus_exact", e_best - exact_e, exact_energy=exact_e)

        r_max = min(int(ctx.cfg["benchmark"]["r_max"]), model.n_sites - 1 - model.center)
        corr = _static_correlations(best_psi, model.center, r_max)
        ctx.save_json("static_correlations", corr)

        sub = int(dyn["reference_substeps"])
        step = trotter_circuit(model, float(dyn["dt"]) / sub, sub, order=int(dyn["trotter_order"]))
        signs = [1, -1]  # both signs: exact for any state, cheap classically
        runs = [kicked_dynamics(model, best_psi, step, kick(model, s), int(dyn["n_steps"]),
                                max_bond=int(dyn["ref_max_bond"])) for s in signs]
        z = np.array([r["z"] for r in runs])
        max_bond_used = max(r["max_bond_used"] for r in runs)
        ctx.save_npz("reference_rgf", rgf=rgf_from_z(z), z=z, signs=np.array(signs),
                     dt=float(dyn["dt"]), center=model.center)
        ctx.info("reference dynamics done", substeps=sub, kick_signs=signs, max_bond_used=max_bond_used)
        return {"ground_energy": e_best, "exact_energy": exact_e, "converged_chi": chi_conv,
                "chi_list": list(dcfg["chi_list"]), "reference_max_bond_used": max_bond_used,
                "kick_signs": signs}
