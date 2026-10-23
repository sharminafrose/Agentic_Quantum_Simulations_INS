"""Agent a02 -- variational ground-state preparation (VQE with a HVA).

* optimises the Hamiltonian variational ansatz by minimising <H> (VQE) for
  each initial state branch (``singlet`` / ``neel``) and keeps the best one
* optional sweep over system sizes x layers (Fig. S2): fidelity with the
  exact ground state for small chains
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..physics.circuits import circuit_stats, hva_ansatz, kick_signs, n_hva_params
from ..physics.models import SpinModel
from ..physics.mps import circuit_to_mps, decompose, energy, fidelity_with_vector, z_profile
from .base import Agent, TaskContext, load_model, register_agent


def _initial_params(model: SpinModel, layers: int, rng: np.random.Generator) -> np.ndarray:
    per = 3 if model.has_field else 2
    base = np.tile([0.0, np.pi / 2, 0.0][:per], layers)
    return base + rng.normal(scale=0.1, size=n_hva_params(model, layers))


def optimise_hva(model: SpinModel, layers: int, init: str, opt_cfg: dict[str, Any],
                 rng: np.random.Generator) -> dict[str, Any]:
    """VQE: minimise <H> over HVA angles; returns params, energy, history and bound circuit."""
    ansatz = decompose(hva_ansatz(model, layers, init))
    params = list(ansatz.parameters)
    history: list[float] = []

    def cost(x: np.ndarray) -> float:
        e = energy(ansatz.assign_parameters(dict(zip(params, x))), model, int(opt_cfg.get("max_bond", 64)))
        history.append(e)
        return e

    t0 = time.time()
    res = minimize(cost, _initial_params(model, layers, rng), method=opt_cfg.get("optimizer", "COBYQA"),
                   options={"maxiter": int(opt_cfg["maxiter"])})
    bound = ansatz.assign_parameters(dict(zip(params, res.x)))
    return {"init": init, "layers": layers, "energy": float(res.fun), "params": res.x.tolist(),
            "n_evals": len(history), "success": bool(res.success), "message": str(res.message),
            "wall_s": round(time.time() - t0, 3), "history": history, "circuit": bound}


@register_agent
class GroundStateAnsatzAgent(Agent):
    name = "a02_gs_ansatz"
    title = "Ground-state ansatz (VQE / HVA)"
    fanout = "model"
    deps = ("a00_model_spec",)

    def _sweep(self, ctx: TaskContext, model: SpinModel, acfg: dict[str, Any]) -> list[dict[str, Any]]:
        from dataclasses import replace
        rows = []
        for n in acfg["sweep"]["sizes"]:
            small = replace(model, n_sites=int(n))
            e_exact, vec = small.exact_ground_state()
            for layers in acfg["sweep"]["layers"]:
                for init in acfg["inits"]:
                    r = optimise_hva(small, int(layers), init, acfg, ctx.rng)
                    fid = fidelity_with_vector(r["circuit"], vec)
                    rows.append({"n_sites": int(n), "layers": int(layers), "init": init, "energy": r["energy"],
                                 "exact_energy": e_exact, "fidelity": fid, "n_evals": r["n_evals"]})
                    ctx.metric("sweep_fidelity", fid, n_sites=n, layers=layers, init=init)
        return rows

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        model = load_model(ctx)
        acfg = ctx.cfg["ansatz"]
        layers = int(acfg["layers"])

        branches = []
        for init in acfg["inits"]:
            r = optimise_hva(model, layers, init, acfg, ctx.rng)
            ctx.metric("vqe_energy", r["energy"], init=init, layers=layers, n_evals=r["n_evals"])
            branches.append(r)
        best = min(branches, key=lambda r: r["energy"])
        ctx.decision("initial_state_branch", best["init"],
                     reasons={b["init"]: b["energy"] for b in branches}, criterion="lowest VQE energy")

        fid_exact, e_exact = None, None
        if model.n_sites <= 14:
            e_exact, vec = model.exact_ground_state()
            fid_exact = fidelity_with_vector(best["circuit"], vec)
            ctx.metric("fidelity_vs_exact", fid_exact)

        max_abs_z = float(np.max(np.abs(z_profile(circuit_to_mps(best["circuit"], max_bond=int(acfg.get("max_bond", 64)))))))
        signs = kick_signs(model, max_abs_z, float(ctx.cfg["dynamics"].get("symmetry_tol", 1e-3)))
        ctx.decision("kick_signs", signs, reasons={"max_abs_z": max_abs_z, "has_field": model.has_field},
                     criterion="two signs unless the state is spin-flip symmetric (<Z_j> = 0)")
        ctx.save_circuits("ground_state_circuit", [best["circuit"]])
        ctx.save_json("branches", [{k: v for k, v in b.items() if k != "circuit"} for b in branches])
        if acfg["sweep"].get("enabled", False):
            ctx.save_json("fidelity_sweep", self._sweep(ctx, model, acfg))
        stats = circuit_stats(best["circuit"])
        return {"best_init": best["init"], "layers": layers, "energy": best["energy"], "exact_energy": e_exact,
                "fidelity_vs_exact": fid_exact, "max_abs_z": max_abs_z, "kick_signs": signs, "optimizer": acfg.get("optimizer", "COBYQA"),
                "circuit_stats": stats}
