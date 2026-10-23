"""Agent a08 -- quantum hardware execution with dynamic error mitigation.

1. pick the backend + qubit chain (best calibration score from a04, or fixed)
2. extract circuit features and *select* the mitigation stack (DD / Pauli
   twirling / TREX / ZNE) with the rule policy or the vLLM advisor
3. execute in one of three modes:
   * ``local_emulator`` -- Aer MPS with the chosen chain's calibrated noise;
     TREX is emulated by readout-matrix inversion; DD/PT/ZNE are recorded but
     not emulated (they need the real device)
   * ``ibm``            -- EstimatorV2 on the real backend with the chosen options
   * ``dry_run``        -- transpile + select only (no data)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..mitigation import circuit_features, estimator_options, select_mitigation
from ..physics.dsf import rgf_from_z
from ..physics.hardware import Calibration
from ..physics.noise import calibrated_noise_model, readout_correct, run_z, to_line_isa
from .base import Agent, TaskContext, load_model, register_agent


@register_agent
class HardwareExecutionAgent(Agent):
    name = "a08_hw_execution"
    title = "Quantum hardware execution (PT / DD / TREX / ZNE)"
    fanout = "model"
    deps = ("a00_model_spec", "a07_assembly", "a04_hw_calibration")

    def _choose_backend(self, ctx: TaskContext, n: int) -> tuple[str, dict[str, Any]]:
        cands = {b: r["summary"]["layouts"][str(n)] for b, r in ctx.up_all("a04_hw_calibration").items()}
        wanted = ctx.cfg["execution"]["backend"]
        if wanted != "auto" and wanted not in cands:
            raise ValueError(f"execution.backend={wanted!r} is not in hardware.backends {sorted(cands)}")
        name = max(cands, key=lambda b: cands[b]["log_fidelity"]) if wanted == "auto" else wanted
        ctx.decision("backend", name, reasons={b: round(c["est_layer_fidelity"], 5) for b, c in cands.items()},
                     criterion="highest estimated chain layer fidelity" if wanted == "auto" else "fixed by config")
        return name, cands[name]

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        model = load_model(ctx)
        ecfg = ctx.cfg["execution"]
        backend_name, layout = self._choose_backend(ctx, model.n_sites)
        a04 = ctx.up("a04_hw_calibration", backend_name)
        cal = Calibration(**ctx.load_json(a04["artifacts"]["calibration"]))
        chain = layout["chain"]
        layouts = ctx.load_json(a04["artifacts"]["layouts"])
        chain_score = layouts[str(model.n_sites)]["score"]

        arts = ctx.up("a07_assembly")["artifacts"]
        circuits = ctx.load_circuits(arts["circuits"])
        n_signs = len(ctx.up("a07_assembly")["summary"]["kick_signs"])
        isa = to_line_isa(circuits, seed=ctx.seed, optimization_level=int(ecfg.get("optimization_level", 2)))

        features = circuit_features(isa, chain_score)
        sel = select_mitigation(features, ecfg["mitigation"], ctx.llm())
        ctx.decision("mitigation", sel["choice"], reasons=sel["reasons"], policy=sel["policy_used"],
                     features=features, rule_choice=sel["rule_choice"], llm_raw=sel["llm_raw"])
        ctx.save_json("mitigation", {"features": features, **sel})

        mode = ecfg["mode"]
        meta: dict[str, Any] = {"mode": mode, "backend": backend_name, "chain": chain, "shots": int(ecfg["shots"])}
        if mode == "dry_run":
            ctx.save_json("execution", meta)
            return {**meta, "executed": False, "mitigation": sel["choice"]}
        if mode == "local_emulator":
            z, extra = self._emulate(ctx, cal, chain, isa, sel["choice"])
        elif mode == "ibm":
            z, extra = self._ibm(ctx, model, backend_name, chain, circuits, sel["choice"])
        else:
            raise ValueError(f"unknown execution.mode {mode!r}")
        meta.update(extra)
        z = z.reshape(n_signs, -1, model.n_sites)
        ctx.save_npz("hardware_rgf", rgf=rgf_from_z(z), z=z, dt=float(ctx.cfg["dynamics"]["dt"]), center=model.center)
        ctx.save_json("execution", meta)
        return {**meta, "executed": True, "mitigation": sel["choice"], "policy": sel["policy_used"]}

    def _emulate(self, ctx: TaskContext, cal: Calibration, chain: list[int], isa: list[Any],
                 choice: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        nm = calibrated_noise_model(cal, chain, include_readout=True)
        z = run_z(isa, nm, int(ctx.cfg["execution"]["shots"]), ctx.seed)
        applied = []
        if choice["trex"]:
            p01 = np.array([cal.qubits[q].get("p01") or cal.qubits[q].get("readout_error") or 0.0 for q in chain])
            p10 = np.array([cal.qubits[q].get("p10") or cal.qubits[q].get("readout_error") or 0.0 for q in chain])
            z = readout_correct(z, p01, p10)
            applied.append("readout_inversion(TREX-emulated)")
        not_emulated = [k for k in ("dd", "twirling", "zne") if choice[k]]
        if not_emulated:
            ctx.warning("mitigation selected but not emulated locally", techniques=not_emulated)
        return z, {"applied_mitigation": applied, "not_emulated": not_emulated}

    def _ibm(self, ctx: TaskContext, model: Any, backend_name: str, chain: list[int], circuits: list[Any],
             choice: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

        ecfg = ctx.cfg["execution"]
        service = QiskitRuntimeService(instance=ecfg.get("ibm", {}).get("instance"))
        backend = service.backend(backend_name)
        cmap = set(map(tuple, backend.coupling_map.get_edges()))
        missing = [(a, b) for a, b in zip(chain, chain[1:]) if (a, b) not in cmap and (b, a) not in cmap]
        if missing:  # e.g. layout found on fake/stale calibration data
            ctx.warning("chain edges missing on live device; transpiler will insert SWAPs", missing=missing)
        pm = generate_preset_pass_manager(optimization_level=int(ecfg.get("optimization_level", 2)),
                                          backend=backend, initial_layout=chain, seed_transpiler=ctx.seed)
        from ..physics.qlock import TRANSPILE_LOCK
        with TRANSPILE_LOCK:
            isa = pm.run(circuits)
        n = model.n_sites
        obs = [SparsePauliOp.from_sparse_list([("Z", [j], 1.0)], n) for j in range(n)]
        pubs = [(c, [o.apply_layout(c.layout) for o in obs]) for c in isa]
        options = estimator_options(choice, ecfg)
        estimator = EstimatorV2(mode=backend, options=options)
        estimator.options.environment.job_tags = list(ecfg.get("ibm", {}).get("job_tags", ["qins"]))
        job = estimator.run(pubs)
        ctx.info("submitted IBM job", job_id=job.job_id(), backend=backend_name, n_pubs=len(pubs))
        result = job.result()
        z = np.array([np.asarray(r.data.evs, dtype=float) for r in result])
        return z, {"job_id": job.job_id(), "estimator_options": options,
                             "stds": np.array([np.asarray(r.data.stds) for r in result]).tolist()}
