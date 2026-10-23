"""Agent a10 -- benchmarking / comparison.

For every model compares each reconstructed spectrum against

* the classical MPS reference (always), and
* experimental INS data, if ``<benchmark.experiment_dir>/<model>.npz`` exists
  with arrays ``q`` (rad), ``omega_meV`` and ``S`` (shape n_omega x n_q)

using the validated metric suite (a06): MSE, Wasserstein, SSIM on
max-normalised maps, plus nQFI(q = pi) and two-tangle on sum-rule-normalised
spectra. The DMRG static correlations give the "true" two-tangle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..physics import metrics as M
from ..physics.dsf import regrid
from ..rules import load_rule
from .base import Agent, TaskContext, load_model, register_agent


def _entanglement(s: np.ndarray, q: np.ndarray, w: np.ndarray, r_max: int, isotropic: bool) -> dict[str, float]:
    """nQFI always; two-tangle only for SU(2) models (only S^zz is measured)."""
    sn = M.sum_rule_normalize(s, q, w)
    fq = M.nqfi(sn, w)
    g = M.static_correlation(sn, q, w, r_max)
    return {"nqfi_q_pi": float(fq[np.argmin(np.abs(q - np.pi))]), "nqfi_max": float(fq.max()),
            "two_tangle": M.two_tangle(g)["two_tangle"] if isotropic else float("nan"), "g_zz_r1": float(g[1])}


_INTERPRET_SYSTEM = (
    "You interpret benchmark results of a quantum simulation of inelastic neutron scattering. "
    "Use ONLY the numbers in the table; never invent values. Follow the project rules below: state units and "
    "conventions, call AQC/HVA fidelities MPS-level, remember the simulation is T = 0, never quote absolute "
    "intensities, and do not use em-dashes. Lower mse and wasserstein and higher ssim mean closer to the reference. "
    "Reply ONLY with JSON: {\"summary\": str (3 sentences max), \"best_source\": str, \"worst_source\": str, "
    "\"caveats\": [str], \"suspicious\": [str]}.\n\nPROJECT RULES:\n"
)


def _interpret(ctx: TaskContext, model: Any, rows: list[dict[str, Any]], has_exp: bool) -> dict[str, Any] | None:
    """Advisory LLM write-up of the metrics table (None when the LLM is off or its answer is unusable)."""
    llm = ctx.llm()
    if llm is None or not ctx.cfg["llm"].get("interpret_results", True):
        return None
    keep = ("source", "reference", "mse", "wasserstein", "ssim", "nqfi_q_pi", "two_tangle")
    table = [{k: (round(r[k], 5) if isinstance(r.get(k), float) else r.get(k)) for k in keep if k in r} for r in rows]
    user = (f"Material: {model.material} ({model.name}), N = {model.n_sites} sites, J = {model.energy_scale_meV} meV. "
            f"Experimental INS data available: {has_exp}.\nMetrics (reference = MPS or INS):\n{table}")
    raw = llm.chat_json(_INTERPRET_SYSTEM + load_rule("dsf-assembly"), user)
    known = {r["source"] for r in rows}
    ok = (isinstance(raw, dict) and isinstance(raw.get("summary"), str)
          and raw.get("best_source") in known and raw.get("worst_source") in known)
    ctx.decision("llm_interpretation", "accepted" if ok else "rejected",
                 reasons="valid JSON naming existing sources" if ok else "missing/invalid answer or unknown source name")
    if not ok:
        return None
    ctx.save_json("interpretation", raw)
    return raw


@register_agent
class BenchmarkAgent(Agent):
    name = "a10_benchmark"
    title = "Benchmarking vs MPS reference and INS experiment"
    fanout = "model"
    deps = ("a00_model_spec", "a01_mps_dmrg", "a06_metrics", "a09_dsf")

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        model = load_model(ctx)
        bcfg = ctx.cfg["benchmark"]
        d = ctx.load_npz(ctx.up("a09_dsf")["artifacts"]["dsf"])
        q, w, w_meV = d["q"], d["omega"], d["omega_meV"]
        spectra = dict(zip([str(s) for s in d["sources"]], d["S"]))
        ref = spectra["mps_reference"]
        r_max = int(bcfg["r_max"])
        isotropic = model.jxy == model.jz and not model.has_field
        if not isotropic:
            ctx.decision("two_tangle_from_dsf", "skipped", reasons="model not SU(2): S^xx, S^yy not measured")

        corr = ctx.load_json(ctx.up("a01_mps_dmrg")["artifacts"]["static_correlations"])
        true_tangle = M.two_tangle(np.array(corr["g_zz"]), np.array(corr["g_xx"]), np.array(corr["g_yy"]))
        ctx.metric("two_tangle_dmrg_exact", true_tangle["two_tangle"])

        rows = []
        for name, s in spectra.items():
            row = {"model": model.name, "source": name, "reference": "mps_reference",
                   **M.compare_maps(s, ref, w), **_entanglement(s, q, w, r_max, isotropic)}
            rows.append(row)
            ctx.metric("vs_mps", row["mse"], source=name, wasserstein=row["wasserstein"], ssim=row["ssim"])

        exp_path = Path(bcfg["experiment_dir"]) / f"{model.name}.npz"
        has_exp = exp_path.exists()
        if has_exp:
            with np.load(exp_path) as e:
                eq, ew, es = e["q"], e["omega_meV"], e["S"]
            ctx.info("experimental INS data found", path=str(exp_path), shape=list(es.shape))
            for name, s in spectra.items():
                s_on_exp = regrid(q, w_meV, s, eq, ew)
                row = {"model": model.name, "source": name, "reference": "experiment_INS",
                       **M.compare_maps(s_on_exp, es, ew)}
                rows.append(row)
                ctx.metric("vs_experiment", row["mse"], source=name, wasserstein=row["wasserstein"], ssim=row["ssim"])
        else:
            ctx.warning("no experimental INS data; comparing to MPS reference only", expected_path=str(exp_path))

        ctx.save_json("metrics", rows)
        import pandas as pd
        pd.DataFrame(rows).to_csv(ctx.dir / "metrics.csv", index=False)
        ctx._register("metrics_csv", ctx.dir / "metrics.csv", format="csv")
        interpretation = _interpret(ctx, model, rows, has_exp)
        hw = [r for r in rows if r["source"].startswith("hardware") and r["reference"] == "mps_reference"]
        return {"n_rows": len(rows), "has_experiment": has_exp, "interpretation": interpretation,
                "two_tangle_dmrg": true_tangle["two_tangle"],
                "hardware_vs_mps": {k: hw[0][k] for k in ("mse", "wasserstein", "ssim")} if hw else None}
