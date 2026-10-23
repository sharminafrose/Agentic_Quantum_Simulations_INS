"""Agent a04 -- hardware calibration / layout (one task per IBM backend).

* loads calibration (fake backend, JSON snapshot or live service)
* device characterisation summary (median T1/T2/readout/2q error)
* finds the best qubit chain for every chain length used by the models
  (built-in beam search, Qiskit preset layout, or Sangram's CAHGR search)
* reports estimated layer fidelity / EPLG of each chain and, when the
  calibration snapshot contains them, the vendor-measured layer fidelities
* optional layer-fidelity RB on live hardware (``hardware.layer_fidelity_rb``)
"""

from __future__ import annotations

from typing import Any

from ..physics.hardware import find_layout, load_calibration, score_chain
from ..physics.models import get_model
from .base import Agent, TaskContext, register_agent


def _layer_fidelity_rb(ctx: TaskContext, backend: Any, chain: list[int]) -> dict[str, Any]:
    """Run layer-fidelity RB with qiskit-experiments (live hardware only)."""
    try:
        from qiskit_experiments.library import LayerFidelity  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "qiskit-experiments not installed (uv pip install qiskit-experiments)"}
    two_q = [(a, b) for a, b in zip(chain, chain[1:])]
    layers = [two_q[0::2], two_q[1::2]]
    exp = LayerFidelity(physical_qubits=chain, two_qubit_layers=layers, lengths=[2, 4, 8, 16, 30, 50],
                        backend=backend, num_samples=6, seed=ctx.seed)
    data = exp.run().block_for_results()
    lf = data.analysis_results("LF").value
    eplg = data.analysis_results("EPLG").value
    return {"status": "ok", "layer_fidelity": float(lf.nominal_value), "eplg": float(eplg.nominal_value),
            "experiment_id": data.experiment_id}


@register_agent
class HardwareCalibrationAgent(Agent):
    name = "a04_hw_calibration"
    title = "Hardware calibration + layout search"
    fanout = "backend"
    deps = ()

    def run(self, ctx: TaskContext) -> dict[str, Any]:
        hw = ctx.cfg["hardware"]
        cal, backend = load_calibration(ctx.key, hw["calibration_source"], hw.get("calibration_dir"))
        summary = cal.summary()
        ctx.info("calibration loaded", **summary)
        ctx.save_json("calibration", cal.to_dict())

        sizes = sorted({get_model(m, ctx.cfg).n_sites for m in ctx.cfg["models"]})
        layouts: dict[str, Any] = {}
        for n in sizes:
            chain, method, notes = find_layout(cal, backend, n, hw["layout_method"], hw)
            score = score_chain(cal, chain)
            vendor = [v for v in cal.vendor_layer_fidelity if len(v["qubits"]) == n]
            entry = {"chain": chain, "method": method, "notes": notes, "score": score,
                     "vendor_layer_fidelity_same_length": vendor[:3]}
            if hw.get("compare_layout_methods", True) and method != "chain_search":
                ref = find_layout(cal, backend, n, "chain_search", hw)[0]
                entry["chain_search_reference"] = {"chain": ref, "score": score_chain(cal, ref)}
            if hw.get("layer_fidelity_rb", False):
                entry["layer_fidelity_rb"] = (_layer_fidelity_rb(ctx, backend, chain) if hw["calibration_source"] == "ibm"
                                              else {"status": "skipped", "reason": "needs calibration_source=ibm"})
            for note in notes:
                ctx.warning(note, n=n)
            ctx.decision("layout", chain, reasons=notes or [f"best {method} chain"], n=n, method=method,
                         est_layer_fidelity=score["est_layer_fidelity"])
            layouts[str(n)] = entry
        ctx.save_json("layouts", layouts)
        return {"backend": ctx.key, "calibration": summary,
                "layouts": {n: {"chain": v["chain"], "method": v["method"],
                                "est_layer_fidelity": v["score"]["est_layer_fidelity"],
                                "log_fidelity": v["score"]["log_fidelity"]} for n, v in layouts.items()}}
