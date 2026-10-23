"""Dynamic error-mitigation selection.

From circuit features (depth, 2q count, idle fraction) and the chosen chain's
calibration, pick a combination of

* DD   -- dynamical decoupling (XY4 / XpXm) against idle-qubit dephasing
* PT   -- Pauli twirling of 2q gates (turns coherent errors into Pauli noise)
* TREX -- twirled readout error extinction
* ZNE  -- zero-noise extrapolation (only when enough signal survives)

``policy="rules"`` uses the thresholds below; ``policy="llm"`` asks the vLLM
server for a choice, validates it and falls back to the rules. Both the
features and the reasons are logged, so the selection can be reported.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit

__all__ = ["circuit_features", "select_mitigation", "estimator_options"]

ALLOWED = {
    "dd": [False, True], "dd_sequence": ["XY4", "XpXm", "XX"], "twirling": [False, True],
    "trex": [False, True], "zne": [False, True], "zne_extrapolator": ["exponential", "linear", "polynomial_degree_2"],
}


def circuit_features(isa_circuits: list[QuantumCircuit], chain_score: dict[str, float]) -> dict[str, Any]:
    """Worst-case (deepest circuit) features used by the policy."""
    deepest = max(isa_circuits, key=lambda c: c.count_ops().get("cz", 0))
    ops = deepest.count_ops()
    n2q = int(ops.get("cz", 0))
    depth = deepest.depth()
    n = deepest.num_qubits
    busy = sum(v for k, v in ops.items() if k not in ("barrier", "measure"))
    idle_fraction = 1 - min(busy / max(n * depth, 1), 1.0)
    f_est = float(np.exp(n2q * np.log(max(1 - chain_score["mean_2q_error"], 1e-9))))
    return {"n_qubits": n, "max_n_2q": n2q, "max_depth": depth,
            "max_depth_2q": deepest.depth(lambda ins: ins.operation.num_qubits == 2),
            "idle_fraction": round(idle_fraction, 4), "est_circuit_fidelity": f_est,
            "mean_2q_error": chain_score["mean_2q_error"], "mean_readout_error": chain_score["mean_readout_error"]}


def _rules(f: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    th = cfg.get("thresholds", {})
    choice: dict[str, Any] = {}
    reasons: list[str] = []
    choice["trex"] = f["mean_readout_error"] > th.get("trex_readout", 0.005)
    reasons.append(f"TREX {'on' if choice['trex'] else 'off'}: mean readout error "
                   f"{f['mean_readout_error']:.4f} vs threshold {th.get('trex_readout', 0.005)}")
    choice["dd"] = f["idle_fraction"] > th.get("dd_idle", 0.2)
    choice["dd_sequence"] = "XpXm" if f["max_depth"] > th.get("dd_long_depth", 400) else "XY4"
    reasons.append(f"DD {'on' if choice['dd'] else 'off'} ({choice['dd_sequence']}): idle fraction "
                   f"{f['idle_fraction']:.2f}, depth {f['max_depth']}")
    choice["twirling"] = f["max_n_2q"] > 0
    reasons.append(f"Pauli twirling {'on' if choice['twirling'] else 'off'}: {f['max_n_2q']} two-qubit gates")
    lo, hi = th.get("zne_min_fidelity", 0.05), th.get("zne_max_fidelity", 0.7)
    choice["zne"] = lo < f["est_circuit_fidelity"] < hi
    choice["zne_extrapolator"] = "exponential"
    reasons.append(f"ZNE {'on' if choice['zne'] else 'off'}: estimated circuit fidelity "
                   f"{f['est_circuit_fidelity']:.3f} (useful range {lo}-{hi})")
    return choice, reasons


def _validate(choice: dict[str, Any]) -> bool:
    return all(k in choice and choice[k] in v for k, v in ALLOWED.items())


def select_mitigation(features: dict[str, Any], cfg: dict[str, Any],
                      llm: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Return ``{"choice", "reasons", "policy_used", "rule_choice", "llm_raw"}``."""
    if cfg.get("force"):
        forced = {**_rules(features, cfg)[0], **cfg["force"]}
        return {"choice": forced, "reasons": ["forced by config"], "policy_used": "fixed",
                "rule_choice": None, "llm_raw": None}
    rule_choice, reasons = _rules(features, cfg)
    out = {"choice": rule_choice, "reasons": reasons, "policy_used": "rules",
           "rule_choice": rule_choice, "llm_raw": None}
    if cfg.get("policy") == "llm" and llm is not None:
        system = ("You are an expert in quantum error mitigation on IBM superconducting hardware. "
                  "Choose mitigation settings. Reply ONLY with JSON having keys "
                  f"{list(ALLOWED)} plus 'reasons' (list of strings). Allowed values: {json.dumps(ALLOWED)}.")
        user = (f"Circuit/device features:\n{json.dumps(features, indent=2)}\n\n"
                f"A rule-based policy suggests:\n{json.dumps(rule_choice)}\n"
                "Agree or improve it. Consider cost: ZNE triples shots and amplifies variance.")
        raw = llm.chat_json(system, user)
        out["llm_raw"] = raw
        if raw is not None and _validate(raw):
            out.update(choice={k: raw[k] for k in ALLOWED}, policy_used="llm",
                       reasons=list(raw.get("reasons", [])) or ["(no reasons given)"])
        else:
            out["reasons"] = reasons + ["LLM answer missing/invalid -> used rules"]
    return out


def estimator_options(choice: dict[str, Any], exec_cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate a choice into ``EstimatorV2`` options (qiskit-ibm-runtime)."""
    tw = exec_cfg.get("twirling", {})
    opts: dict[str, Any] = {
        "default_shots": int(exec_cfg.get("shots", 4096)),
        "dynamical_decoupling": {"enable": bool(choice["dd"]), "sequence_type": choice["dd_sequence"]},
        "twirling": {"enable_gates": bool(choice["twirling"]),
                     "num_randomizations": int(tw.get("num_randomizations", 1000)),
                     "shots_per_randomization": int(tw.get("shots_per_randomization", 128))},
        "resilience": {"measure_mitigation": bool(choice["trex"]),
                       "measure_noise_learning": {"num_randomizations": 32, "shots_per_randomization": 100},
                       "zne_mitigation": bool(choice["zne"])},
    }
    if choice["zne"]:
        opts["resilience"]["zne"] = {"noise_factors": [1, 3, 5], "extrapolator": choice["zne_extrapolator"]}
    return opts
