"""Device calibration and 1D-chain layout selection.

Calibration can come from three sources, all mapped to the same
``Calibration`` record so the rest of the pipeline does not care:

* ``fake``  -- qiskit_ibm_runtime fake backends (FakeKingston, ...); offline, reproducible
* ``json``  -- calibration snapshots (e.g. Sangram's ``hardware_data/ibm_*.json``)
* ``ibm``   -- live backends through ``QiskitRuntimeService`` (needs saved credentials)

Layout methods: ``chain_search`` (built-in beam search for the best qubit
path), ``qiskit_default`` (preset pass manager), ``sangram`` (CAHGR
``optimize_layout``) and ``mapomatic`` (if installed). All results are scored
with the same calibration model so they can be compared in the paper.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Calibration", "load_calibration", "score_chain", "chain_search", "find_layout"]


@dataclass
class Calibration:
    backend: str
    source: str
    timestamp: str
    n_qubits: int
    edges: dict[str, float]                  # "a-b" (a < b) -> 2q gate error
    qubits: list[dict[str, float | None]]    # T1, T2 (us), readout_error, p01, p10, sx_error
    two_qubit_gate: str = "cz"
    vendor_layer_fidelity: list[dict[str, Any]] = field(default_factory=list)

    def edge_error(self, a: int, b: int) -> float:
        return self.edges.get(f"{min(a, b)}-{max(a, b)}", 1.0)

    def neighbors(self) -> dict[int, list[int]]:
        nb: dict[int, list[int]] = {q: [] for q in range(self.n_qubits)}
        for key, err in self.edges.items():
            if err < 1.0:
                a, b = map(int, key.split("-"))
                nb[a].append(b)
                nb[b].append(a)
        return nb

    def summary(self) -> dict[str, Any]:
        def med(name: str) -> float | None:
            vals = [q[name] for q in self.qubits if q.get(name) is not None]
            return float(np.median(vals)) if vals else None
        errs = [e for e in self.edges.values() if e < 1.0]
        return {"backend": self.backend, "source": self.source, "timestamp": self.timestamp,
                "n_qubits": self.n_qubits, "n_couplers": len(errs),
                "median_T1_us": med("T1"), "median_T2_us": med("T2"),
                "median_readout_error": med("readout_error"), "median_sx_error": med("sx_error"),
                "median_2q_error": float(np.median(errs)) if errs else None}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- loaders
def _from_target(name: str, source: str, target: Any, timestamp: str) -> Calibration:
    two_q = next((g for g in ("cz", "ecr", "cx") if g in target.operation_names), "cz")
    edges = {}
    for qargs, props in target[two_q].items():
        if props is not None and props.error is not None:
            a, b = qargs
            key = f"{min(a, b)}-{max(a, b)}"
            edges[key] = min(edges.get(key, 1.0), float(props.error))
    qubits = []
    for q in range(target.num_qubits):
        qp = target.qubit_properties[q] if target.qubit_properties else None
        meas = target["measure"].get((q,)) if "measure" in target.operation_names else None
        sx = target["sx"].get((q,)) if "sx" in target.operation_names else None
        ro = float(meas.error) if meas is not None and meas.error is not None else None
        qubits.append({"T1": qp.t1 * 1e6 if qp is not None and qp.t1 else None,
                       "T2": qp.t2 * 1e6 if qp is not None and qp.t2 else None,
                       "readout_error": ro, "p01": ro, "p10": ro,
                       "sx_error": float(sx.error) if sx is not None and sx.error is not None else None})
    return Calibration(name, source, timestamp, target.num_qubits, edges, qubits, two_q)


def _from_json(path: Path) -> Calibration:
    d = json.loads(path.read_text())
    qubits = []
    for q_entries in d["qubits"]:
        m = {e["name"]: e["value"] for e in q_entries}
        qubits.append({"T1": m.get("T1"), "T2": m.get("T2"), "readout_error": m.get("readout_error"),
                       "p01": m.get("prob_meas1_prep0"), "p10": m.get("prob_meas0_prep1"), "sx_error": None})
    edges: dict[str, float] = {}
    two_q = "cz"
    for g in d["gates"]:
        params = {p["name"]: p["value"] for p in g.get("parameters", [])}
        if g["gate"] in ("cz", "ecr") and len(g["qubits"]) == 2:
            two_q = g["gate"]
            a, b = g["qubits"]
            key = f"{min(a, b)}-{max(a, b)}"
            edges[key] = min(edges.get(key, 1.0), float(params.get("gate_error", 1.0)))
        elif g["gate"] == "sx":
            qubits[g["qubits"][0]]["sx_error"] = params.get("gate_error")
    lf_vals = {e["name"]: e["value"] for e in d.get("general", []) if e["name"].startswith("lf_")}
    lf = [{"name": x["name"], "qubits": x["qubits"], "layer_fidelity": lf_vals.get(x["name"])}
          for x in d.get("general_qlists", []) if x["name"].startswith("lf_")]
    return Calibration(d["backend_name"], "json", d.get("last_update_date", ""), len(qubits),
                       edges, qubits, two_q, lf)


def _fake_backend(name: str) -> Any:
    from qiskit_ibm_runtime import fake_provider
    cls_name = "Fake" + name.replace("ibm_", "").capitalize()
    if not hasattr(fake_provider, cls_name):
        raise ValueError(f"no fake backend {cls_name} in qiskit_ibm_runtime.fake_provider")
    return getattr(fake_provider, cls_name)()


def load_calibration(name: str, source: str, calibration_dir: str | None = None) -> tuple[Calibration, Any]:
    """Return ``(Calibration, backend_or_None)`` for ``ibm_<name>``."""
    if source == "fake":
        backend = _fake_backend(name)
        ts = str(getattr(backend.properties(), "last_update_date", "")) if hasattr(backend, "properties") else ""
        return _from_target(name, "fake", backend.target, ts), backend
    if source == "json":
        return _from_json(Path(calibration_dir or ".") / f"{name}.json"), None
    if source == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService().backend(name)
        return _from_target(name, "ibm", backend.target, str(backend.properties().last_update_date)), backend
    raise ValueError(f"unknown calibration source {source!r}")


# ----------------------------------------------------------------- layout
def score_chain(cal: Calibration, chain: list[int]) -> dict[str, float]:
    """Calibration-model score of a qubit path (higher log_fidelity is better)."""
    e2 = [cal.edge_error(a, b) for a, b in zip(chain, chain[1:])]
    ro = [cal.qubits[q].get("readout_error") or 0.0 for q in chain]
    e1 = [cal.qubits[q].get("sx_error") or 0.0 for q in chain]
    lf = float(np.prod([1 - e for e in e2]) * np.prod([(1 - e) ** 2 for e in e1]))
    log_f = sum(math.log(max(1 - e, 1e-9)) for e in e2) + 0.5 * sum(math.log(max(1 - r, 1e-9)) for r in ro)
    return {"log_fidelity": log_f, "est_layer_fidelity": lf,
            "est_eplg": 1 - lf ** (1 / max(len(e2), 1)),
            "mean_2q_error": float(np.mean(e2)) if e2 else 0.0, "max_2q_error": float(max(e2, default=0.0)),
            "mean_readout_error": float(np.mean(ro))}


def chain_search(cal: Calibration, n: int, beam: int = 256) -> list[int]:
    """Beam search for the length-``n`` simple path with the best score."""
    nb = cal.neighbors()
    w_edge = {k: math.log(max(1 - e, 1e-9)) for k, e in cal.edges.items()}
    w_q = [0.5 * math.log(max(1 - (q.get("readout_error") or 0.0), 1e-9)) for q in cal.qubits]
    paths = [((w_q[q]), (q,)) for q in range(cal.n_qubits) if nb[q]]
    for _ in range(n - 1):
        new: dict[tuple[int, ...], float] = {}
        for score, path in paths:
            for end, grow in ((path[-1], lambda p, x: p + (x,)), (path[0], lambda p, x: (x,) + p)):
                for x in nb[end]:
                    if x in path:
                        continue
                    p = grow(path, x)
                    canon = p if p[0] < p[-1] else p[::-1]
                    s = score + w_q[x] + w_edge[f"{min(end, x)}-{max(end, x)}"]
                    if s > new.get(canon, -np.inf):
                        new[canon] = s
        if not new:
            raise ValueError(f"no connected chain of {n} qubits on {cal.backend}")
        paths = sorted(((s, p) for p, s in new.items()), reverse=True)[:beam]
    return list(paths[0][1])


def _line_circuit(n: int) -> Any:
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for start in (0, 1):
        for i in range(start, n - 1, 2):
            qc.cz(i, i + 1)
    return qc


def _qiskit_default(backend: Any, n: int) -> list[int]:
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
    from .qlock import TRANSPILE_LOCK
    with TRANSPILE_LOCK:
        isa = pm.run(_line_circuit(n))
    return list(isa.layout.initial_index_layout(filter_ancillas=True))


def _sangram(cal_json: Path, n: int, sangram_path: str) -> list[int]:
    if sangram_path not in sys.path:
        sys.path.insert(0, sangram_path)
    from cahgr.hardware_kg import build_hardware_knowledge_graph  # type: ignore
    from cahgr.layout_search import optimize_layout  # type: ignore
    kg = build_hardware_knowledge_graph(str(cal_json))
    res = optimize_layout(_line_circuit(n), kg, str(cal_json), objective="maximum_fidelity")
    return [int(res.assignment[i]) for i in range(n)]


def find_layout(cal: Calibration, backend: Any, n: int, method: str, hw_cfg: dict[str, Any]) -> tuple[list[int], str, list[str]]:
    """Return ``(chain, method_used, notes)``; falls back to ``chain_search``."""
    notes: list[str] = []
    try:
        chain: list[int] | None = None
        if method == "qiskit_default":
            if backend is None:
                raise RuntimeError("qiskit_default needs a backend object (use source fake or ibm)")
            chain = _qiskit_default(backend, n)
        elif method == "sangram":
            cal_json = Path(hw_cfg["calibration_dir"]) / f"{cal.backend}.json"
            chain = _sangram(cal_json, n, hw_cfg["sangram_path"])
        elif method == "mapomatic":
            import mapomatic  # type: ignore  # noqa: F401
            raise RuntimeError("mapomatic adapter not implemented for this qiskit version")
        if chain is not None:
            if all(cal.edge_error(a, b) < 1.0 for a, b in zip(chain, chain[1:])):
                return chain, method, notes
            raise RuntimeError(f"layout {chain} is not a connected path (would need SWAPs)")
    except Exception as exc:  # noqa: BLE001 - any adapter failure falls back, but is recorded
        notes.append(f"{method} failed ({type(exc).__name__}: {exc}); fell back to chain_search")
    return chain_search(cal, n, int(hw_cfg.get("beam_width", 256))), "chain_search", notes
