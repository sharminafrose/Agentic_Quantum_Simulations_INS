"""Noisy simulation with Qiskit Aer (matrix-product-state method).

* ``depolarizing_noise_model`` -- uniform depolarising channel (agent a05 sweep)
* ``calibrated_noise_model``   -- per-qubit / per-coupler noise taken from a
  ``Calibration`` restricted to the chosen chain (agent a08 local emulator)
* ``run_z``                    -- <Z_j> for a list of circuits from sampled counts
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from .hardware import Calibration
from .qlock import TRANSPILE_LOCK

__all__ = ["HW_BASIS", "to_line_isa", "depolarizing_noise_model", "calibrated_noise_model",
           "run_z", "readout_correct"]

HW_BASIS = ["cz", "rz", "sx", "x"]


def to_line_isa(circuits: list[QuantumCircuit], seed: int = 0, optimization_level: int = 2) -> list[QuantumCircuit]:
    """Transpile logical chain circuits to a line of qubits in the IBM Heron basis."""
    n = circuits[0].num_qubits
    with TRANSPILE_LOCK:
        isa = transpile(circuits, basis_gates=HW_BASIS, coupling_map=CouplingMap.from_line(n),
                        initial_layout=list(range(n)), optimization_level=optimization_level,
                        seed_transpiler=seed)
    for c in isa:  # Z_j is read from qubit j: routing must not permute qubits
        if c.layout is not None and c.layout.final_layout is not None and \
                any(c.layout.final_index_layout()[j] != j for j in range(n)):
            raise RuntimeError(f"{c.name}: routing permuted qubits; circuits must be nearest-neighbour")
    return isa


def depolarizing_noise_model(p2: float, one_qubit_ratio: float = 0.1, readout: float = 0.0) -> Any:
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel(basis_gates=HW_BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz"])
    if one_qubit_ratio > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 * one_qubit_ratio, 1), ["sx", "x"])
    if readout > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm


def calibrated_noise_model(cal: Calibration, chain: list[int], include_readout: bool = True) -> Any:
    """Noise model on virtual qubits 0..n-1 where virtual i == physical ``chain[i]``."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel(basis_gates=HW_BASIS)
    for i, (a, b) in enumerate(zip(chain, chain[1:])):
        p = min(cal.edge_error(a, b) * 4 / 3, 1.0)  # gate error -> depolarising parameter
        nm.add_quantum_error(depolarizing_error(p, 2), ["cz"], [i, i + 1])
        nm.add_quantum_error(depolarizing_error(p, 2), ["cz"], [i + 1, i])
    for i, q in enumerate(chain):
        e1 = cal.qubits[q].get("sx_error") or 0.0
        if e1 > 0:
            nm.add_quantum_error(depolarizing_error(min(e1 * 2, 1.0), 1), ["sx", "x"], [i])
        if include_readout:
            p01 = cal.qubits[q].get("p01") or cal.qubits[q].get("readout_error") or 0.0
            p10 = cal.qubits[q].get("p10") or cal.qubits[q].get("readout_error") or 0.0
            nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [i])
    return nm


def readout_correct(z: np.ndarray, p01: np.ndarray, p10: np.ndarray) -> np.ndarray:
    """Invert independent single-qubit assignment errors on <Z_j> (TREX-like, emulated)."""
    m1 = (1 - z) / 2
    p1 = (m1 - p01) / np.clip(1 - p01 - p10, 1e-6, None)
    return np.clip(1 - 2 * p1, -1.0, 1.0)


def run_z(circuits: list[QuantumCircuit], noise_model: Any, shots: int, seed: int,
          method: str = "matrix_product_state") -> np.ndarray:
    """Sample each circuit and return <Z_j> with shape (n_circuits, n_qubits)."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method=method, noise_model=noise_model, seed_simulator=seed,
                       max_parallel_threads=int(os.environ.get("QINS_THREADS", 0)))
    measured = []
    for qc in circuits:
        m = qc.copy()
        m.measure_all()
        measured.append(m)
    result = sim.run(measured, shots=shots).result()
    n = circuits[0].num_qubits
    out = np.zeros((len(circuits), n))
    for k in range(len(circuits)):
        counts = result.get_counts(k)
        for bitstr, c in counts.items():
            bits = bitstr.replace(" ", "")[::-1]  # little-endian: bits[j] is qubit j
            out[k] += c * np.array([1.0 if b == "0" else -1.0 for b in bits[:n]])
        out[k] /= shots
    return out
