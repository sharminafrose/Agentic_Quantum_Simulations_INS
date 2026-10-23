"""Tensor-network (quimb) helpers: DMRG, circuit-MPS simulation, Z profiles.

Site ``i`` of every MPS corresponds to qubit ``i`` of the Qiskit circuits.
For chains with ``n <= 16`` the exact statevector is used where cheaper.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from .models import SpinModel
from .qlock import TRANSPILE_LOCK

__all__ = [
    "decompose", "circuit_to_mps", "z_profile", "energy", "dmrg", "fidelity_with_mps",
    "fidelity_with_vector", "kicked_dynamics",
]

_QUIMB_BASIS = ["cx", "rz", "sx", "x", "h", "rx", "ry", "rzz", "rxx", "ryy"]
_Z = np.diag([1.0, -1.0])
EXACT_MAX_QUBITS = 16


def decompose(qc: QuantumCircuit) -> QuantumCircuit:
    """Rewrite PauliEvolutionGates etc. into gates quimb understands."""
    with TRANSPILE_LOCK:
        return transpile(qc, basis_gates=_QUIMB_BASIS, optimization_level=0)


def circuit_to_mps(qc: QuantumCircuit, max_bond: int = 128, cutoff: float = 1e-10,
                   psi0: Any = None) -> Any:
    """Apply ``qc`` to ``psi0`` (default |0...0>) and return the resulting quimb MPS."""
    import quimb.tensor as qtn
    import qiskit_quimb
    circ = (qtn.CircuitMPS(psi0=psi0.copy(), max_bond=max_bond, cutoff=cutoff) if psi0 is not None
            else qtn.CircuitMPS(qc.num_qubits, max_bond=max_bond, cutoff=cutoff))
    circ.apply_gates(qiskit_quimb.quimb_gates(decompose(qc)))
    return circ.psi


def z_profile(psi: Any) -> np.ndarray:
    """<Z_j> for every site of an MPS, using the moving orthogonality centre."""
    psi = psi.copy()
    out = np.empty(psi.L)
    for j in range(psi.L):
        psi.canonize(j)
        t = psi[j]
        out[j] = ((t.H @ t.gate(_Z, psi.site_ind(j))) / (t.H @ t)).real
    return out


def energy(qc: QuantumCircuit, model: SpinModel, max_bond: int = 64) -> float:
    """<H> for a bound circuit (statevector for small n, MPS otherwise)."""
    if qc.num_qubits <= EXACT_MAX_QUBITS:
        return float(Statevector(qc).expectation_value(model.hamiltonian()).real)
    psi = circuit_to_mps(qc, max_bond=max_bond)
    return float(np.real(psi.H @ model.quimb_mpo().apply(psi)))


def dmrg(model: SpinModel, chi: int, cutoff: float = 1e-10, max_sweeps: int = 20,
         tol: float = 1e-8) -> tuple[float, Any, dict[str, Any]]:
    """Two-site DMRG at fixed max bond dimension ``chi``."""
    import quimb.tensor as qtn
    t0 = time.time()
    solver = qtn.DMRG2(model.quimb_mpo(), bond_dims=[min(chi, 16), chi], cutoffs=cutoff)
    converged = solver.solve(tol=tol, max_sweeps=max_sweeps, verbosity=0)
    psi = solver.state
    info = {"chi": chi, "energy": float(np.real(solver.energy)), "converged": bool(converged),
            "max_bond_reached": int(psi.max_bond()), "n_sweeps": len(solver.energies),
            "sweep_energies": [float(np.real(e)) for e in solver.energies],
            "wall_s": round(time.time() - t0, 3)}
    return info["energy"], psi, info


def fidelity_with_mps(qc: QuantumCircuit, target: Any, max_bond: int = 128) -> float:
    psi = circuit_to_mps(qc, max_bond=max_bond)
    return float(abs(psi.H @ target) ** 2 / abs(target.H @ target))


def fidelity_with_vector(qc: QuantumCircuit, vec: np.ndarray) -> float:
    """Fidelity against an exact (qiskit-ordered) state vector."""
    return float(abs(np.vdot(vec, Statevector(qc).data)) ** 2)


def kicked_dynamics(model: SpinModel, psi0: Any, step_circuit: QuantumCircuit, kick_circuit: QuantumCircuit,
                    n_steps: int, max_bond: int = 128, cutoff: float = 1e-10) -> dict[str, np.ndarray]:
    """<Z_j(t)> (shape (n_steps+1, n)) after a centre kick, evolving ``psi0`` with ``step_circuit``."""
    psi = circuit_to_mps(kick_circuit, max_bond=max_bond, cutoff=cutoff, psi0=psi0)
    z = [z_profile(psi)]
    for _ in range(n_steps):
        psi = circuit_to_mps(step_circuit, max_bond=max_bond, cutoff=cutoff, psi0=psi)
        z.append(z_profile(psi))
    return {"z": np.array(z), "max_bond_used": int(psi.max_bond())}
