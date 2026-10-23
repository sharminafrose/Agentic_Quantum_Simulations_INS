"""Circuit builders: HVA ground-state ansatz, neutron kick, Trotter evolution.

Follows the IBM ``simulate-neutron-scattering`` tutorial, generalised to
anisotropic couplings and (staggered) fields.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from .models import SpinModel
from .qlock import TRANSPILE_LOCK

__all__ = ["hva_ansatz", "initial_state", "kick", "kick_signs", "trotter_circuit", "n_hva_params", "circuit_stats"]


def _bond_op(model: SpinModel) -> SparsePauliOp:
    return SparsePauliOp.from_list(model.bond_pauli_terms())


def initial_state(model: SpinModel, init: str) -> QuantumCircuit:
    """``singlet``: product of nearest-neighbour singlets; ``neel``: |0101...>."""
    n = model.n_sites
    qc = QuantumCircuit(n, name=f"init_{init}")
    if init == "singlet":
        for i in range(0, n - 1, 2):
            qc.x(i)
            qc.h(i)
            qc.cx(i, i + 1)
            qc.x(i + 1)
    elif init == "neel":
        for i in range(1, n, 2):
            qc.x(i)
    else:
        raise ValueError(f"unknown init {init!r}")
    return qc


def n_hva_params(model: SpinModel, layers: int) -> int:
    return layers * (3 if model.has_field else 2)


def hva_ansatz(model: SpinModel, layers: int, init: str = "singlet") -> QuantumCircuit:
    """Hamiltonian variational ansatz: alternating odd/even bond layers (+ field layer)."""
    n = model.n_sites
    theta = ParameterVector("theta", n_hva_params(model, layers))
    qc = initial_state(model, init)
    bond = _bond_op(model)
    per = 3 if model.has_field else 2
    for layer in range(layers):
        odd, even = theta[per * layer], theta[per * layer + 1]
        for i in range(1, n - 1, 2):
            qc.append(PauliEvolutionGate(bond, time=odd), [i, i + 1])
        for i in range(0, n - 1, 2):
            qc.append(PauliEvolutionGate(bond, time=even), [i, i + 1])
        if model.has_field:
            for i in range(n):
                if model.field_at(i):
                    qc.rz(theta[per * layer + 2] * model.field_at(i), i)
    return qc


def kick(model: SpinModel, sign: int = 1) -> QuantumCircuit:
    """Rz(+-pi/2) on the centre site: the 'neutron spin transfer' perturbation."""
    qc = QuantumCircuit(model.n_sites, name=f"kick{'+' if sign > 0 else '-'}")
    qc.rz(sign * np.pi / 2, model.center)
    return qc


def kick_signs(model: SpinModel, max_abs_z: float | None = None, tol: float = 1e-3) -> list[int]:
    """Kick signs needed for an exact retarded Green's function.

    With U = Rz(+-pi/2)_c the kicked signal is <Z_j(t)>_+- = A_j(t) +- G_j(t),
    where G is the retarded response and A = (<Z_j(t)> + <Z_c Z_j(t) Z_c>)/2.
    A vanishes only for spin-flip symmetric states (<Z_j> = 0), so one kick
    suffices only then; otherwise (fields, or e.g. a Neel-like VQE state) we
    run both and use G = (z+ - z-)/2. ``max_abs_z`` is max_j |<Z_j>| of the
    prepared state; when given it is checked as well as the model fields.
    """
    broken = model.has_field or (max_abs_z is not None and max_abs_z > tol)
    return [1, -1] if broken else [1]


def _bond_layer(qc: QuantumCircuit, model: SpinModel, start: int, t: float) -> None:
    bond = _bond_op(model)
    for i in range(start, model.n_sites - 1, 2):
        qc.append(PauliEvolutionGate(bond, time=t), [i, i + 1])


def _field_layer(qc: QuantumCircuit, model: SpinModel, t: float) -> None:
    for i in range(model.n_sites):
        if model.field_at(i):
            qc.rz(model.field_at(i) * t, i)  # exp(-i t h/2 Z)


def trotter_circuit(model: SpinModel, dt: float, n_steps: int, order: int = 2) -> QuantumCircuit:
    """``n_steps`` Trotter steps of exp(-i H dt) (even/odd bond splitting)."""
    qc = QuantumCircuit(model.n_sites, name=f"trotter{n_steps}")
    for _ in range(n_steps):
        if order == 1:
            _bond_layer(qc, model, 0, dt)
            _bond_layer(qc, model, 1, dt)
            _field_layer(qc, model, dt)
        else:
            _field_layer(qc, model, dt / 2)
            _bond_layer(qc, model, 0, dt / 2)
            _bond_layer(qc, model, 1, dt)
            _bond_layer(qc, model, 0, dt / 2)
            _field_layer(qc, model, dt / 2)
    return qc


def circuit_stats(qc: QuantumCircuit) -> dict[str, int]:
    """Two-qubit gate count and depth after decomposing to CX + 1q gates."""
    from qiskit import transpile
    with TRANSPILE_LOCK:
        t = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
    ops = t.count_ops()
    return {
        "n_qubits": t.num_qubits,
        "depth": t.depth(),
        "depth_2q": t.depth(lambda ins: ins.operation.num_qubits == 2),
        "n_2q": int(ops.get("cx", 0)),
        "n_1q": int(sum(v for k, v in ops.items() if k in ("rz", "sx", "x"))),
        "n_params": len(qc.parameters),
    }
