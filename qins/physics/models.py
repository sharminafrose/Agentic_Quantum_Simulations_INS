"""Generalised spin-1/2 chain models for neutron-scattering benchmarks.

Every material is described by one parameter set of the XXZ chain with
uniform and staggered longitudinal fields (spin operators S = sigma/2)::

    H = J * sum_i [ jxy (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + jz Sz_i Sz_{i+1} ]
        + sum_i (hz + (-1)^i hs) Sz_i

Dynamics are run in units of J (J = 1); ``energy_scale_meV`` converts the
frequency axis to meV for comparison with INS data. New materials only need a
new preset (or a ``model_overrides`` entry in the config, or agent a00's
paper-derived parameters) -- nothing else in the pipeline is model specific.

NOTE: the preset numbers for ``two_soliton`` and ``cscox3`` are starting
points; agent a00 records which values were actually used and where they came
from. Verify against the literature before quoting them in a paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

__all__ = ["SpinModel", "PRESETS", "get_model", "model_from_dict"]


@dataclass(frozen=True)
class SpinModel:
    name: str
    n_sites: int = 12
    jxy: float = 1.0
    jz: float = 1.0
    hz: float = 0.0
    hs: float = 0.0
    energy_scale_meV: float = 1.0
    material: str = ""
    description: str = ""
    search_terms: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- helpers
    @property
    def center(self) -> int:
        """Site that receives the neutron 'kick' (matches the IBM tutorial)."""
        return self.n_sites // 2 - 1

    @property
    def has_field(self) -> bool:
        return abs(self.hz) > 0 or abs(self.hs) > 0

    def field_at(self, i: int) -> float:
        return self.hz + (-1) ** i * self.hs

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["search_terms"] = list(self.search_terms)
        d["references"] = list(self.references)
        return d

    # --------------------------------------------------------- operators
    def bond_pauli_terms(self) -> list[tuple[str, float]]:
        """Pauli coefficients of one bond (S.S = sigma.sigma / 4)."""
        terms = [("XX", self.jxy / 4), ("YY", self.jxy / 4), ("ZZ", self.jz / 4)]
        return [(p, c) for p, c in terms if c != 0.0]

    def hamiltonian(self) -> SparsePauliOp:
        n = self.n_sites
        terms = [(p, [i, i + 1], c) for i in range(n - 1) for p, c in self.bond_pauli_terms()]
        terms += [("Z", [i], self.field_at(i) / 2) for i in range(n) if self.field_at(i) != 0.0]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()

    def quimb_mpo(self) -> Any:
        import quimb.tensor as qtn
        builder = qtn.SpinHam1D(S=1 / 2)
        if self.jxy:
            builder += self.jxy, "X", "X"
            builder += self.jxy, "Y", "Y"
        if self.jz:
            builder += self.jz, "Z", "Z"
        for i in range(self.n_sites):
            if self.field_at(i):
                builder[i] += self.field_at(i), "Z"
        return builder.build_mpo(self.n_sites)

    def exact_ground_state(self) -> tuple[float, np.ndarray]:
        """Sparse diagonalisation (qiskit little-endian ordering); small chains only."""
        if self.n_sites > 16:
            raise ValueError("exact diagonalisation limited to n_sites <= 16")
        from scipy.sparse.linalg import eigsh
        mat = self.hamiltonian().to_matrix(sparse=True)
        vals, vecs = eigsh(mat, k=1, which="SA")
        return float(vals[0]), vecs[:, 0]


PRESETS: dict[str, SpinModel] = {
    "xx": SpinModel(
        name="xx", jxy=1.0, jz=0.0, energy_scale_meV=1.0,
        material="XX chain (free fermions)",
        description="Gapless XX chain; exactly solvable via Jordan-Wigner, sharp benchmark.",
        search_terms=("XX chain", "free fermion", "Jordan-Wigner"),
    ),
    "kcuf3": SpinModel(
        name="kcuf3", jxy=1.0, jz=1.0, energy_scale_meV=33.5,
        material="KCuF3",
        description="Near-ideal spin-1/2 Heisenberg chain; two-spinon continuum (Bethe ansatz).",
        search_terms=("KCuF3", "Heisenberg chain", "spinon", "Bethe ansatz"),
        references=("arXiv:2603.15608", "Lake et al., Nat. Mater. 4, 329 (2005)"),
    ),
    "two_soliton": SpinModel(
        name="two_soliton", jxy=0.3, jz=1.0, energy_scale_meV=1.0,
        material="Ising-like XXZ chain",
        description="Gapped Ising-like chain; neutron creates a two-domain-wall (soliton) continuum.",
        search_terms=("two-soliton", "domain wall", "Ising-like", "XXZ"),
    ),
    "cscox3": SpinModel(
        name="cscox3", jxy=0.14, jz=1.0, hs=0.05, energy_scale_meV=13.0,
        material="CsCoX3 (X = Cl, Br)",
        description="Ising-like chain with staggered interchain field; confined solitons (Zeeman ladder).",
        search_terms=("CsCoBr3", "CsCoCl3", "soliton", "Zeeman ladder", "confinement"),
    ),
}


def model_from_dict(d: dict[str, Any]) -> SpinModel:
    known = {f for f in SpinModel.__dataclass_fields__}
    clean = {k: v for k, v in d.items() if k in known}
    for key in ("search_terms", "references"):
        if key in clean:
            clean[key] = tuple(clean[key])
    return SpinModel(**clean)


def get_model(name: str, cfg: dict[str, Any]) -> SpinModel:
    """Preset + global ``system.n_sites`` + per-model ``model_overrides``."""
    base = PRESETS.get(name, SpinModel(name=name))
    base = replace(base, n_sites=int(cfg.get("system", {}).get("n_sites", base.n_sites)))
    overrides = (cfg.get("model_overrides") or {}).get(name) or {}
    if overrides:
        base = model_from_dict({**base.to_dict(), **overrides})
    return base
