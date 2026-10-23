"""Physics checks against exact (statevector / dense) results on small chains."""

from dataclasses import replace

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from qins.physics.circuits import hva_ansatz, kick, kick_signs, trotter_circuit
from qins.physics.dsf import rgf_from_z, rgf_to_dsf
from qins.physics.models import PRESETS, get_model
from qins.physics.mps import circuit_to_mps, dmrg, z_profile


@pytest.mark.parametrize("name", list(PRESETS))
def test_dmrg_matches_exact(name):
    model = replace(PRESETS[name], n_sites=8)
    e_dmrg, _, _ = dmrg(model, chi=32)
    e_exact, _ = model.exact_ground_state()
    assert abs(e_dmrg - e_exact) < 1e-8


@pytest.mark.parametrize("name", ["kcuf3", "cscox3"])
def test_trotter_converges_to_exact_evolution(name):
    model = replace(PRESETS[name], n_sites=6)
    t = 0.6
    exact = expm(-1j * t * model.hamiltonian().to_matrix())
    psi0 = Statevector(hva_ansatz(model, 1, "neel").assign_parameters(np.zeros(len(hva_ansatz(model, 1).parameters))))
    target = exact @ psi0.data
    errs = []
    for steps in (1, 4):
        qc = trotter_circuit(model, t / steps, steps)
        errs.append(1 - abs(np.vdot(target, psi0.evolve(qc).data)) ** 2)
    assert errs[1] < errs[0] and errs[1] < 1e-4


def test_mps_z_profile_matches_statevector():
    model = replace(PRESETS["kcuf3"], n_sites=6)
    qc = hva_ansatz(model, 1, "singlet").assign_parameters([0.3, 0.7]).compose(kick(model))
    qc = qc.compose(trotter_circuit(model, 0.6, 2))
    sv = Statevector(qc)
    z_sv = [sv.probabilities([j]) @ np.array([1, -1]) for j in range(6)]
    assert np.allclose(z_profile(circuit_to_mps(qc, max_bond=64)), z_sv, atol=1e-8)


def test_kick_signs_and_rgf_combination():
    assert kick_signs(PRESETS["kcuf3"]) == [1]
    assert kick_signs(PRESETS["kcuf3"], max_abs_z=1e-6) == [1]
    assert kick_signs(PRESETS["kcuf3"], max_abs_z=0.6) == [1, -1]  # e.g. Neel-like VQE state
    assert kick_signs(PRESETS["cscox3"]) == [1, -1]
    z = np.random.default_rng(0).normal(size=(2, 4, 3))
    assert np.allclose(rgf_from_z(z), (z[0] - z[1])[1:] / 2)
    assert np.allclose(rgf_from_z(z[:1]), z[0, 1:])


def test_dsf_of_single_mode_peaks_at_its_frequency():
    dt, n_t, w0 = 0.2, 200, 1.5
    t = dt * np.arange(1, n_t + 1)
    rgf = np.zeros((n_t, 5))
    rgf[:, 2] = -np.sin(w0 * t)  # local response at the centre site
    out = rgf_to_dsf(rgf, dt, center=2, n_omega=300)
    w_peak = out["omega"][np.argmax(out["S"][:, 10])]
    assert abs(w_peak - w0) < 0.05


def test_model_overrides():
    cfg = {"system": {"n_sites": 10}, "model_overrides": {"cscox3": {"hs": 0.2}}}
    m = get_model("cscox3", cfg)
    assert m.n_sites == 10 and m.hs == 0.2 and m.has_field
