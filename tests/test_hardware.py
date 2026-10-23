from qins.physics.hardware import chain_search, find_layout, load_calibration, score_chain
from qins.physics.noise import calibrated_noise_model


def test_fake_calibration_and_chain():
    cal, backend = load_calibration("ibm_kingston", "fake")
    assert cal.n_qubits == 156 and len(cal.edges) > 100
    chain = chain_search(cal, 10, beam=64)
    assert len(set(chain)) == 10
    assert all(cal.edge_error(a, b) < 1 for a, b in zip(chain, chain[1:]))
    s = score_chain(cal, chain)
    assert 0 < s["est_layer_fidelity"] <= 1
    nm = calibrated_noise_model(cal, chain)
    assert "cz" in nm.noise_instructions


def test_qiskit_default_layout_is_a_path():
    cal, backend = load_calibration("ibm_kingston", "fake")
    chain, method, notes = find_layout(cal, backend, 6, "qiskit_default", {})
    assert len(chain) == 6 and all(cal.edge_error(a, b) < 1 for a, b in zip(chain, chain[1:]))
