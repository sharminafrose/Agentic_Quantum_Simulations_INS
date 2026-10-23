---
paths:
  - "qins/physics/dsf.py"
  - "qins/agents/a07_assembly.py"
  - "qins/agents/a08_hw_execution.py"
  - "qins/agents/a09_dsf.py"
  - "qins/agents/a10_benchmark.py"
  - "qins/physics/models.py"
---

# Neutron-scattering DSF pipeline (KCuF3, 1D Heisenberg chain)

Reference implementation: IBM Quantum tutorial "Simulate neutron scattering in quantum
materials with quantum circuits" (quantum.cloud.ibm.com/docs/en/tutorials/simulate-neutron-scattering),
which reproduces Lee et al., arXiv:2603.15608. When this file and the tutorial disagree,
the tutorial code is the source of truth for behavior; flag the disagreement, do not silently
pick one.

This file covers shared conventions and the assembly of the stages into S(q, w).
Stage-specific rules live in the companion files:
- `state-prep.md`: DMRG reference, HVA ground-state circuit, perturbation
- `time-evolution.md`: second-order Trotter, AQC compression, mixed AQC + Trotter circuits

## Rigor rules for this project

- Every number written to a figure, table, log line meant for a paper, or docstring must come
  from code that was actually run in this repo. Never paste numbers from the tutorial as if
  they were ours.
- State units and conventions next to every reported quantity (J convention, sigma vs S,
  time in units of 1/J).
- Distinguish three kinds of claims in comments and notes: (a) reproduced from the tutorial,
  (b) derived by us, (c) verified numerically by us (name the test). Do not upgrade (a) or (b)
  to (c) without running the test.
- Fidelities from AQC and HVA are MPS-level fidelities against bond-truncated targets. Always
  say "MPS fidelity at chi = X" and never call them exact fidelities.
- Do not use em-dashes in any text, comments, or docs written for this project.

## Physical model and conventions (shared by all stages)

- H = J * sum_i [ Sz_i Sz_{i+1} + eps (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) ], open boundary conditions,
  S = sigma / 2, J = 1 is the energy unit, eps = 1 (isotropic point).
- In Pauli form the bond term is (J/4)(eps XX + eps YY + ZZ). This is the `pair_ham` used in
  Trotter steps. The quimb MPO uses spin-1/2 operators directly.
- Factor-of-2 warning: the paper's Eq. 3 carries an extra overall factor of 2, so the paper's J
  is twice this J. Any comparison of energy axes with the paper or with INS data must convert.
- Verified by us (exact diagonalization, n = 10, OBC): E0 = -4.258035207 in these units, which
  matches the tutorial's DMRG value -4.258035. Keep this as a regression test.
- n_qubits must be even (singlet-pair initial state and brickwork layers assume it).
- Reference site: `center = n_qubits // 2 - 1` (0-indexed). For even n there is no true center;
  this is the left-of-center site. The DSF momentum phases are built around this exact choice,
  so changing `center` in one stage without the other is a bug.
- Time grid: t_m = m * dt for m = 1..n_steps. The t = 0 point is not measured; G^R(t=0) = 0
  identically because Z_j and Z_c commute at equal times.

## Tutorial parameter sets (for reproduction; label results accordingly)

| | small (statevector) | large (QPU) |
|---|---|---|
| n_qubits | 10 | 50 |
| dt | 0.6 | 0.6 |
| n_steps | 10 | 20 |
| AQC checkpoints (1-layer + 2-layer) | 3 + 2 | 6 + 4 |
| HVA layers | 3 | 5 |
| MPS max bond | 32 | 128 |
| MPS cutoff | 1e-8 | 1e-8 |
| Execution | StatevectorEstimator | EstimatorV2 on Heron, DD XY4, gate twirling, TREX |

## What is measured, and why it is G^R

Circuit for time step m: |GS> -> Rz(pi/2) on `center` -> U_Trotter(m dt) -> measure <Z_j> for all j.

Rz(pi/2) = exp(-i pi/4 Z) = (I - i Z_c)/sqrt(2). Expanding U^dagger Z_j(t) U gives exactly

  <Z_j>_meas(t) = -(i/2) <[Z_j(t), Z_c]>  +  (1/2)( <Z_j(t)> + <Z_c Z_j(t) Z_c> ),

all expectations in the prepared reference state. The first term is G^R in Pauli units
(real valued). The second "background" term vanishes when the reference state is an eigenstate
of the global spin flip P = prod_i X_i (P commutes with H, and P anticommutes with each Z). The
exact ground state for even n satisfies this. Note this is a slightly stronger requirement than
"zero magnetization", which is how the tutorial phrases it.

Verified by us (exact diagonalization, n = 8, t = 1.7): measured <Z_j> equals G^R to 1e-10 for
the exact GS; with a reference state that mixes P-parity, they differ (about 0.02 absolute at one
site). Consequence: the identity "raw <Z_j> = G^R, no subtraction" is exact only if the prepared
state (HVA output, or AQC output, which does not enforce symmetry) is P-symmetric. Hardware noise
also breaks it. Required diagnostic for every run: report the deviation from P-symmetry, for
example via a control run without the Rz perturbation (which measures <Z_j(t)> directly and
should be zero), and say whether the background term was subtracted.

Unit conversion: G^R in spin units is 1/4 of the Pauli-unit value (two spin operators). The
`/ 4` in `get_dsf` implements exactly this.

## Assembly: from circuits to S(q, w)

1. Build circuits (stage modules). Output: a list `all_circuits` of length n_steps, index m-1
   is time m * dt. Steps 1..K are AQC circuits, steps K+1..n_steps are AQC(K) + Strang(m - K).
2. Observables: `SparsePauliOp.from_sparse_list([("Z", [i], 1)], num_qubits=n)` for each i.
   On hardware, apply `obs.apply_layout(isa_circuit.layout)` per circuit. Never reuse a layout
   from a different transpiled circuit.
3. Execute: one PUB per time step, `(circuit, observables)`.
4. `rgf_mat = np.stack([r.data.evs for r in result])`, shape (n_steps, n_qubits), row m-1 is time
   m dt, column j is site j. Assert the shape before calling `get_dsf`.
5. `get_dsf(n_qubits, rgf_mat, dt, n_steps, n_q, n_w)` as in the tutorial.

### What `get_dsf` actually computes (do not misdescribe it)

- Frequency grid: w in linspace(0, pi/dt, n_w). pi/dt is the Nyquist frequency of the sampled
  time series (about 5.24 for dt = 0.6).
- Time transform: G(j, w) = (dt/4) * sum_{m=1}^{n_steps} G^R_pauli(t_m, j) exp(i w t_m). This is a
  plain rectangle-rule sum with a hard cutoff at T = n_steps * dt: no window, no damping, no
  zero padding. Expect sinc ringing and finite resolution of order 2 pi / T (about 0.52 at T = 12,
  about 1.05 at T = 6).
- Space transform: sum over j of G(j, w) exp(-i q (j - center)), q in linspace(0, 2 pi, n_q).
  Only the single reference site `center` is used ("center-site approximation"); there is no
  average over reference sites, and OBC breaks translation invariance.
- Output: -(Im[...] symmetrized under q -> 2 pi - q), then clipped at zero. The symmetrization is
  exact on this grid because linspace(0, 2 pi, n_q) is symmetric about pi. The clip hides
  negative lobes from truncation and noise; when assessing quality, inspect the unclipped array
  too.
- Normalization: the result is proportional to the T = 0 DSF for w > 0 via the
  fluctuation-dissipation relation (S proportional to -Im G^R), with a convention-dependent
  constant that the code does not carry. `plot_dsf` divides by the max. Do not quote absolute
  intensities from this pipeline.
- Temperature: the simulation is T = 0. The KCuF3 INS data are at 6 K. State this when
  comparing.

### Physics checks for the final S(q, w)

- Two-spinon continuum for H = J sum S.S (our convention): lower edge w_L(q) = (pi/2) J |sin q|,
  upper edge w_U(q) = pi J |sin(q/2)|. Spectral weight should concentrate near q = pi at low w
  and lie mostly between these curves, up to the resolution above. Overlay both curves on every
  DSF plot. If you compare with the paper's axes, apply the factor-of-2 J conversion first.
- Symmetry about q = pi is enforced by construction, so it is not evidence of correctness.
- Before any hardware result is described as agreeing with theory, run the same `get_dsf` on
  (a) exact diagonalization or exact statevector evolution at n = 10 to 12, and (b) the
  uncompressed Trotter circuits, so that Trotter error, AQC error, finite-T truncation, and
  hardware error can be separated.

## Output and provenance

- Save `rgf_mat`, the unclipped DSF array, all parameters, package versions
  (qiskit, qiskit-ibm-runtime, qiskit-addon-aqc-tensor, quimb, jax), backend name, and job IDs
  together, for example in one `.npz` plus a JSON sidecar.
- Hardware runs: record transpiled two-qubit depth and count per circuit, and all Estimator
  options. Tag jobs.

## Dependencies

`pip install qiskit qiskit-ibm-runtime 'qiskit-addon-aqc-tensor[quimb-jax]'` The tutorial also imports
`qiskit_quimb`; if the extra does not pull it in, install `qiskit-quimb`. Pin versions in a lockfile once a working set is
found; the AQC addon API has changed across releases.
