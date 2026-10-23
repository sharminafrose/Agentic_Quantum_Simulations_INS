---
paths:
  - "qins/agents/a01_mps_dmrg.py"
  - "qins/agents/a02_gs_ansatz.py"
  - "qins/physics/mps.py"
  - "qins/physics/circuits.py"
---

# Stage 1: ground-state preparation and perturbation

Read `dsf-assembly.md` first for conventions (H, units, `center`, even n).

## Scope

Produces three things:
1. `dmrg.state`: reference ground-state MPS and energy from DMRG (classical, quimb).
2. `gs_circuit`: Hamiltonian variational ansatz (HVA) circuit with optimized parameters.
3. `perturbed`: `gs_circuit` followed by `rz(pi/2, center)`.

Important coupling: in the tutorial workflow, `gs_circuit` is NOT what runs on hardware for the
compressed time steps. The AQC targets in Stage 2 are the full circuits (HVA + Rz + Trotter), so
AQC re-compiles the state preparation as well. `gs_circuit` reaches hardware only through the
targets it defines. Any HVA infidelity is therefore inherited by every AQC target, and AQC
cannot recover it.

## Hamiltonian MPO

`xxz_hamiltonian_mpo(n, J, eps)` via `qtn.SpinHam1D(S=1/2)`:
- `+= J*eps*0.5, "+", "-"`, `+= J*eps*0.5, "-", "+"`, `+= J, "Z", "Z"`, then `build_mpo(L=n)`.
- 0.5 (S+S- + S-S+) = Sx Sx + Sy Sy, and quimb's spin-1/2 "Z" is Sz, so this matches H in `dsf-assembly.md`.
- Regression test: for n = 10, J = eps = 1, E0 = -4.258035207 (verified by our exact
  diagonalization). Keep a pytest that checks DMRG against dense ED for n <= 12 to 1e-8.

## DMRG reference

- `qtn.DMRG2(ham_mpo).solve(tol=1e-8)`; record final energy, max bond, and whether it converged.
- At n = 50 the tutorial reports E0 = -21.972109. Treat as a sanity value, not a test, unless we
  reproduce it with our own run and record the bond dimensions used.

## HVA circuit (`build_ground_state_ansatz(n, n_layers)`)

- Initial state: singlets on pairs (2i, 2i+1) via `x(2i); x(2i+1); h(2i+1); cx(2i+1, 2i)`.
  This yields (|01> - |10>)/sqrt(2) on each pair (checked by hand: |11> -> |-,1> -> singlet).
- Each layer r: "odd" bonds (2i-1, 2i) with angle theta[2r], then "even" bonds (2i, 2i+1) with
  angle theta[2r+1]. Gate: `PauliEvolutionGate(0.5*(XX+YY+ZZ), time=theta)`.
- 0.5 (XX+YY+ZZ) = SWAP - I/2, so the gate is a SWAP up to global phase at theta = pi/2 and the
  identity at theta = 0. The initialization x0 = tile([0, pi/2]) + N(0, 0.1) uses this.
- Parameter count is 2 * n_layers. Ordering: theta[2r] odd layer, theta[2r+1] even layer.
- Symmetry facts (derived): the singlet product state is a total-spin singlet and every gate is
  SU(2) invariant, so the HVA output is an exact SU(2) singlet for any theta. Hence it is also
  P-symmetric (global spin flip), so the background term in `dsf-assembly.md` vanishes for this
  state in the absence of noise. Keep this property; do not add symmetry-breaking gates to the
  ansatz without updating the measurement analysis in `dsf-assembly.md`.

## HVA optimization (`optimize_ground_state_ansatz`)

- Objective: 1 - |<psi_HVA(theta)|psi_DMRG>|^2, where psi_HVA is simulated with
  `quimb_circuit(..., quimb_circuit_class=qtn.CircuitMPS, max_bond, cutoff)` after
  `decompose(["PauliEvolution"])`.
- Optimizer: `scipy.optimize.minimize(method="COBYQA", options={"maxiter": 100})`, gradient free.
- Seed: `np.random.default_rng(12345)` in the tutorial. Record the seed for every run.
- Report both the MPS fidelity and the energy <H> of the optimized state
  (`qtn.expec_TN_1D(psi.H, mpo, psi)`), together with chi and cutoff. Report the relative energy
  error (E_HVA - E0)/|E0|.
- Tutorial reference values (not ours until reproduced): n = 10, 3 layers: fidelity 0.984,
  E = -4.2326. n = 50, 5 layers: fidelity 0.646, E = -21.6167.
- The fidelity drop with n is expected heuristically: at fixed per-site accuracy, a global
  many-body fidelity generically decays exponentially in system size. Report the energy error per site alongside the global fidelity
  so that the two are not conflated. Increasing `n_layers` or `maxiter` is the documented lever.
- Do not claim the optimizer converged unless `result.success` is true and the fidelity is stable
  under a restart from a different seed.

## Perturbation

- `perturbed = gs_circuit.copy(); perturbed.rz(np.pi / 2, center)`.
- Rz(pi/2) = (I - i Z_c)/sqrt(2) exactly (global phase convention of Qiskit's Rz:
  Rz(phi) = exp(-i phi Z / 2)). Do not replace it with `s` or `sdg`: those differ by a global
  phase only, which is harmless, but a sign flip (Rz(-pi/2)) flips the sign of the measured G^R.

## Tests for this stage

1. ED vs DMRG energy, n in {6, 8, 10, 12}.
2. HVA output: <Z_j> = 0 for all j, and total S^2 = 0, at random theta (exact statevector, n <= 12).
3. Optimized HVA at n = 10: MPS fidelity agrees with statevector fidelity |<psi_HVA|psi_ED>|^2
   to within the truncation error at chi = 32.
4. Rz(pi/2) on `center`: compare the full statevector against (I - i Z_c)/sqrt(2) applied directly.

## Outputs to persist

`theta_opt`, `n_layers`, seed, chi, cutoff, MPS fidelity, E_HVA, E0 (DMRG), and the QPY of
`perturbed`.
