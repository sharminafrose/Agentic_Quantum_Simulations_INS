---
paths:
  - "qins/agents/a03_aqc.py"
  - "qins/agents/a07_assembly.py"
  - "qins/physics/circuits.py"
---

# Stage 2: Trotter time evolution and AQC compression

Read `dsf-assembly.md` (conventions) and `state-prep.md` (where `perturbed` comes
from) first.

## Scope

Input: `perturbed` (HVA + Rz on `center`), H parameters, dt, n_steps.
Output: `all_circuits`, length n_steps, where index m-1 prepares the state at time m * dt.
Also produces the uncompressed reference list `circuits` (needed for validation, keep it).

## Second-order Trotter (`trotter_evolution`)

- Bond gate: `PauliEvolutionGate(pair_ham, time=tau)` with
  pair_ham = 0.25 J eps XX + 0.25 J eps YY + 0.25 J ZZ (equals the bond term of H in Pauli form).
- Brickwork, OBC, even n: "even" bonds (2i, 2i+1), i < n/2; "odd" bonds (2i+1, 2i+2), i < n/2 - 1.
- m steps of Strang splitting with merged half steps:
  half-even, odd, [even, odd] x (m-1), half-even.
  So each sublayer appears with total time m * dt. Error is O(dt^2) per unit time for the
  evolution operator (second order).
- The function is a generator of `CircuitInstruction`s; it returns nothing for n_steps = 0.
- The tutorial notes the paper used a hand-optimized Trotter circuit; PauliEvolutionGate
  transpiled at `optimization_level=3` gives the same 2q gate count and depth (tutorial claim,
  verify before repeating it). Per the tutorial, Qiskit synthesizes each bond gate into an optimal 3-CNOT decomposition.
- Uncompressed circuit for step m: `perturbed.copy()` then append `trotter_evolution(..., m)`.

## AQC compression (qiskit-addon-aqc-tensor)

Purpose: replace the first K Trotter steps, including the HVA and Rz, by a shallower
parameterized circuit whose parameters maximize MPS fidelity with the deep target.

Two stages of checkpoints, K = K1 + K2:
- steps 1..K1 use the "one-layer" ansatz, built by `generate_ansatz_from_circuit` from target
  circuit 1;
- steps K1+1..K use the "two-layer" ansatz, built from target circuit 2.
Both calls use `qubits_initially_zero=True` and are given the circuit after
`decompose(["PauliEvolution"])` (the MPS backend does not accept PauliEvolutionGate). The
ansatz structure is derived from the two-qubit gate layout of the input circuit; check the
addon docs for the exact construction before describing it in writing.

Simulator:
```python
aqc_sim = QuimbSimulator(
    quimb_circuit_factory=partial(qtn.CircuitMPS,
                                  gate_opts=dict(max_bond=chi, cutoff=1e-8)),
    autodiff_backend="jax",
)
target_mps[k] = tensornetwork_from_circuit(circuits[k-1].decompose(["PauliEvolution"]), aqc_sim)
```

Optimization per checkpoint k:
- `obj = MaximizeStateFidelity(target_mps[k], ansatz, aqc_sim)`;
  `scipy.optimize.minimize(obj.loss_function, x0, method="L-BFGS-B", jac=True,
  options=dict(maxiter=100))`. Reported fidelity = 1 - result.fun.
- Warm start: x0 = optimized params of k-1 only if k-1 is in the same stage; at the stage
  boundary (k = K1+1) and at k = 1, start from the generator's initial params.
- Record per k: MPS fidelity, `target_mps[k].psi.max_bond()`, whether it saturated chi,
  `result.success`, iterations, wall time.

Interpretation rules:
- The fidelity is against the bond-truncated target MPS. When the target bond dimension hits
  chi (the tutorial's n = 50 run saturates chi = 128 from k = 8 on), the target itself is
  approximate and the reported fidelity is only a proxy, not the fidelity to the exact
  Trotter state. Say so whenever such numbers are reported.
- Fidelity is with respect to the Trotter state (which already includes HVA error), not the
  exact time-evolved ground state.
- Two-layer fidelities may come out below one-layer ones at the same n because they are used
  for later, more entangled times. This is not a regression by itself; compare at equal k only.
- The AQC ansatz does not enforce SU(2) or spin-flip symmetry. The resulting state can mix
  P-parity, which makes the background term in `dsf-assembly.md` nonzero. Measure it (control run
  without Rz, or evaluate <Z_j> and <Z_c Z_j Z_c> on the MPS) for every AQC checkpoint and
  report the maximum over j.

Tutorial reference values (labels only, not ours until reproduced):
- n = 10, chi = 32: fidelities 1.000, 0.998, 0.989 (1-layer), 0.998, 0.996 (2-layer). Final 2q
  depth 163 (Trotter) vs 99 (AQC + Trotter), uniform basis, step 10.
- n = 50, chi = 128: fidelities fall from 1.000 to 0.815 (k = 6, 1-layer) and 0.67 to 0.77
  (k = 7..10, 2-layer). Step 20: 2q depth 307 vs 171. Wall time about six hours, dominated by the
  two-layer checkpoints.

## Mixed AQC + Trotter circuits

```python
all_circuits = [aqc_circuits[k] for k in range(1, K + 1)]
base = aqc_circuits[K]
for m in range(1, n_steps - K + 1):
    c = base.copy()
    for instr in trotter_evolution(c.qubits, J, eps, dt, m):
        c.append(instr)
    all_circuits.append(c)
```
- Because the merged half-steps at the junction recombine, Strang(K) followed by Strang(m) equals
  Strang(K + m) exactly (derived). So circuit K + m targets the same Trotter state as
  `circuits[K + m - 1]`, with the first K steps replaced by the AQC approximation.
- AQC error at checkpoint K propagates unitarily into all later steps; there is no further
  compression after K.

## Depth accounting

- `uniform_2q_depth`: transpile at `optimization_level=0` to basis ["cz","id","rz","sx","x"],
  count depth over 2q gates only. Use it for all Trotter vs AQC comparisons.
- Hardware depth is different: report the ISA depth after
  `generate_preset_pass_manager(optimization_level=3, backend=backend)` separately and label it.
  The tutorial's n = 50 run on ibm_fez gave max ISA 2q depth 105 (2574 2q gates); tutorial values
  only.

## Tests for this stage

1. Trotter correctness: for n <= 12, compare the statevector of `circuits[m-1]` against
   expm(-i H m dt) applied to the `perturbed` statevector; check the infidelity scales as dt^4
   at fixed total time (second-order Trotter, state infidelity goes as the square of the O(dt^2)
   error). Report the measured exponent.
2. Generator edge cases: n_steps = 0 yields nothing; for m steps the total evolution time on
   every bond equals m * dt (half layers counted as 1/2).
3. AQC at n = 10: compute the exact statevector fidelity of each `aqc_circuits[k]` with
   `circuits[k-1]` and compare with the MPS-level number.
4. Junction identity: for small n, statevector of AQC(K) + Strang(m) with AQC replaced by the
   exact Trotter(K) circuit equals Strang(K+m).
5. Observable-level check (the one that matters for the DSF): rgf from `all_circuits` vs from
   `circuits` on StatevectorEstimator at n = 10; report max abs difference per time step.

## Practical notes

- At n = 50, chi = 128, JAX/XLA may print "Very slow compile" messages lasting a few minutes on
  first trace. The tutorial reports these as benign. Do not kill the job for that alone.
- The companion Serverless tutorial offloads AQC to Qiskit Serverless; consider it, or an HPC
  node, for n = 50.
- Save optimized AQC parameters per k (npz) and the circuits (QPY) so checkpoints can be reused
  without re-optimizing.
