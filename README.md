# qins — multi-agent benchmark of quantum simulations of inelastic neutron scattering

Compute the dynamical structure factor S(q, ω) of spin-chain materials on IBM quantum
hardware, a noisy-MPS model and classical DMRG, and benchmark all three against each
other and against experimental INS data. Four models are included: the XX chain, KCuF₃,
a two-soliton chain and CsCoX₃.

The workflow follows the IBM tutorials
([simulate-neutron-scattering](https://quantum.cloud.ibm.com/docs/en/tutorials/simulate-neutron-scattering),
[serverless version](https://quantum.cloud.ibm.com/docs/en/tutorials/simulate-neutron-scattering-with-a-serverless-workflow))
and arXiv:2603.15608. It adds three things on top of them:

* **Hardware calibration:** a calibration-aware chain layout across several backends.
* **Generalized materials:** any XXZ chain with fields, with parameters optionally taken from papers through an LLM.
* **Dynamic error-mitigation selection:** DD, PT, TREX and ZNE are chosen per circuit from its features.

## Install

```bash
uv venv --python 3.12 && uv pip install -e ".[ui,dev,notebook]"
```

## Three ways to run

```bash
# CLI
qins run -c quick                                   # ~2 min smoke run, 4 models x 3 backends
qins run -c default -s system.n_sites=30 -s 'models=[kcuf3,cscox3]'
qins run -c quick --resume runs/quick_20261015_222154   # reuse finished tasks
qins agents | qins dag -c quick | qins show runs/<run> --events 30

# Streamlit
qins ui                                             # = streamlit run app/streamlit_app.py
```

```python
# Python
from qins import Pipeline, load_config
res = Pipeline(load_config("quick", ["models=[kcuf3]"])).run()
print(res.run_dir, res.status)
```

## The agent DAG

Each agent is one small file in `qins/agents/`. The scheduler (`qins/dag.py`, about 150
lines) expands each agent into tasks, one per **model** or one per **backend**. A task
starts as soon as its own upstream tasks finish. Models therefore pipeline independently:
KCuF₃ can already be on hardware while CsCoX₃ is still compiling.

```
a00 model spec ──┬─> a01 MPS/DMRG ───────────────────────────────┐
  (per model)    └─> a02 GS ansatz ─> a03 AQC ─> a07 assembly ─┬─> a05 noise sweep ─┐
a04 HW calibration (per backend) ──────────────────────────────┴─> a08 HW execution ─┴─> a09 DSF ─> a10 benchmark
a06 metric self-test ─────────────────────────────────────────────────────────────────────────────────┘
```

| agent | what it does | tool |
|---|---|---|
| a00 model spec | Hamiltonian parameters per material: preset → config → optional LLM proposal from `Papers/` | pypdf + Muse-Glimmer (vLLM) |
| a01 MPS/DMRG | χ-sweep convergence, ED cross-check, static correlations, fine-step reference RGF | quimb |
| a02 GS ansatz | VQE of the HVA, singlet vs. Néel branches, Fig. S2 sweep (fidelity vs. N × layers) | Qiskit + COBYQA |
| a03 AQC | tensor-network compression of GS + kick + t Trotter steps | qiskit-addon-aqc-tensor |
| a04 HW calibration | device characterization, best qubit chain, layer-fidelity estimate, optional LF-RB | fake / JSON / live; chain search, Qiskit VF2 |
| a05 noise model | depolarizing sweep (7.5e-3 → 1e-3) on the same circuits as the hardware | Aer MPS |
| a06 metrics | MSE, Wasserstein, SSIM, nQFI and two-tangle, self-tested on synthetic spectra | numpy/scipy |
| a07 assembly | GS + AQC block + Trotter tail; gate-count table; noiseless circuit simulation | Qiskit |
| a08 HW execution | backend choice, **mitigation selection** (rules or Muse-Glimmer), local emulator or IBM EstimatorV2 | qiskit-ibm-runtime |
| a09 DSF | the same Fourier estimator applied to every RGF source | numpy |
| a10 benchmark | every source compared with the MPS reference and with INS data (if present); optional LLM write-up | a06 metrics + Muse-Glimmer (vLLM) |

### Places where this design differs from `Agent division.pdf`

1. **a00 is a separate agent.** The PDF has agent 1 build the Hamiltonian, but agents 2, 3 and 5 also need it. Splitting it out lets agents 1, 2, 4 and 6 run in parallel.
2. **a03 AQC depends on a02.** AQC-tensor compresses *states*, so it needs the ground-state circuit as its initial state. It still fans out per model, so the four AQC jobs run in parallel.
3. **a05 depends on a07.** The noisy-MPS sweep simulates exactly the circuits sent to hardware, so the hardware vs. noisy-MPS comparison is like for like. It runs at the same time as a08. To make it independent as in the PDF, give it its own GS + Trotter circuits and change `deps` in `a05_noise_model.py`.
4. **Signed kicks for magnetized states.** A single Rz(π/2) kick gives the retarded G only when ⟨Z_j⟩ = 0. With a field, or with a Néel-like VQE state, a non-retarded term gets mixed in. a02 measures max|⟨Z_j⟩| of the prepared state and, when it is nonzero, the circuits use both ±π/2 kicks with G = (z₊ − z₋)/2, which is exact. This costs 2× circuits, only for those models. a01 always uses both, since that is cheap classically. See `kick_signs` in `qins/physics/circuits.py`.

## Logs and metadata (for debugging and for the paper)

Each run writes everything to `runs/<run>/`:

| file | contents |
|---|---|
| `manifest.json` | resolved config, package versions, host, argv, SHA-256 of every input file (papers, calibrations, INS data), task graph, per-task status, duration, seed and artifact hashes |
| `events.jsonl` | one JSON line per event: `task_start/end/skip/cached`, `artifact` (path + sha256 + shape), `metric`, `decision` (choice + reasons + features), `llm_call`, `warning`, `error` (with traceback) |
| `run.log` | the same events as human-readable lines |
| `llm_calls.jsonl` | full request and response, latency and token usage of every LLM call |
| `config.yaml`, `dag.mmd` | exact config and a Mermaid diagram of the DAG |
| `tasks/<agent>[<key>]/` | `result.json` plus all artifacts (`.npz` arrays, `.qpy` circuits, `.json` tables, `.png` figures) |
| `report.md`, `metrics_all.csv` | summary table across models and sources |

Each task gets a deterministic seed derived from `run.seed` and its task id. That seed is logged.

## LLM (local Muse-Glimmer-30B via vLLM)

The LLM is advisory and off by default. Its output is validated, rules are the fallback, and every call is logged.
It is used in three places:

| agent | LLM task | validation / fallback |
|---|---|---|
| a00 model spec | propose Hamiltonian parameters from `Papers/` excerpts | range checks; applied only if `model_spec.accept_llm=true` |
| a08 HW execution | choose DD / PT / TREX / ZNE | schema check; falls back to the rule policy |
| a10 benchmark | write a short interpretation of the metrics table, given `.claude/rules/dsf-assembly.md` as context | must name existing sources; added to `report.md` |

```bash
bash scripts/serve_muse.sh        # /home/pza/local_models/Muse-Glimmer-30B, fp8, GPU 2, port 8000
qins run -c quick -s llm.enabled=true -s execution.mitigation.policy=llm -s model_spec.accept_llm=true
```

* The server uses the vLLM in `/home/pza/venvs/llmserve` (it knows the `muse_glimmer` architecture). bf16 weights are
  about 60 GB, so the script quantises to fp8 to fit one 48 GB GPU. Pick the GPU with `GPU=1`, the port with `PORT=8001`.
* Any OpenAI-compatible server works: set `llm.base_url` and `llm.model`. `scripts/serve_vllm.sh` serves other models.
* `llm.reasoning` (`low | medium | high`) sets Muse's "Reasoning strength" line; `llm.temperature` defaults to 0 for reproducibility.
* All agents share one server, and vLLM batches concurrent requests.
* `bash scripts/qins_ui.sh` starts the server and the Streamlit UI together and stops the server when the UI exits,
  on Ctrl-C, or when the terminal (or VS Code) is closed. `LLM=0` starts the UI only. Closing just the browser tab does
  not stop Streamlit. A server that was already running is reused and left running.

## Real hardware

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(token="...", channel="ibm_quantum_platform")
```

```bash
qins run -c ibm_hardware        # calibration_source=ibm, execution.mode=ibm
```

`execution.mode=local_emulator` (the default) runs the chosen chain's calibrated noise
model in Aer. TREX is emulated by readout inversion. DD, PT and ZNE are *recorded* as
selected but not emulated, and the run logs a warning saying so.

## Adding a material

Add a preset to `qins/physics/models.py`, or add the material in config:

```yaml
models: [kcuf3, cuso4]
model_overrides:
  cuso4: {jxy: 1.0, jz: 1.0, energy_scale_meV: 0.27, material: "CuSO4·5H2O", search_terms: [CuSO4, copper sulfate]}
```

Then drop the relevant PDFs into `Papers/` and, optionally, INS data into `data/ins/<model>.npz` (see `data/ins/README.md`).

## Tests

```bash
pytest -q        # 25 tests, ~5 s: DMRG vs ED, Trotter vs expm, MPS vs statevector, DSF peak,
                 # metrics, DAG failure/skip/resume, LLM client vs mock server, mitigation, layout
```

## Caveats

* The parameters of the `two_soliton` and `cscox3` presets are placeholders. Verify them, or let a00 extract them from papers, before quoting results.
* The two-tangle computed from a measured S^zz assumes SU(2) symmetry. That is exact for KCuF₃ and only approximate for the anisotropic chains; a10 also reports the exact DMRG value.
* `mapomatic` has no adapter for Qiskit 2.x yet; selecting it falls back to `chain_search`, and the fallback is logged.
