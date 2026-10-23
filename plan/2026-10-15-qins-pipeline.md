# qins pipeline — design notes and open items (2026-10-15)

## Status
- 11 agents (a00–a10) implemented, DAG scheduler (thread / process), CLI, Streamlit, Python API.
- Verified: `qins run -c quick` → 40/40 tasks OK (4 models × 3 backends), 25 unit tests pass,
  LLM path tested end-to-end against a mock OpenAI-compatible server.

## Decisions to confirm with the team
1. a00 split out of agent 1 (Hamiltonian spec needed by 1, 2, 3, 5).
2. a03 AQC depends on a02 (AQC compresses states → needs the GS circuit).
3. a05 noisy-MPS simulates the *assembled hardware circuits* (depends on a07), runs parallel to a08.
4. Magnetised ground states (fields, or a Néel-like VQE state; decided by a02 from max|⟨Z_j⟩|) use ±π/2 kicks,
   G = (z₊ − z₋)/2 (exact); doubles circuits only for those models. a01 always uses both (cheap).
5. Presets for `two_soliton` (jxy = 0.3) and `cscox3` (jxy = 0.14, hs = 0.05, J = 13 meV) are placeholders.

## Open items
- [ ] Real INS data → `data/ins/<model>.npz` (KCuF₃ 6 K data from arXiv:2603.15608 first).
- [ ] Fix NVIDIA driver/library mismatch (`nvidia-smi` fails) before serving vLLM; then try
      Llama-3.1-8B vs Mistral-Small-24B for a00 parameter extraction quality.
- [ ] Shot budget for CsCoX₃: |G| ≈ 0.02–0.03, so ≥ 20k shots needed for SNR > 3 (quick config uses 1k).
- [ ] mapomatic adapter for Qiskit 2.x (currently falls back to chain_search, logged).
- [ ] Live layer-fidelity RB needs `uv pip install qiskit-experiments` + `calibration_source=ibm`.
- [ ] Obsidian project memory not bootstrapped for this repo (run `/obsidian-init` if wanted).

## Suggested additions to `Papers/` (verify each with citation-verification before citing)
- KCuF₃: Lake, Tennant, Frost, Nagler, Nat. Mater. 4, 329 (2005); Scheie et al., Nat. Commun. 13, 5796 (2022).
- Entanglement witnesses from INS: Scheie et al., PRB 103, 224434 (2021); Hauke et al., Nat. Phys. 12, 778 (2016).
- XX / Heisenberg DSF: Müller, Thomas, Beck, Bonner, PRB 24, 1429 (1981).
- Two-soliton continuum (Ising-like XXZ): Ishimura & Shiba, Prog. Theor. Phys. 63, 743 (1980).
- CsCoX₃ confinement / Zeeman ladder: Shiba, Prog. Theor. Phys. 64, 466 (1980); Villain, Physica B 79, 1 (1975);
  Goff, Tennant, Nagler, PRB 52, 15992 (1995).
- Quantum DSF: Chiesa et al., Nat. Phys. 15, 455 (2019); Baez et al., PNAS 117, 26123 (2020).
- AQC: Robertson et al., arXiv:2301.08609.
- Hardware/mitigation: McKay et al., arXiv:2311.05933 (layer fidelity); Nation & Treinish, PRX Quantum 4, 010327 (2023)
  (mapomatic); van den Berg, Minev, Temme, PRA 105, 032620 (2022) (TREX); Temme, Bravyi, Gambetta, PRL 119, 180509
  (2017) and Kim et al., Nature 618, 500 (2023) (ZNE).
