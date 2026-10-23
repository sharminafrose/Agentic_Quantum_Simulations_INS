# Experimental INS spectra

Drop one file per model, named after the model key (`kcuf3.npz`, `cscox3.npz`, ...), with

| array       | shape            | meaning                                  |
|-------------|------------------|------------------------------------------|
| `q`         | (n_q,)           | momentum transfer along the chain (rad, 0..2pi) |
| `omega_meV` | (n_omega,)       | energy transfer (meV)                    |
| `S`         | (n_omega, n_q)   | measured S(q, omega), background-subtracted |

```python
np.savez("data/ins/kcuf3.npz", q=q, omega_meV=e, S=s)
```

Agent a10 picks these up automatically, interpolates every simulated spectrum
onto the experimental grid and adds `reference = experiment_INS` rows to the
metric tables. The SHA-256 of each file is recorded in the run manifest.
