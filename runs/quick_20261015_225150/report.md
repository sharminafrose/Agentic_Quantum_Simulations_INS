# Run report: quick_20261015_225150

- wall time: 22.0 s
- models: kcuf3
- backends: ibm_kingston
- execution mode: local_emulator

## Tasks

| task | status | duration (s) |
|---|---|---|
| a00_model_spec[kcuf3] | ok | 0.235 |
| a01_mps_dmrg[kcuf3] | ok | 1.057 |
| a02_gs_ansatz[kcuf3] | ok | 2.553 |
| a03_aqc[kcuf3] | ok | 6.374 |
| a04_hw_calibration[ibm_kingston] | ok | 0.208 |
| a05_noise_model[kcuf3] | ok | 7.284 |
| a06_metrics[all] | ok | 0.223 |
| a07_assembly[kcuf3] | ok | 0.677 |
| a08_hw_execution[kcuf3] | ok | 10.7 |
| a09_dsf[kcuf3] | ok | 0.283 |
| a10_benchmark[kcuf3] | ok | 0.244 |

## Metrics

| model   | source                 | reference     |       mse |   wasserstein |   ssim |   nqfi_q_pi |   two_tangle |
|:--------|:-----------------------|:--------------|----------:|--------------:|-------:|------------:|-------------:|
| kcuf3   | mps_reference          | mps_reference | 0         |       0       | 1      |       2.864 |       0.185  |
| kcuf3   | noiseless_circuit      | mps_reference | 0.0002789 |       0.07154 | 0.9653 |       3.008 |       0.1833 |
| kcuf3   | noisy_p=0.005          | mps_reference | 0.0003742 |       0.2288  | 0.8783 |       2.705 |       0.1462 |
| kcuf3   | noisy_p=0.001          | mps_reference | 0.000409  |       0.1611  | 0.9095 |       2.892 |       0.1587 |
| kcuf3   | hardware(ibm_kingston) | mps_reference | 0.0002104 |       0.192   | 0.8985 |       2.894 |       0.1232 |
