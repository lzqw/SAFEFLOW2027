# STAR P4.6 direction-loss weight Pareto final report

Conclusion: **partial_tradeoff_only**

Protocol: `p4.6-direction-weight-pareto-v1` (`c473da2ed99b31cde673a1cd038ec0477053b23ac63a130376ebccc4271ad93e`)

C0 and C1 are checksum-verified P4.5 evidence reused without retraining. C025 and C050 are the six new 100k runs. Every final seed is shown below.

| arm | lambda | seed | return | success | mean cost | CVaR20 | dangerous v_n+ | path | stall | clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 0.00 | 2027 | 1.076536 | 0.450 | 16.250000 | 63.750000 | 0.176915 | 3.555917 | 0.050 | 0.023762 |
| C0 | 0.00 | 2028 | 0.386441 | 0.250 | 17.650000 | 81.250000 | 0.142824 | 2.861478 | 0.100 | 0.030412 |
| C0 | 0.00 | 2029 | 0.729578 | 0.200 | 13.750000 | 65.250000 | 0.157987 | 3.679286 | 0.050 | 0.042850 |
| C025 | 0.25 | 2027 | 0.111680 | 0.400 | 18.650000 | 61.500000 | 0.169529 | 6.340397 | 0.050 | 0.068025 |
| C025 | 0.25 | 2028 | 0.297083 | 0.150 | 10.750000 | 48.750000 | 0.176742 | 2.461359 | 0.150 | 0.016663 |
| C025 | 0.25 | 2029 | -0.712105 | 0.300 | 21.950000 | 83.500000 | 0.143176 | 6.615177 | 0.100 | 0.045437 |
| C050 | 0.50 | 2027 | -0.746077 | 0.000 | 9.400000 | 47.000000 | 0.085085 | 2.270942 | 0.000 | 0.008975 |
| C050 | 0.50 | 2028 | 0.375707 | 0.200 | 13.050000 | 55.250000 | 0.153138 | 2.673715 | 0.200 | 0.012375 |
| C050 | 0.50 | 2029 | -0.034428 | 0.150 | 3.100000 | 15.500000 | 0.115598 | 4.689527 | 0.100 | 0.026863 |
| C1 | 1.00 | 2027 | 1.228598 | 0.450 | 10.650000 | 49.500000 | 0.180419 | 3.182057 | 0.100 | 0.032537 |
| C1 | 1.00 | 2028 | -1.457632 | 0.050 | 15.200000 | 65.750000 | 0.114308 | 6.839948 | 0.000 | 0.124925 |
| C1 | 1.00 | 2029 | 0.451291 | 0.200 | 0.000000 | 0.000000 | 0.113892 | 2.368723 | 0.250 | 0.024837 |

## Pooled 60-episode Pareto points

| arm | lambda | return | success | mean cost | CVaR20 | dangerous v_n+ | path | stall | training cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 0.00 | 0.730852 | 0.300 | 15.883333 | 70.083333 | 0.159242 | 3.365560 | 0.067 | 14100 |
| C025 | 0.25 | -0.101114 | 0.283 | 17.116667 | 66.500000 | 0.163149 | 5.138978 | 0.100 | 8348 |
| C050 | 0.50 | -0.134932 | 0.117 | 8.516667 | 41.750000 | 0.117940 | 3.211395 | 0.100 | 10262 |
| C1 | 1.00 | 0.074086 | 0.233 | 8.616667 | 43.083333 | 0.136207 | 4.130243 | 0.117 | 9972 |

## Locked gates

### C025 (`lambda=0.25`)

- Safety gate: False.
- Performance gate: False (return loss +0.831966, success loss +0.016667).
- Seed robustness gate: True (2028 path/C0 0.860, 2029 stall increase +0.050).
- Fully qualified: False.

### C050 (`lambda=0.50`)

- Safety gate: True.
- Performance gate: False (return loss +0.865784, success loss +0.183333).
- Seed robustness gate: True (2028 path/C0 0.934, 2029 stall increase +0.050).
- Fully qualified: False.

Selected arm: `None`; selected weight: `None`; tie break: `not_applicable`.

## Evidence and figures

All new final full-replay checkpoints received the unchanged 64 x 256 A/B/C audit. The combined CSVs retain all checkpoints, 5k diagnostics, fixed-seed evaluations, checkpoint hashes, and trajectory paths.

- `p4_6_return_success.png`
- `p4_6_cost_cvar_training_cost.png`
- `p4_6_geometry.png`
- `p4_6_critic_scales.png`
- `p4_6_final_pareto.png`
- `p4_6_trajectory_eval_seed_12027.png`
- `p4_6_trajectory_eval_seed_12036.png`
- `p4_6_trajectory_eval_seed_12046.png`

Large checkpoints and replay remain server-only. `p4_6_checkpoint_sha256.csv` records their immutable identities.
