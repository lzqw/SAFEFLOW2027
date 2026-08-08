# STAR P4.5 unified cost geometry final report

Conclusion: **safety_benefit_with_performance_cost**

Protocol: `p4.5-unified-cost-geometry-v1` (`8abebc101021ec71a00fdf6d2c5580b5f97dada710e58de00936a75490e85c11`)

All six C0/C1 x seed runs were required to finish 100k unless an engineering failure occurred. The table below reports every final run; pooled values never replace the per-seed evidence.

| arm | seed | return | success | mean cost | CVaR20 | dangerous v_n+ | V_N/V_T | path | stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 2027 | 1.076536 | 0.450 | 16.250000 | 63.750000 | 0.176915 | 0.999478 | 3.555917 | 0.050 |
| C0 | 2028 | 0.386441 | 0.250 | 17.650000 | 81.250000 | 0.142824 | 0.999806 | 2.861478 | 0.100 |
| C0 | 2029 | 0.729578 | 0.200 | 13.750000 | 65.250000 | 0.157987 | 0.998830 | 3.679286 | 0.050 |
| C1 | 2027 | 1.228598 | 0.450 | 10.650000 | 49.500000 | 0.180419 | 0.999896 | 3.182057 | 0.100 |
| C1 | 2028 | -1.457632 | 0.050 | 15.200000 | 65.750000 | 0.114308 | 0.992280 | 6.839948 | 0.000 |
| C1 | 2029 | 0.451291 | 0.200 | 0.000000 | 0.000000 | 0.113892 | 1.000000 | 2.368723 | 0.250 |

## Pooled final result

| arm | episodes | return | success | mean cost | CVaR20 | dangerous v_n+ | path | stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 60 | 0.730852 | 0.300 | 15.883333 | 70.083333 | 0.159242 | 3.365560 | 0.067 |
| C1 | 60 | 0.074086 | 0.233 | 8.616667 | 43.083333 | 0.136207 | 4.130243 | 0.117 |

## Locked interpretation

- `mean_cost`: C1 lower in 3/3 paired seeds ([2027, 2028, 2029]); pooled delta -7.266667; condition met.
- `cost_cvar_20`: C1 lower in 3/3 paired seeds ([2027, 2028, 2029]); pooled delta -27.000000; condition met.
- `dangerous_normal_velocity`: C1 lower in 2/3 paired seeds ([2028, 2029]); pooled delta -0.023035; condition met.
- Safety-majority condition: True.
- Pooled C0-C1 return loss: +0.656766; allowed maximum 0.10.
- Pooled C0-C1 success loss: +0.066667; allowed maximum 0.10.
- Performance maintained: False.

## Audit and critic evidence

Each final full-replay checkpoint received the sealed 64-state x 256-action A/B/C common-randomness audit. Cost-critic TD, Q scale, action gradients, cost-to-go MAE/Spearman, and five risk bins are retained in the raw CSV/JSON evidence. Probability AUROC, Brier, skill, and calibration remain diagnostic-only.

## Figures

- `p4_5_return_success.png`
- `p4_5_cost_cvar_training_cost.png`
- `p4_5_geometry.png`
- `p4_5_critic_scales.png`
- `p4_5_trajectory_eval_seed_12027.png`
- `p4_5_trajectory_eval_seed_12036.png`
- `p4_5_trajectory_eval_seed_12046.png`

Large checkpoints and replay stay on the server. `p4_5_checkpoint_sha256.csv` records their immutable identities; the GitHub evidence bundle contains compact CSV/JSON, plots, and this report.
