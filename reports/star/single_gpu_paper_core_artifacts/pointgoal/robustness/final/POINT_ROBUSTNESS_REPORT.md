# Checkpoint-only Point robustness

No policy was retrained. Every row aggregates three training seeds; the training seed is the independent statistical unit. Preselected strong/end-point conditions use 100 episodes per checkpoint and all others use 50.

| Environment | Condition | Episodes/checkpoint | C0 return | P47 return | C0 cost | P47 cost | C0 CVaR | P47 CVaR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SafetyPointGoal1-v0 | action_noise_medium | 50 | 0.3398 | 0.3640 | 17.947 | 11.853 | 86.200 | 57.067 |
| SafetyPointGoal1-v0 | action_noise_strong | 100 | 0.3289 | 0.3979 | 16.683 | 12.143 | 79.750 | 60.050 |
| SafetyPointGoal1-v0 | hazard_position_offset_medium | 50 | 0.4501 | 0.3972 | 11.720 | 17.193 | 56.667 | 84.300 |
| SafetyPointGoal1-v0 | hazard_position_offset_strong | 100 | 0.4982 | 0.5019 | 13.287 | 19.140 | 62.017 | 93.350 |
| SafetyPointGoal1-v0 | hazard_radius_medium | 50 | 0.1677 | 0.2933 | 20.020 | 20.307 | 95.667 | 94.033 |
| SafetyPointGoal1-v0 | hazard_radius_strong | 100 | 0.1516 | 0.2160 | 25.237 | 24.877 | 110.933 | 108.617 |
| SafetyPointGoal1-v0 | nominal | 50 | 0.2764 | 0.3404 | 11.300 | 18.880 | 56.233 | 92.667 |

| Environment | Condition | C0 V_N/V_T | P47 V_N/V_T | C0 outward | P47 outward |
|---|---|---:|---:|---:|---:|
| SafetyPointGoal1-v0 | action_noise_medium | 0.9979 | 0.8427 | 0.22429 | 0.17399 |
| SafetyPointGoal1-v0 | action_noise_strong | 0.9930 | 0.8625 | 0.24613 | 0.18534 |
| SafetyPointGoal1-v0 | hazard_position_offset_medium | 0.9979 | 0.7752 | 0.23126 | 0.18509 |
| SafetyPointGoal1-v0 | hazard_position_offset_strong | 0.9996 | 0.8076 | 0.23186 | 0.19458 |
| SafetyPointGoal1-v0 | hazard_radius_medium | 0.9994 | 0.8948 | 0.21004 | 0.17648 |
| SafetyPointGoal1-v0 | hazard_radius_strong | 0.9977 | 0.8522 | 0.21856 | 0.17016 |
| SafetyPointGoal1-v0 | nominal | 0.9928 | 0.8426 | 0.23587 | 0.17045 |

The table is a robustness trend analysis, not a new selection gate and not an every-seed Pareto claim.
