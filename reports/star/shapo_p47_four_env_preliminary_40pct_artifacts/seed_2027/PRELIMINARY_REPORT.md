# SHAPO P4.7 Four-Environment 40% Snapshot — PRELIMINARY ONLY

> **Not a formal result.** This snapshot contains one training seed (`n=1`) and must not be used for uncertainty, significance, or cross-seed generalization claims.

- Generated at: `2026-08-19T09:20:52.465282+00:00`
- Frozen formal source commit: `62ca7399357c10645a955dcbead7ed2153a82c69`
- Preliminary protocol SHA-256: `3e48cc7425ebf2900b610f1f62ab17dca19fd4720aa9be0d025e2069e493cf12`
- Training seed: `2027`
- Snapshot: exactly `40%` of each environment-specific training budget
- Evaluation: 5 fixed base seeds × 20 episodes = 100 episodes per policy
- Scientific early stopping: disabled in the continuing formal runs
- Checkpoint/configuration selection from these results: no

## Primary snapshot

| Environment | Step | Budget | Return | Mean cost | CVaR-20 | Training cost |
|---|---:|---:|---:|---:|---:|---:|
| SafetyPointGoal1-v0 | 4,000,000 | 40% | -0.2519 | 11.7500 | 58.7500 | 70912 |
| SafetyPointButton1-v0 | 4,000,000 | 40% | -1.7277 | 15.9700 | 52.2500 | 95762 |
| Ant-v4 | 2,000,000 | 40% | 5823.8252 | 0.1100 | 0.5500 | 509 |
| Walker2d-v4 | 2,000,000 | 40% | 5810.7547 | 0.0100 | 0.0500 | 2312 |

Metrics are absolute within-task values; their scales are not comparable across different environments.

## Task diagnostics

| Environment | Success rate | Failure rate | Time-limit rate |
|---|---:|---:|---:|
| SafetyPointGoal1-v0 | 0.0000 | 0.0000 | 1.0000 |
| SafetyPointButton1-v0 | 0.0300 | 0.0000 | 1.0000 |
| Ant-v4 | 0.0000 | 0.1100 | 0.8900 |
| Walker2d-v4 | 0.0000 | 0.0100 | 0.9900 |

## High-risk direction diagnostics

| Environment | Top-20% V_N/V_T | Dangerous outward velocity |
|---|---:|---:|
| SafetyPointGoal1-v0 | 0.9455 | 0.1892 |
| SafetyPointButton1-v0 | 0.9825 | 0.0495 |
| Ant-v4 | 0.7006 | 0.3270 |
| Walker2d-v4 | 0.7185 | 0.2186 |

## Interpretation boundary

- This is an early frozen-P4.7 snapshot, not a method comparison.
- The independent statistical unit is the training seed; there is only one here.
- The 100 evaluation episodes do not convert this into `n=100` training repeats.
- The complete 4-environment × 5-seed formal study continues unchanged.
- Only the completed formal study may replace this preliminary snapshot.
