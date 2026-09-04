# STAR P4.7 safety-aware geometry final report

P4.7 is the final PointGoal method modification. Scientific effectiveness is interpreted primarily from the equal-training-budget P47@200k versus C0@200k comparison. C1@100k remains a historical reference and part of the untouched preregistered machine audit.

## Aggregate formal results (3 training seeds; 300 evaluation episodes per arm)

The independent statistical unit is the training seed (`n=3` per arm). The 300 evaluation episodes per arm are pooled observations and are not independent training repetitions.

| Arm | Step | Return | Success | Cost | CVaR-20 | Train cost | High-risk V_N/V_T | High-risk outward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 200000 | 0.4621 | 0.227 | 13.567 | 67.750 | 7305.3 | 0.9982 | 0.2209 |
| C1 | 100000 | -0.0777 | 0.210 | 15.550 | 75.750 | 3324.0 | 0.9972 | 0.2143 |
| P47 | 200000 | 0.6545 | 0.253 | 9.433 | 46.800 | 5582.3 | 0.8744 | 0.1834 |

## Primary scientific conclusion: P47@200k vs C0@200k

**equal_budget_pareto_improvement**

### 1. Overall safety-performance

P47 achieves an aggregate Pareto improvement under an equal training budget on SafetyPointGoal1-v0: all three aggregate safety indicators improve without an aggregate return or success-rate decrease.

- Evaluation mean-cost reduction: 30.5% (13.567 -> 9.433)
- Evaluation CVaR-20 reduction: 30.9% (67.750 -> 46.800)
- Return delta: +0.1924; success-rate delta: +2.7 percentage points; task completion observed: True
- Paired training-seed improvements: return 2/3; success 1/3. The aggregate Pareto claim is not an all-seed Pareto claim.

### 2. Training-period safety exploration

**safer_training_period_exploration**

Mean cumulative training-cost reduction is 23.6% (7305.3 -> 5582.3); P47 is lower in 3/3 paired seeds.

### 3. Direction-geometry mechanism

**direction_mechanism_supported**

Both high-risk diagnostics are lower under P47, supporting activation of the intended direction mechanism.

- High-risk V_N/V_T: 0.9982 -> 0.8744 (delta -0.1237)
- High-risk dangerous outward velocity: 0.2209 -> 0.1834 (delta -0.0374)

This evidence is computed only on the highest-risk 20% of samples and is diagnostic; it is not an elimination or freeze gate.

## C1 historical reference

C1 was trained for 100k steps, so it is not the primary equal-budget effectiveness comparator. Its values remain visible above and remain inputs to the preregistered audit, but C1 alone does not determine whether P47 had an effective safety-performance tradeoff.

## Preserved preregistered machine audit

**freeze_P47**

- `safety_efficacy_vs_C0_200k`: PASS
- `training_safety_vs_C0_200k`: PASS
- `safety_retention_vs_C1_100k`: PASS
- `performance_retention_vs_C1_100k`: PASS

The four gates and their freeze result are preserved exactly as preregistered; they are a conservative machine audit, not a post-hoc replacement for the equal-budget scientific interpretation above.

The PointGoal weight/method search is now closed. The machine-audit-selected method is frozen in `p4_7_frozen_pointgoal_config.yaml`; the next stage is the preregistered SHAPO four-environment, five-training-seed experiment.

## Evidence files

- `p4_7_checkpoint_metrics.csv`: common 5x4 checkpoint curves
- `p4_7_formal_final_seed_metrics.csv`: every formal arm-by-training-seed result
- `p4_7_paired_final_deltas.csv`: paired P47 deltas
- `p4_7_checkpoint_sha256.csv`: immutable checkpoint identities
- `p4_7_scientific_interpretation.json`: the three scientific conclusions
- `p4_7_final_summary.json`: scientific interpretation plus preserved machine audit
