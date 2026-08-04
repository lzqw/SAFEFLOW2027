# STAR P4.1 correctness and training-length audit

Captured 2026-08-05 on the RTX 3090 experiment server. This report keeps the legacy
training-length diagnosis separate from the corrected P4.1 implementation.

## Outcome

The legacy failure is not explained by a 300k training cap. Its registered reward-TD stop fired at
350k, after goal completion and movement had already collapsed relative to initialization.

P4.1 passed the engineering ladder and completed the preregistered `100k × 3 seeds` minimum
mechanism test. It is not promoted to 200k:

- only seed 2029 entered W3; seeds 2027 and 2028 remained in W2;
- resident boundary support ended at `0 / 0 / 5`, versus the W3 threshold of 128;
- pooled goal completions fell from 10/60 at 50k to 2/60 at 100k;
- every seed stalled in at least 10% of episodes, and two seeds completed no goals at 100k;
- fixed-random-flow Brier skill was negative for all three seeds;
- reward TD rose monotonically over 85k, 90k, 95k, and 100k for all three seeds;
- dangerous normal velocity did not decrease for any seed.

The pooled 100k mean cost is 7.03, below the budget of 10, but this is not sufficient evidence of
safe convergence: seed 2028 remains above budget at 11.95, goal completion collapses, and stall is
nonzero. The decision is therefore `stop_at_100k_do_not_launch_200k`. Because 100k is explicitly a
minimum mechanism test, `final_p4_verdict` remains JSON `null`; this is not presented as a final P4
pass or failure.

The 200k/300k/600k/1M continuations, formal 50-seed evaluation, and four-environment five-seed main
experiment were not launched.

## Provenance

| Evidence | Frozen source |
| --- | --- |
| Legacy training continuation | `1a4533c` |
| Legacy fixed-random-flow evaluator | `03d8ac6db0c35e1f3c66dbdb428ef9c6d2541ae8` |
| P4.1 training | `e71e7a1d78002203cfba430f3ba5948b3a1b1fdd` |
| 0/20k evaluation invocation | `25ebf09dc79251030bc43b22ee6bdf9e5e9c3e10` |
| 50k evaluation and final 20k rendering | `267d1a189180f1ba23ad1770caf3280dd0c356ec` |
| 100k evaluation and replay audit | `614dda284c7392628a6206139467a5c1b2366e2b` |
| 100k mechanism rendering and scoped summary | `ddc90a351d0d3a7d7e6c2401c0a7f475b6343ea8` |

The fixed-random-flow evaluator implementation is identical across the listed P4.1 invocation
commits. Training and analysis ran from clean detached worktrees; the corrected model was never
mixed with a legacy continuation artifact.

## Correctness repair

P4.1 implements the requested semantics:

- actor, reward critic, cost critic, and violation-probability ensemble share
  `[observation, budget ratio, horizon ratio]`;
- reward targets mask `terminated`, while cost targets mask `terminated | truncated`;
- the probability ensemble trains only on finalized episode labels
  `1[cost-to-go > remaining budget]`;
- W3 geometry uses the probability UCB and one Gaussian boundary definition centered at 0.5;
- W3 checks both label classes, positive Brier skill, AUROC, valid gradients, disagreement, and
  boundary support;
- actor loss is normalized by `1 + lambda`, and the global safety dual is record-only before W2;
- mechanism selection is deterministic and independent of predicted risk, with fixed
  `1.0/0.5/0.2/0.0 × safe/boundary/high-risk` strata.

The final exact analysis source passes all 78 STAR tests. Relative to `baaad93`, 37 test functions
were added, including synthetic two-dimensional normal-direction, contraction, and tangent
preservation checks. The four-environment 5k gate and the PointGoal 20k scope gate both passed.

## Legacy length audit

The primary fixed-random-flow audit used 20 fixed evaluation seeds per checkpoint and retained the
zero-base result only as a center-flow diagnostic.

| Step | Mean return | Mean cost | Goals / 60 | Stall |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.364 | 57.33 | 29 | 3.3% |
| 300k | -0.029 | 20.13 | 0 | 26.7% |
| 350k | 0.137 | 46.90 | 4 | 35.0% |

At 350k, reward TD was 80.5×, 73.9×, and 94.5× the registered two-times-minimum thresholds for
seeds 2027, 2028, and 2029, with a positive final slope in every seed. Continuations to 400k and
beyond were therefore not launched. The additional legacy audit cost was 4.91 CNY.

## P4.1 fixed-random-flow evaluation

Each checkpoint uses the same 20 evaluation seeds for each training seed. The table below pools the
raw 60 episodes; confidence intervals use the preregistered normal approximation.

| Step | Mean return (95% CI) | Mean cost | Cost CVaR20 | Goals / 60 | Goal episodes | Stall | Step-stall | Displacement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -1.343 `[-1.839, -0.847]` | 44.68 | 182.25 | 5 | 8.3% | 1.7% | 5.7% | 2.336 |
| 20k | 0.014 `[-0.256, 0.284]` | 81.65 | 364.08 | 9 | 15.0% | 6.7% | 23.1% | 1.184 |
| 50k | 0.178 `[-0.086, 0.442]` | 46.73 | 224.67 | 10 | 11.7% | 20.0% | 38.8% | 0.927 |
| 100k | -0.257 `[-0.459, -0.056]` | 7.03 | 35.17 | 2 | 3.3% | 16.7% | 39.8% | 0.964 |

The 100k cost reduction is accompanied by fewer goals, a significantly negative pooled return CI,
lower action magnitude, and sustained step-level stall. It is therefore not treated as successful
safety learning.

### 100k evidence by seed

| Seed | W stage / boundary | Return | Cost | Goals | Stall | External AUROC / Brier skill | Replay AUROC / Brier skill | Reward TD min → final |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2027 | W2 / 0 | -0.243 | 1.50 | 2 | 25% | 0.771 / -32.750 | 0.999999 / 0.9978 | 0.00769 → 0.09335 |
| 2028 | W2 / 0 | -0.354 | 11.95 | 0 | 15% | 0.860 / -2.672 | 1.000000 / 0.9999 | 0.00880 → 0.22069 |
| 2029 | W3 / 5 | -0.175 | 7.65 | 0 | 10% | 0.556 / -2.084 | 0.999119 / 0.9402 | 0.00708 → 0.04279 |

The replay classifier is nearly perfect on its on-policy finalized labels, while fixed-random-flow
calibration is poor. Combined with the almost empty boundary pool, this indicates a replay/support
and generalization problem rather than an untrained probability head.

## Mechanism evidence

For each seed, all 12 fixed-budget strata were populated from a deterministic 50k replay sample,
and the trajectory storyboard contains 0/20k/50k/100k primary trajectories. Saturated safe and
high-risk regions correctly report undefined normals rather than inventing directions.

Tangent preservation is 1.0 for all seeds. Dangerous normal velocity does not improve: it is
unchanged at the final W2 row for seeds 2027 and 2028, while seed 2029 rises from 0.0408 at its final
W2 reference to 0.1166 at 100k. Seed 2029 also has an undefined counterfactual boundary-gate mean
because its sampled counterfactual batch contains no boundary item; the report preserves this as
missing evidence instead of crashing or fabricating a value.

## Artifacts and cost

- [100k machine report](p4_1_pointgoal_100k.json)
- [100k compact CSV/assessment artifacts](p4_1_pointgoal_100k_artifacts/)
- [20k scope report](p4_1_pointgoal_20k.json)
- [four-environment 5k report](p4_1_four_env_5k.json)
- [legacy length audit](p4_1_legacy_length_audit.json)

The three P4.1 chains cost 9.66 CNY in total, including their initial 20k segments. Full checkpoints
remain under `/root/autodl-tmp/SAFE/runs/active/p4.1-e71e7a1-pointgoal-100k`; raw trajectories,
replay audits, mechanism PNGs, and diagnostic figures remain under
`/root/autodl-tmp/SAFE/artifacts/p4.1/promotion-e71e7a1/pointgoal-100k`. Only compact, reviewable
evidence is committed to GitHub.
