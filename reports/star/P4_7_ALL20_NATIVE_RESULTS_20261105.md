# P4.7: all 20 native final-budget results

Updated 2026-11-05 (UTC+08:00). **Training is complete, but final strict
scientific acceptance is not established.** These are descriptive summaries
of the existing, prespecified final evaluations, not a new evaluation campaign.

## Coverage and current state

- Four environments, five fixed training seeds (2027–2031) each: **20/20**.
- PointGoal and PointButton: 10,000,000 steps per training seed.
- Ant and Walker2d: 5,000,000 steps per training seed.
- Total accepted-lineage budget: **150,000,000 steps**, without reduction.
- Each training seed has the native final evaluation on five fixed evaluation
  seeds (32027–32031), 20 episodes each: 100 episodes per training seed.
- Frozen scientific source: `62ca7399357c10645a955dcbead7ed2153a82c69`.
- All selected native receipts report `passed`, clean scientific source, and
  a completed training summary with the full budget and no engineering hard stop.
  Native receipt status is **not** a final provenance/checkpoint audit pass.
- Latest scheduled GPU observation reports no SAFEFLOW training workers on
  A100, 4080 or 5080. No new seed, duplicate trainer, configuration search,
  rollout or interruption of another project's work was introduced.

The old 19/20 collector display was stale. The final A100 Goal2030 run had
finished, but copying its 2.11 GiB terminal container into the archive was
blocked by the operational 100 GiB free-space reserve. On Nov 5, the last
copy was completed with a narrowly restricted 32 GiB reserve for this one
small, completed primary-source run. The source was retained, no files were
deleted, and no scientific requirement was relaxed. The cache now has 20/20.

## Final-budget metrics

Values are means across **five training seeds**; return includes sample SD.
CVaR-20 is the mean of the five per-training-seed CVaRs, not a pooled-episode
tail estimate. Costs are task-specific and not comparable across environments.

| Environment | Return, mean ± SD | Task metric, mean | Evaluation cost | CVaR-20 | Training cumulative cost, mean per seed |
|---|---:|---:|---:|---:|---:|
| PointGoal | -0.528 ± 0.283 | Success 4.6% | 8.192 | 38.110 | 152,169.0 |
| PointButton | -14.302 ± 2.575 | Success 2.2% | 7.292 | 30.770 | 167,320.2 |
| Ant | 3572.18 ± 2946.75 | Failure 24.0% | 0.240 | 0.710 | 1,158.0 |
| Walker2d | 3986.00 ± 2108.40 | Failure 29.8% | 0.298 | 0.680 | 4,905.8 |

Training costs are accepted-lineage counters, not the total physical cost of
all interrupted/recovered segments. Locomotion failure is not Point success.
The statistical sample size is five training seeds per environment, not 500
independent training replicates. All five seeds, including weak ones, remain
in each summary; no best-checkpoint or best-seed selection was applied.

## Directional geometry diagnostics

Existing diagnostics use the highest-risk 20% samples. Entries below are
five-training-seed means, not thresholds or new selection gates.

| Environment | Normal/tangent variance ratio | Dangerous outward velocity | Reliability weight |
|---|---:|---:|---:|
| PointGoal | 0.9616 | 0.1137 | 0.02896 |
| PointButton | 0.9908 | 0.0327 | 0.00707 |
| Ant | 0.8716 | 0.2957 | 0.13200 |
| Walker2d | 0.7839 | 0.2579 | 0.21364 |

Point-task variance ratios remain close to one and reliability is small;
normal suppression is weak in these visited high-risk samples. Locomotion
shows stronger suppression, but neither that observation nor these unmatched
results establishes a causal safety benefit.

All 20 native training summaries end in W2 with the legacy
`training_geometry_enabled=false` flag. This is **not by itself evidence that
P4.7 was disabled**: the frozen STAR implementation explicitly bypasses the
legacy W3 probability-evidence switch for unified Q-cost stochastic geometry
(`_advance_warmup`, `_flow_reference`, `_update_actor` in
`omnisafe/algorithms/off_policy/star.py`). The P4.7 directional-loss and
anisotropy configuration flags remain enabled. The measured diagnostics,
rather than the legacy flag alone, describe effective geometric strength.

## Interpretation

The full budgets did **not** resolve the low Point success rates. PointGoal
averages 4.6% success and PointButton 2.2%; these are not satisfactory task
performance results. Their average evaluation costs are below the nominal
10-unit budget, but that does not imply every seed or tail is safe. Ant and
Walker2d returns are much higher numerically, with large seed-to-seed spread
and substantial failure rates; reward scales differ across tasks.

No matched C0/C1 control matrix is included here. Do not compare these 10M/5M
results directly with the earlier 100k/200k Point experiments to claim a
Pareto improvement. No superiority, significance or method-success claim is
made. The current scientific configuration remains frozen; this report does
not authorize a new method variant or another weight search.

## Native evidence

Each environment summary contains per-seed metrics, native evaluation JSONs,
and source-file SHA-256 references back to copied native receipt manifests:

- [PointGoal five-seed summary](shapo_p47_four_env_artifacts/partial_milestones/point-goal_native_final_all5/summary.json)
- [PointButton five-seed summary](shapo_p47_four_env_artifacts/partial_milestones/point-button_native_final_all5/summary.json)
- [Ant five-seed summary](shapo_p47_four_env_artifacts/partial_milestones/ant_native_final_all5/summary.json)
- [Walker2d five-seed summary](shapo_p47_four_env_artifacts/partial_milestones/walker2d_native_final_all5/summary.json)

## Final strict audit: attempted once, failed

At **2026-11-05 14:51 UTC+08:00**, the unchanged remote finalizer was run
once after all 20 terminal containers were collected. It exited with status
1 on the first PointGoal-2027 checkpoint provenance/path check. The
[preserved traceback](shapo_p47_four_env_artifacts/operations/20261105/final_strict_audit_20261105.log)
names this original parent:

```text
/root/autodl-tmp/SAFE/shapo_p47_four_env/formal/runs/point-goal-2027/training/STAR-{SafetyPointGoal1-v0}/seed-2027-2026-08-10-19-30-03/checkpoints/latest_full.pt
```

The validator could not map it into the current container through the recorded
relocation origin. A targeted read-only follow-up also found no file at that
exact historical relative path in either the A100 primary container or its
terminal archive copy. The archive has later checkpoints, which must not be
substituted for this missing original. The current A100 account cannot inspect
the retired server's `/root` path; this is not a statement that no external
backup could exist. Earlier indexed-archive searches already document other
unresolved original-parent identities in the
[provenance decisions](shapo_p47_four_env_artifacts/operations/20261019/provenance_decisions/point_goal2027_parent_decision_refresh_20261020_1302.json).

This attempt stops at the first failing cell; it is **not** evidence that the
other 19 strict validations passed. No final complete-evidence seal or
strictly accepted formal bundle was produced. Failed staging is retained at:

```text
/storage_0/SAFEFLOW2027_lzqw_uid1007/terminal_matrix_20261024/runs/shapo_p47_four_env/.shapo-p47-finalization.1HpceX
```

The collector now records the attempt durably and refuses automatic audit
reruns until manual evidence review. Its
[state](shapo_p47_four_env_artifacts/operations/20261105/result_collection_state.json)
distinguishes 20/20 completed training runs from the failed final delivery.
Publication-only recovery input restoration was moved after the remote audit;
an in-progress local copy was interrupted and retained, not deleted. Original
durable 4T/source checkpoints were not changed. The new collector invocation
uses the ordinary 100 GiB reserve; the earlier 32 GiB exception was only for
the final missing container copy and is preserved in commit `9ad5961`.

The immediate deliverable is this clearly labelled all-20 **native-results
report**, not a formally accepted matrix. Recovering authentic historical
parent bytes and supporting relocation evidence could enable a later review;
without them, relaxing the protocol or launching replacement experiments
would be a separate scientific decision requiring user direction. No such
change or additional compute campaign has been started.
