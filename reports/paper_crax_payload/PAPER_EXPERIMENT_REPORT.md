# SAFEFlow CRAX / Payload / Piper — verified delivery report

Date: 2026-11-13. This report is the current status of the compact artifacts
on `codex/paper-crax-payload-v1`; historical interim reports are retained under
the same directory.

## 1. Fast and effective PointGoal configuration

The frozen PointGoal matrix contains three methods (SAC-Lag, Stable-Iso, EA),
three training seeds, and 10M transitions per run. Common validation used five
seeds, five checkpoint opportunities, and 50 episodes per candidate (225
candidate/seed evaluations). The common risk rule was mean episode cost ≤25.

| method | selected seed/checkpoint | common cost | common goals/episode | independent cost | independent goals/episode |
|---|---:|---:|---:|---:|---:|
| SAC-Lag | 51029 / 2,016,256 | 11.63 | 1.66 | 13.20 | 1.93 |
| Stable-Iso | 51028 / 10,000,128 | 24.02 | 27.52 | 26.85 | 28.29 |
| EA | 51029 / 8,000,000 | 22.00 | 35.26 | 21.32 | 35.43 |

EA is the overall task-performing risk-qualified candidate. Its exact policy
bytes were tested on five fresh independent seeds ×100 episodes. Measured
PointGoal full-training SPS and checkpoint details are in
`CRAX_POINTGOAL_RESULTS.csv`, `POINTGOAL_SELECTION_AND_INDEPENDENT_20261113.md`,
and `pointgoal-formal-selection-20261113/`.

## 2. Engineering acceleration versus method changes

Engineering changes include batched head-VJPs and joint target draws: the Fast
EA profile increased steady throughput from 1,082 to 2,482 transitions/s at
matched sampling settings. Compiled evaluation reuse also reduced repeated
validation overhead. These are throughput changes, not evidence of an
algorithmic task gain. Algorithmic changes are the native cumulative-cost
learner, Stable-Iso covariance, EA directional geometry, and their fixed
training/selection rules. The final Piper EA run reports 10.256 transitions/s
end-to-end on A100 GPU0 and 298,000 learner updates for 300,000 collected
transitions.

## 3. Independent EA benefit

All three selected methods received the same common-validation and locked
independent-test procedure. On the independent test, EA reached 35.43
goals/episode at mean cost 21.32, versus Stable-Iso 28.29 at 26.85 and SAC-Lag
1.93 at 13.20. Thus EA provides the strongest task progress among the
risk-qualified candidates while remaining below Stable-Iso's independent mean
cost. This is a selected-policy comparison, not a claim that every seed wins.

## 4. Seeds, endpoints, and failures

All nine formal PointGoal runs and their fixed endpoints are retained; weak
intermediate checkpoints, failed profiling attempts, and negative baselines
are not hidden. Policy hashes, common-validation receipts, independent-test
receipts, and the complete candidate history are stored in
`pointgoal-formal-selection-20261113/` and `TRIAL_LEDGER.csv`. Selection was
performed before the independent test; the independent test did not alter the
selection.

## 5. Payload, robustness, mechanisms, and Piper

- Payload common validation contains all 30 registered candidates. SAC-Lag
  seed 41028 at 300k is the deployment-eligible risk-qualified candidate;
  Stable-Iso and EA stress/offset trade-offs remain explicitly reported.
  Held-out condition summaries are in `PAYLOAD_RESULTS.csv`,
  `ROBUSTNESS_RESULTS.csv`, and the Payload independent-test report.
- Robustness includes geometry, execution, and alternative-route conditions;
  the offset-route diagnostic is marked as not proving initial admissibility.
- Arm M1–M4, PointGoal common-state analysis, and the small theory numerics are
  recorded in `MECHANISM_RESULTS.csv`, `ARMMECHANISM_REPORT.md`, and
  `THEORY_NUMERICS.md`. They are diagnostic evidence, not a universal risk
  certificate.
- Piper uses the pinned official standard Piper arm and standard gripper model.
  Iso, Iso-Low, and EA Reach simulations are complete. The final EA 300k
  evaluation (30 episodes per condition) achieved nominal safe success 1.00,
  new-geometry safe success 0.333, and execution-scale-0.85 safe success
  0.833; Iso reached 0.900/0.300/0.833 and Iso-Low reached 0/0/0 with
  timeouts. No collision was recorded in the three EA conditions. The receipt records 6,000,000 training
  physics substeps, 6,318,898 distance queries, 29,311.6 seconds elapsed, and
  zero physical-robot interactions. Piper remains simulation-only; no CAN,
  SDK motion, or random physical exploration was enabled.
- Piper Payload has a validated simulation/smoke and constructive-feasibility
  path. The learned 300k candidates are now running on two A100 GPUs: EA is at
  146k/300k transitions; Iso has completed its 300k/300k terminal run, and
  Iso-Low is running on the released A100 GPU0 at 70k/300k steps. These
  are live training
  states, not final learned-policy results, and remain separate from the
  completed CRAX Payload experiments. Iso's 120k interim evaluation had
  0/30 safe successes, 0 violations, and 30/30 safe timeouts in each of the
  nominal, new-geometry, and execution-scale conditions.
  EA's 60k interim evaluation likewise had 0/30 safe successes in each
  condition; nominal and execution-scale had no violations, while
  new-geometry had 4/30 violations. These are interim learning diagnostics,
  not final candidate results. Iso's 180k evaluation had 0/30 safe successes
  in all conditions, with 1/30 violation only in new-geometry. The 240k
  evaluation still had 0/30 safe successes in all conditions; nominal and
  execution-scale had 0/30 violations and new-geometry had 6/30 violations,
  with no collision or physical-robot interaction. Iso's terminal 300k
  evaluation also had 0/30 safe successes in every condition; nominal and
  execution-scale had 0/30 violations, new-geometry had 1/30 violation, and
  no collision or physical-robot interaction was recorded.

## 6. Actual cost and reproducibility

Actual training budgets and non-training simulator queries are recorded in
`TRIAL_LEDGER.csv`, `NATIVE_PROGRESS.json`, and the method receipts. Large
checkpoints remain server-side/4080 archive; GitHub contains code, compact
results, configurations, reports, and receipts. Required delivery artifacts
are present in this directory: protocol, fast profile, PointGoal/Payload/
robustness/mechanism/parameter tables, model selection, trial ledger, Piper
deployment plan, and this report. No physical interaction cost was incurred.

The final selection JSON is `MODEL_SELECTION.json`; its previous interim
version is retained as `MODEL_SELECTION_HISTORY.json`.
