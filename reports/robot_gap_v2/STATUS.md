# SAFEFlow Robot-Gap v2

Parent PR14: `5a403e04d3064c2a02d8c28023bb050e3098040c`.
User plan: PointGoal closure; Piper Payload learning/transfer; Piper Reach geometry/deployment;
existing Payload targeted geometry repair. New training cap2.20M, not a spending target.

Current decisions and unresolved external conditions:
[FINAL_DECISION_REPORT.md](FINAL_DECISION_REPORT.md). A100's launched training,
Payload independent closure and final PointGoal mechanism are complete. The
whole goal is not complete:4080 parent access/repair and actual deployment-PC
latency remain outstanding. No further RL is launched to fill idle capacity.

| Task | State | Next evidence |
|---|---|---|
| Old Piper Reach/Payload300K candidates | All six completed on A100 | Compact final evaluations/receipts |
| Payload failure diagnosis | Fixed10-layout diagnostic completed, no training | See diagnostic_motion_episodes.csv and diagnostic_summary.json |
| Piper speed | Three10K profiles complete:10.63/32.46/57.66SPS | Batched-large chosen; all short scores zero, no learning-equivalence claim |
| Reach→Payload | Both matched300K continuations complete | Iso100%/10%/66.67%, EA83.33%/23.33%/63.33% nominal/geometry/scale development success |
| Reach geometry adaptation≤0.20M | Both100K adaptations complete | New-geometry safe success Iso80%, EA73.33%; nominal/scale100% both |
| Existing Payload repair≤1.00M | Not launched;4080 original IP unreachable | Recover connection; preserve shared adaptation budget |
| Final PointGoal mechanism/efficiency/figures | Complete: final selected64×32/four samplers on native CPU,16,704 controls, zero RL | paper/final_mechanism/: small covariance changes, no within-state cost-event-rate differences |
| Reach deployment candidate/control program | Adapted endpoints frozen and tested on900 independent-layout episodes; actual bundles/replay exported | See reach_adapted_release/README.md; geometry violations28%/30%, hardware eligibility false |
| Payload post-training closure | Complete:1500 common-validation+600 independent episodes; selected Iso60K and EA120K | Independent success Iso100/100/100%, EA96/75/98%; actual frozen bundles in payload_closure/ |

Snapshot:2026-11-13T19:30:13UTC (November14,03:30:13+08). Initial campaign,
Reach followup and Payload continuation controllers exited after completing
their jobs. Both genuine300K continuation receipts exist; workers1263426 and
1263428 are absent after normal completion. All830K reserved new transitions
are now complete, not just progress prefixes. CPU closure controller1282516
completed74,988 native controls,1,499,760 physics substeps and1,578,950 distance
queries. Its actual completed receipt exists and the process is absent; all
listed training/closure/final-mechanism handles are terminal. No task is
restarted merely because its process exited after completion.
Controllers check every30minutes,
retain other projects and never restart failed jobs. The final mechanism's
GPU-waiting controller was verified before any output/query and replaced by
CPU-only controller1319063/native worker1319072, without taking either Payload
GPU slot. Both exited normally with the genuine16,704-control receipt at
50.73s runtime. Execution source1322745e, SHA81d9edd7. CPU/GPU RNG streams are
not claimed bitwise equivalent; all sampler arms share the same CPU increments.
All outputs use the data disk; no additional PointGoal RL is run.

Transfer100K nominal/new-geometry/scale safe success: Iso50%/3.33%/56.67%,
EA66.67%/10%/70%. Iso briefly reached100%/96.67%/100% at60K then regressed;
all five checkpoints are retained in piper_results/, not hidden behind the
endpoint. The evidence permits the planned matched continuation, not an
automatic claim that transfer or EA solves robustness. Original Reach geometry
adaptation improved both methods substantially; Iso is slightly better than EA
on the fixed new-geometry development set. These are one-training-seed results.
Independent adapted Reach confirmation is now complete: Iso nominal/55mm/scale
safe success100%/72%/99%, EA100%/70%/100%,100 episodes per condition. Geometry
violations28%/30% still fail the10% release rule.55mm was in the adaptation
training family: independent layouts, not an unseen-radius claim. Iso-Low's
original120K reference remains0% successful and was not given matched adaptation.
Actual400K bundles are separate from the preserved original release.

Payload-Iso's completed300K endpoint is100% nominal,10% geometry and66.67%
scale safe success (30 development episodes/condition), with0%/90%/33.33%
violations. Its120/180/240K geometry scores66.67/43.33/56.67% also remain in
the full curve. Continued nominal learning did not fix robustness. The fixed
endpoint is retained separately from the queued matched-five-candidate
validation/independent selection. EA's300K nominal/geometry/scale success is
83.33%/23.33%/63.33%, with16.67%/76.67%/36.67% violations. Its120K values
96.67%/80%/96.67% fell sharply at240K, then only partially recovered at300K.
Both complete curves are retained. No robustness solution or consistent EA
advantage is established, and no additional training is launched.

The separately validation-selected Payload candidates now have independent
confirmation on100 new layouts/condition: Iso60K100%/100%/100% safe success
with0% violations; EA120K96%/75%/98% with4%/25%/2% violations. Both branches
actually consumed300K Payload training after the common300K Reach parent.
The five-candidate selection was frozen before test, not chosen using these
independent labels. The one-training-seed result supports a useful Iso
simulation candidate, not a physical or global zero-risk guarantee. Both actual
inference packages/normalizers/manifests/receipts are delivered, motion false.
Full300K endpoints/regressions remain separate from this selected-policy test.

The first fixed PointGoal trajectory layout61027 gives Iso36goals/17.76cost,
EA37goals/28.31cost. It is retained despite EA exceeding25 on that example;
no prettier layout is substituted. All2000 native controls and plots are in
paper/selected_examples/, distinct from independent aggregate tests/mechanism.

Latest A100 finals change the older plan snapshot: original Payload EA at300K has
nominal90%, new geometry53.33%, execution-scale63.33% safe success (30 development
episodes/condition). Original Iso and Iso-Low have0% success throughout. Do not
claim all three failed, or label development outcomes independent-test results.
Do not extend any completed old trial.

Fixed10-layout diagnosis: zero0/10, detour10/10, Iso0/10, Iso-Low0/10, EA10/10.
Iso's mean action norm is0.654, path length0.424m, and final goal distance0.172m
from an initial0.240m; it is not immobile. EA ends at0.0278m with a0.357m path.
This supports a learned task/path behavior gap rather than an unreachable task;
it does not by itself distinguish reward, critic, exploration, or all causal
mechanisms. Tracking error is a control-interval measurement, not a static IK
solver residual. Full raw per-step reward/risk diagnostics stay on the data disk.

Implemented engineering: explicit integer updates_per_vector_step, opt-in packed
two4-sample reward targets with separate evolving-Q evaluations, batched per-head
VJP, independent-process environments with unchanged physics substeps/queries.
Task transfer copies Actor weights to online/target only and rebuilds all task
critics,margin,replay,optimizers,dual,alpha and counters. Legacy resume stays strict.
12 focused tests pass in the established A100 environment, including real Piper
serial/process parity. Local base Python lacks panda_gym; no extra installation
was performed to force that environment to run these tests.

Reservation: profiles30K + transfer2×100K + Reach2×100K + Payload continuation
2×200K =830K, against the2.20M cap. Actual completed new transitions are830K at
this snapshot. The1.37M remaining
unreserved allowance is not a spending target. The Reach decision follows the
fixed risk diagnostic: a radius prior corrected values without changing the
paired frozen rollout outcomes, so one shared finite-radius training family is
used instead. The original paired transfer has learning signals; only its
authorized200K/method continuation is now launched. No curriculum or matrix.
Profiles force full geometry to measure tail/VJP costs within10K; these are
engineering/short-learning diagnostics, not formal algorithm comparisons.
All scientific transfer runs restore the native transition geometry ramp.

All large new artifacts belong on A100 data disk under
`/storage_0/SAFEFLOW2027_lzqw_uid1007/robot_gap_v2`. Source and compact reports are
uploaded through the local branch. A100's migration completed: five experiment
directories totaling68.54GiB logically moved, each checksum-compared before
switching the old path to a symlink. `/home` free space rose approximately79→145GiB.
New launch/cache/temp paths use the data disk and retain40GiB free headroom.
Other projects remain running, GPU sharing continues,5080 is not used and no
physical robot transport is opened. Final physical calibration/activation needs
现场人员确认; simulation action scales are not calibrated physical limits.
