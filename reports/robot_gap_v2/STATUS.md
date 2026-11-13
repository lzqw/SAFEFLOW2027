# SAFEFlow Robot-Gap v2

Parent PR14: `5a403e04d3064c2a02d8c28023bb050e3098040c`.
User plan: PointGoal closure; Piper Payload learning/transfer; Piper Reach geometry/deployment;
existing Payload targeted geometry repair. New training cap2.20M, not a spending target.

| Task | State | Next evidence |
|---|---|---|
| Old Piper Reach/Payload300K candidates | All six completed on A100 | Compact final evaluations/receipts |
| Payload failure diagnosis | Fixed10-layout diagnostic completed, no training | See diagnostic_motion_episodes.csv and diagnostic_summary.json |
| Piper speed | Three10K profiles complete:10.63/32.46/57.66SPS | Batched-large chosen; all short scores zero, no learning-equivalence claim |
| Reach→Payload | Matched100K pair complete; both200K additional continuations running | Fixed100K endpoint parents, identical full-state resume to300K; no curriculum |
| Reach geometry adaptation≤0.20M | Both100K adaptations complete | New-geometry safe success Iso80%, EA73.33%; nominal/scale100% both |
| Existing Payload repair≤1.00M | Not launched;4080 original IP unreachable | Recover connection; preserve shared adaptation budget |
| Final PointGoal mechanism/efficiency/figures | Three-seed figures, native SAC/Flow CPU latency and fixed selected-policy trajectories complete; mechanism queued | Final-selected64×32/four samplers, after Piper work and≥8GiB free GPU memory |
| Reach deployment candidate/control program | Adapted endpoints frozen and tested on900 independent-layout episodes; actual bundles/replay exported | See reach_adapted_release/README.md; geometry violations28%/30%, hardware eligibility false |
| Payload post-training closure | Five canonical candidates/method fixed; CPU controller1282516 waits for completed continuations |50common-validation and100independent-layout episodes/condition; no RL or extra candidate opportunities |

Snapshot:2026-11-13T17:44:40UTC (November14,01:44:40+08). Initial campaign and
Reach followup controllers exited after completing all their jobs. New controller
PID1263416 launched Payload Iso PID1263426 and EA PID1263428, each200K additional
transitions from its fixed100K endpoint. All three processes were confirmed live
again at01:45+08. Observed new training is592K including162K continuation prefixes
(Iso203K/300K cumulative, EA159K/300K cumulative), not
a final result or perpetual liveness guarantee. Controllers check every30minutes,
retain other projects and never restart failed jobs. Final mechanism controller
was replaced before any query to explicitly supply the pinned native CRAX
source path; its current PID/state are in TASK_STATUS.json. It waits for the
Payload continuations and≥8GiB shared GPU memory. All outputs use the
data disk; no additional PointGoal RL is run.

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
2×200K =830K, against the2.20M cap. Actual observed new transitions are592K at
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
