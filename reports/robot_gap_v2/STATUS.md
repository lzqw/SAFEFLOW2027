# SAFEFlow Robot-Gap v2

Parent PR14: `5a403e04d3064c2a02d8c28023bb050e3098040c`.
User plan: PointGoal closure; Piper Payload learning/transfer; Piper Reach geometry/deployment;
existing Payload targeted geometry repair. New training cap2.20M, not a spending target.

| Task | State | Next evidence |
|---|---|---|
| Old Piper Reach/Payload300K candidates | All six completed on A100 | Compact final evaluations/receipts |
| Payload failure diagnosis | Fixed10-layout diagnostic completed, no training | See diagnostic_motion_episodes.csv and diagnostic_summary.json |
| Piper speed | Three10K profiles complete:10.63/32.46/57.66SPS | Batched-large chosen; all short scores zero, no learning-equivalence claim |
| Reach→Payload | Matched100K Iso/EA branches running on A100GPU0/1 | Same fixed original Reach/Iso300K Actor parent |
| Reach geometry adaptation≤0.20M | One Iso/EA100K pair queued after each own transfer slot | Shared radii45/50/55mm; fixed-radius evaluation unchanged |
| Existing Payload repair≤1.00M | Not launched;4080 original IP unreachable | Recover connection; preserve shared adaptation budget |
| Final PointGoal mechanism/efficiency/figures | Three-seed figures/efficiency and Flow CPU latency complete; mechanism queued | Final-selected64×32/four samplers, after Piper work and≥8GiB free GPU memory |
| Reach deployment candidate/control program | All15 candidates/2250 validation episodes complete; three native bundles and offline replays exported | See reach_release/README.md; geometry remains weak, hardware eligibility false |

Snapshot:2026-11-13T13:03:17UTC (21:03:17+08). A100 campaign controller PID1039792
launched Iso PID1062014 and EA PID1062015. New training totals125K including
30K profiles and95K combined transfer prefixes; these are not final results or
perpetual liveness guarantees. Reach followup scheduler PID1064715 checks
every30minutes and launches only the queued100K/method finite-radius fork when
that method's own transfer is complete and≥2GiB GPU memory remains. Final
mechanism scheduler PID1068085 waits for Piper followups and≥8GiB. Neither
preempts other projects or restarts a failed job. All outputs use the data disk.

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

Reservation: profiles30K + transfer2×100K + Reach2×100K =430K, against the2.20M
cap. Actual observed new transitions are125K at this snapshot. The1.77M remaining
unreserved allowance is not a spending target. The Reach decision follows the
fixed risk diagnostic: a radius prior corrected values without changing the
paired frozen rollout outcomes, so one shared finite-radius training family is
used instead. No curriculum, optional continuation, or matrix is auto-launched.
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
