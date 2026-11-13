# SAFEFlow Robot-Gap v2

Parent PR14: `5a403e04d3064c2a02d8c28023bb050e3098040c`.
User plan: PointGoal closure; Piper Payload learning/transfer; Piper Reach geometry/deployment;
existing Payload targeted geometry repair. New training cap2.20M, not a spending target.

| Task | State | Next evidence |
|---|---|---|
| Old Piper Reach/Payload300K candidates | All six completed on A100 | Compact final evaluations/receipts |
| Payload failure diagnosis | Fixed10-layout diagnostic completed, no training | See diagnostic_motion_episodes.csv and diagnostic_summary.json |
| Piper speed | Reference10K profile running; next two queued, total30K | Full synchronized SPS, stage time, updates and nominal learning |
| Reach→Payload | Explicit actor-only transfer implemented | One fixed Iso/300K parent, paired100K Iso/EA |
| Reach geometry adaptation≤0.20M | Not launched | Frozen risk-value/direction diagnosis first |
| Existing Payload repair≤1.00M | Not launched;4080 original IP unreachable | Recover connection; preserve shared adaptation budget |
| Final PointGoal mechanism/efficiency/figures | Pending; no new RL training | Selected final models,64 states×32 samples |
| Reach deployment candidate/control program | Existing offline interface retained; further work pending | Common validation selection, bundles, complete latency and replay |

Actual A100 campaign controller PID1039792; first profile PID1039826, confirmed
live during this snapshot. These are host-local observations, not perpetual
liveness guarantees. The controller will run the other two bounded profiles,
then launch the matched Iso/EA100K transfers on GPU0/GPU1 respectively. It does
not launch later adaptation/curriculum stages. All outputs are on the data disk.

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

Initial-phase reservation: profiles30K + transfer2×100K =230K, charged against the
2.20M cap. Later phases require actual profile/diagnostic/adaptation results;
no curriculum, selected continuation, or additional matrix is auto-launched.
Profiles force full geometry to measure tail/VJP costs within10K; these are
engineering/short-learning diagnostics, not formal algorithm comparisons.
All scientific transfer runs restore the native transition geometry ramp.

All large new artifacts belong on A100 data disk under
`/storage_0/SAFEFLOW2027_lzqw_uid1007/robot_gap_v2`. Source and compact reports are
uploaded through the local branch. A100's older data migration is still live in
checksum verification; do not duplicate it or write new outputs into its source.
Other projects remain running, GPU sharing continues,5080 is not used and no
physical robot transport is opened. Final physical calibration/activation needs
现场人员确认; simulation action scales are not calibrated physical limits.
