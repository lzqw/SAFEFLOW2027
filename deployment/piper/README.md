# Piper preparation: offline only

This package never opens CAN or imports/initializes the robot SDK. Physical
motion is not implemented or authorized. User identified the standard Piper and
standard gripper; model/gripper identity is CONFIRMED, not an unanswered field.
The pinned official standard URDF is selected. Measured TCP, firmware,
calibrated frame, control period, workspace, payload and speed limits remain
missing; fill `hardware_pending.json` with the on-site operator. Provisional
simulation limits do **not** substitute for those answers.

Pinned official sources are in `model_lock.json`. Assets stay outside GitHub:

```bash
python -m deployment.piper.assets --cache /path/to/external/cache --receipt /path/to/new-assets.json
python -m deployment.piper.simulation --assets-receipt /path/to/new-assets.json --task Reach --steps 3 --output /path/to/new-reach-smoke.json
python -m deployment.piper.simulation --assets-receipt /path/to/new-assets.json --task Payload --steps 3 --output /path/to/new-payload-smoke.json
python -m deployment.piper.offline dry-run --output /path/to/new-offline-record.jsonl
python -m deployment.piper.offline replay --input /path/to/new-offline-record.jsonl
python -m deployment.piper.protocol --output /path/to/new-real-protocol.json
```

Assets fetch the pinned official ROS model, resolve package URIs to external
mesh paths, and retain source commit/digest. They do not execute vendor scripts
or silently upgrade. The two environments use the actual six-joint Piper model,
not Panda, and the user-identified standard gripper with a provisional TCP. Payload is a
light rigidly held body; no grasp learning. Arm-obstacle and payload-obstacle
distance minima cover every substep, including the start of the control
interval. Allowed payload-gripper contact filtering never removes obstacle
collision checks. Pose convention is metres/quaternion xyzw in robot base;
Cartesian actions are normalized deltaXYZ. All observable payload geometry is
in the shared 34D observation.

Offline controller enforces finite values, vector speed/workspace bounds,
observation freshness and monotonicity, and inference deadline. Rejected
proposals have no command/executed action. Recording distinguishes raw policy
proposals from separately measured execution; replay does not execute anything.
Pure SDK-unit conversion is documented against the pinned source, not tested
against physically calibrated hardware.

`bundle.py` exports native Piper Actor, frozen EMA geometry margin, frozen
normalizer, variance, action scale, source/model versions and integrity hashes.
It requires native training provenance and rejects Panda/untrained policies.
`FrozenPiperPolicy` keeps input gradients for pullback without updating model
parameters. A deployment bundle is **not** a full replay/recovery checkpoint.
Iso-Low must actually have lower reference variance, not just a renamed label.
Three genuine frozen Reach bundles are now delivered in
[`reports/robot_gap_v2/reach_release`](../../reports/robot_gap_v2/reach_release/README.md).
The legacy filename `candidates_pending.json` now indexes those actual bundles;
it no longer means training has not started. Iso180K and EA300K are explicit
Pareto candidates that fail the cross-condition risk criterion. Iso-Low120K
haszero task success despitezero violations. None is marked eligible for
physical deployment. Model smokes and dry-runs are not physical interactions.

Training must use `scripts.paper_crax_payload.piper_helpers.PiperTrainingEnvironment`
and its declared common configuration, not the provisional zero-action CLI.
Follow-up checks found the base200-iteration goal IK insufficient. The training
wrapper uses the same actual solver with1200 iterations, static safe/reachable
endpoint admission, and actual payload-centroid completion; constructive
feasibility is recorded separately, not counted as a learned policy result.

The current Robot-Gap v2 first on-site plan is Reach only:90 **planned**, not
executed trials,3methods×3conditions×10 interleaved repeats, in
[`planned_onsite_reach90.json`](../../reports/robot_gap_v2/reach_release/planned_onsite_reach90.json).
The older360-trial Reach/Payload generator below is retained as historical
preparation, not the current execution request. An independently reviewed
on-site transport and calibrated common protection conditions are still needed.
Physical robot training/adaptation is not enabled.

## Current offline entry point

All six original Reach/Payload×Iso/Iso-Low/EA300K runs completed on A100.
The common Reach selection evaluated15 original candidates,2250 episodes;
actual three-method bundle replays each logged30 proposals,zero missed
deadlines andzero physical commands. Their observation fixture is simulated,
not a recording from physical hardware. Replay acceptance is not a safety
certificate. CPU p50/p95 on the training server is Iso1.510/1.569ms,
Iso-Low1.500/1.576ms and EA10.235/10.432ms. Timing on the actual deployment
computer remains pending.

Example from the repository root, using the genuine exported EA policy:

```bash
python -m deployment.piper.replay_policy \
  --bundle reports/robot_gap_v2/reach_release/bundles/ea \
  --observations reports/robot_gap_v2/reach_release/observations.jsonl \
  --output /path/on/data-disk/new-offline-replay.jsonl \
  --device cpu
```

`replay_policy` measures full action inference and records raw actions, bounded
proposals, timeouts and deadlines. It cannot connect to or move the robot.
Do not describe proposed actions as measured executed actions.

Equal100K/method Reach geometry adaptation subsequently reached nominal/new
geometry/scale safe success Iso100%/80%/100% and EA100%/73.33%/100% on30 fixed
development episodes/condition. These are one-training-seed results, not
independent tests or the50-episode common-selection scores. Geometry violations
remain20%/26.67%; the adapted policies are not risk-qualified or substituted
into the frozen original release. Payload transfer continuations are still
running. Current budget/results are in
[`Robot-Gap status`](../../reports/robot_gap_v2/STATUS.md).

## Historical engineering and queue registration (superseded execution state)

The paragraphs below retain the original launch history, not current process
state. In particular, the old4080 Payload queue was superseded by completion
on A100, and old training caps are not a new allowance. The Robot-Gap v2 new
round remains capped at2.20M with current reservations in TASK_STATUS.json.

The approved native Payload Iso-Low CPU engineering run completed3210→3220
transitions with full replay/RNG/partial-state recovery, source70269ac. Its
actual variance.01/energy target75/initialalpha.0002 are retained, but that smoke
is not an eligible deployment policy. Native Reach Iso candidate training now
runs on one shared A100GPU0 slot; the other Reach methods are queued. Piper
Payload training and independent testing have not started. Raw evidence is linked
from `../../reports/paper_crax_payload/PIPER_TRAINING_IMPLEMENTATION.md`.

Six fresh candidates are registered: Reach/Payload ×Iso/Iso-Low/EA ×seed41027,
300k each, total1.8M. With the already charged3220 engineering transitions, the
planned training charge is1,803,220 of3M, not six copies of a3M allowance.
The original all4080 combined and two partition plans are now SUPERSEDED and
rejected by the finite queue guard, preserving their registration without
allowing duplicate candidates. Current placement is
`piper_queue_a100_reach_v1.json`: one sharedGPU0 slot, Reach Iso→Iso-Low→EA;
and `piper_queue_4080_payload_v1.json`: three Payload candidates, prepared only.
The same six unique candidate identities, five checkpoints and common selection
remain; no extra training or engineering/Panda weight transfer is introduced.

A100 native queue57815 and Reach Iso57816 started with source6f9e3bf, nice10 and
CPU thread limits1; actual first3k then5k progress is in
`../../reports/paper_crax_payload/PIPER_A100_DEPLOYMENT.json`. No GPU reservation,
MPS/device changes or dependency upgrades were made; PointGoal may share the
card and remains primary. Sharing is not a promise of zero compute slowdown.
Prioritize pending primary Payload jobs on4080. A coordinator must inspect actual
native handles, GPU/disk, absent state/output/log paths and pinned source/assets.
The admission text is a coordinator preflight contract, not an implemented extra
watchdog or automatically enforced cross-queue scheduler. Do not restart any
failed job invisibly or stop foreign processes to make room. No5080 or hardware.

Official assets are already staged on4080 under
`/home/carla/4T/SAFEFLOW2027_PAPER/vendor/piper-model-cache`; receipt
`/home/carla/4T/SAFEFLOW2027_PAPER/analysis/piper/assets-ac41fcb.json` verifies
the same model commit/source URDF/mesh hashes. Absolute mesh URI resolution gives
a server-specific resolved-URDF digest; that is not a change to official geometry.
For4080 Piper only assets were staged: its candidate training is not launched.
The same pinned assets were streamed to A100 without local staging and resolved
there, before the verified native launch. No simulator steps were used to stage
either host's assets. Actual measured physical calibration is still pending.
