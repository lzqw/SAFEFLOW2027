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
There are currently **no trained Piper deployment candidates**; see
`candidates_pending.json`. The model smoke and synthetic dry-run must not be
reported as learned success, candidate generation, or real interactions.

Training must use `scripts.paper_crax_payload.piper_helpers.PiperTrainingEnvironment`
and its declared common configuration, not the provisional zero-action CLI.
Follow-up checks found the base200-iteration goal IK insufficient. The training
wrapper uses the same actual solver with1200 iterations, static safe/reachable
endpoint admission, and actual payload-centroid completion; constructive
feasibility is recorded separately, not counted as a learned policy result.

The real experiment generator produces 360 **planned** trials, interleaving
Iso/Iso-Low/EA within task/condition/repetition blocks. A separately enabled
on-site implementation is needed before execution. Protection parameters must
be common across methods. Physical robot training/real adaptation is not enabled.

## Simulation candidates: Reach running, Payload queue prepared

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
