# Piper preparation: offline only

This package never opens CAN or imports/initializes the robot SDK. Physical
motion is not implemented or authorized. User identified the standard Piper and
standard gripper; the pinned official standard URDF is selected. TCP, firmware,
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

The real experiment generator produces 360 **planned** trials, interleaving
Iso/Iso-Low/EA within task/condition/repetition blocks. A separately enabled
on-site implementation is needed before execution. Protection parameters must
be common across methods. Training/real adaptation is also not enabled.
