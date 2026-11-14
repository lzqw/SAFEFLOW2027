# Deployment preparation: no5080 load and no physical transport

User identified5080 as the deployment PC but asked to avoid using it. No5080
program is run in this phase; a short CPU-only offline timing window was asked
about asynchronously. Simulation/analysis continue without waiting for it.

`deployment.piper.replay_policy --benchmark` now additionally emits p99, raw
latencies and OPTIONAL diagnostic deadline misses (`--period-s`). If no real
period is provided, deadline counts stay null instead of inventing a control
frequency. Even a provided value is labelled uncalibrated. Existing CPU timing
artifacts are NOT rewritten or falsely promoted to deployment-PC measurements.

Offline action timing includes normalization, full actor, geometry/tail/input
VJPs and host return. It excludes real observation acquisition, communication
and actuation. Consequently running this on the deployment PC would only
partially close E09; full I/O latency and deadline performance still require a
separately arranged, reviewed interface with on-site personnel. No SDK, CAN,
socket or hardware transport was added to obtain a prettier complete receipt.

The preparation-only protocol CLI now defaults to the already requested
balanced Reach90 (3methods×3conditions×10), not the older360-Reach/Payload matrix.
The legacy plan remains explicitly available as`--protocol legacy360`, not
silently scheduled. All episode statuses are planned/not executed and motion
authorization remains false. Conditions are labels, not permission to copy
the high-violation simulation stress geometry onto a real robot.

E10 remains missing: measured TCP/flange/base transforms, units, feedback and
control period, speed/load/workspace limits, protected layout, emergency stop
and operator presence. Their unknown values in`hardware_pending.json` remain
null/false, not guessed from simulator constants. E11 actual physical count
remains zero; E12 Payload and E13 real online RL are not automatically added.

To prepare a plan file locally, without any motion or server execution:

```bash
python -m deployment.piper.protocol --output /a/new/path/reach90.json
```
