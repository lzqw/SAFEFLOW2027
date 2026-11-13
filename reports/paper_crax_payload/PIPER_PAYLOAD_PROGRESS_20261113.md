# Piper Payload simulation training progress

Piper Payload training was started after the Reach candidates completed. The
standard Piper model and gripper, seed 41027, common 300k transition budget,
and `piper_sim_v1.json` configuration are shared across methods. No physical
robot connection or motion is enabled.

- EA: running on A100 GPU1; latest snapshot 180k/300k transitions, about 10.10
  transitions/s after optimizer warm-up.
- Iso: completed 300k/300k transitions on A100 GPU0 at about 26.1
  transitions/s; its terminal checkpoint and evaluation are recorded below.
- Iso-Low: running on the released A100 GPU0; latest snapshot is 152k/300k
  transitions at about 24.71 transitions/s after warm-up. Remaining time is
  roughly 2.4 hours at this rate.

The two live status receipts are `piper-progress-20261113/PAYLOAD_*_STATUS_LATEST.json`.
At the observed speeds, EA is estimated at roughly 3.3 hours and Iso-Low at
roughly 1.7 hours. These are estimates, not completion results. Checkpoints remain on
the server and only compact receipts/results are committed to GitHub.

Iso also produced an interim 60k evaluation (30 episodes per condition): all
three conditions timed out without collision or violation. This is an early
learning snapshot, not a final candidate assessment; the process remains live.

EA's 60k interim evaluation had 0/30 safe successes in all three conditions;
nominal and execution-scale had 0 violations and 30/30 timeouts, while the
new-geometry condition had 4/30 violations and 26/30 timeouts. No collision or
physical-robot interaction was recorded.

At 120k, Iso again evaluated 30 episodes per condition: nominal,
new-geometry, and execution-scale conditions all had 0 safe successes,
0 violations, and 30/30 safe timeouts. The 120k receipt records 180,000
evaluation physics substeps and zero physical-robot interactions. This remains
an interim diagnostic, not a final learned-policy result.

Iso's 180k evaluation likewise had 0/30 safe successes in every condition.
Nominal and execution-scale had 0 violations and 30/30 timeouts; new-geometry
had 1/30 violation and 29/30 timeouts. No collision or physical-robot
interaction was recorded.

Iso's 240k evaluation still had 0/30 safe successes in all three conditions.
Nominal and execution-scale had 0 violations and 30/30 timeouts; new-geometry
had 6/30 violations and 24/30 timeouts. No collision or physical-robot
interaction was recorded. This remains an interim diagnostic while the 300k
training run is still live.

Iso's terminal 300k evaluation also had 0/30 safe successes in all conditions.
Nominal and execution-scale had 0 violations and 30/30 timeouts; new-geometry
had 1/30 violation and 29/30 timeouts. No collision or physical-robot
interaction was recorded. The terminal receipt reports 178,020 physics
substeps and 187,101 distance queries.
