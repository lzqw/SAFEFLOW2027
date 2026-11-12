# Piper Payload simulation training progress

Piper Payload training was started after the Reach candidates completed. The
standard Piper model and gripper, seed 41027, common 300k transition budget,
and `piper_sim_v1.json` configuration are shared across methods. No physical
robot connection or motion is enabled.

- EA: running on A100 GPU1; latest snapshot 54k/300k transitions, about 11.6
  transitions/s after optimizer warm-up.
- Iso: running on A100 GPU0; latest snapshot 126k/300k transitions, about 26.8
  transitions/s after optimizer warm-up.
- Iso-Low: queued until one A100 slot is released; it will use the same seed,
  budget, and source commit without duplicating either active run.

The two live status receipts are `piper-progress-20261113/PAYLOAD_*_STATUS_LATEST.json`.
At the observed speeds, EA is estimated at roughly 5.9 hours and Iso at roughly
1.8 hours; these are estimates, not completion results. Checkpoints remain on
the server and only compact receipts/results are committed to GitHub.

Iso also produced an interim 60k evaluation (30 episodes per condition): all
three conditions timed out without collision or violation. This is an early
learning snapshot, not a final candidate assessment; the process remains live.

At 120k, Iso again evaluated 30 episodes per condition: nominal,
new-geometry, and execution-scale conditions all had 0 safe successes,
0 violations, and 30/30 safe timeouts. The 120k receipt records 180,000
evaluation physics substeps and zero physical-robot interactions. This remains
an interim diagnostic, not a final learned-policy result.
