# Piper Payload simulation training progress

Piper Payload training was started after the Reach candidates completed. The
standard Piper model and gripper, seed 41027, common 300k transition budget,
and `piper_sim_v1.json` configuration are shared across methods. No physical
robot connection or motion is enabled.

- EA: running on A100 GPU1; latest snapshot 10k/300k transitions, about 33.4
  transitions/s after optimizer warm-up.
- Iso: running on A100 GPU0; latest snapshot 9k/300k transitions, about 34.9
  transitions/s after optimizer warm-up.
- Iso-Low: queued until one A100 slot is released; it will use the same seed,
  budget, and source commit without duplicating either active run.

The two live status receipts are `piper-progress-20261113/PAYLOAD_*_STATUS_LATEST.json`.
At the observed speeds, each active 300k run is estimated at roughly 2–2.5
hours; this is an estimate, not a completion result. Checkpoints remain on the
server and only compact receipts/results are committed to GitHub.
