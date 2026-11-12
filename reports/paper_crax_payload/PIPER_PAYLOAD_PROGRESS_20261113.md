# Piper Payload simulation training progress

Piper Payload training was started after the Reach candidates completed. The
standard Piper model and gripper, seed 41027, common 300k transition budget,
and `piper_sim_v1.json` configuration are shared across methods. No physical
robot connection or motion is enabled.

- EA: running on A100 GPU1; latest snapshot 5k/300k transitions, about 43.9
  transitions/s after warm-up.
- Iso: running on A100 GPU0; latest snapshot 4k/300k transitions, about 51.4
  transitions/s after warm-up.
- Iso-Low: queued until one A100 slot is released; it will use the same seed,
  budget, and source commit without duplicating either active run.

The two live status receipts are `piper-progress-20261113/PAYLOAD_*_STATUS_LATEST.json`.
At the observed speeds, each active 300k run is estimated at roughly 1.5–2
hours; this is an estimate, not a completion result. Checkpoints remain on the
server and only compact receipts/results are committed to GitHub.
