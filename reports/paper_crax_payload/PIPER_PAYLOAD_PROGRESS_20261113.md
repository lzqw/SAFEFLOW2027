# Piper Payload simulation training progress

Piper Payload training was started after the Reach candidates completed. The
standard Piper model and gripper, seed 41027, common 300k transition budget,
and `piper_sim_v1.json` configuration are shared across methods. No physical
robot connection or motion is enabled.

- EA: running on A100 GPU1; latest snapshot 25k/300k transitions, about 14.1
  transitions/s after optimizer warm-up.
- Iso: running on A100 GPU0; latest snapshot 50k/300k transitions, about 28.3
  transitions/s after optimizer warm-up.
- Iso-Low: queued until one A100 slot is released; it will use the same seed,
  budget, and source commit without duplicating either active run.

The two live status receipts are `piper-progress-20261113/PAYLOAD_*_STATUS_LATEST.json`.
At the observed speeds, EA is estimated at roughly 4.5 hours and Iso at roughly
2.5 hours; these are estimates, not completion results. Checkpoints remain on
the server and only compact receipts/results are committed to GitHub.
