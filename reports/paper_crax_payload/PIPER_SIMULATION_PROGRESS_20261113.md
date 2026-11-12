# Piper simulation deployment preparation

Piper remains simulation-only; no physical connection or motion has been
enabled. The official Piper asset receipt and model-family identity are stored
beside this report.

- Reach / Iso: 300k simulated transitions completed; 8,103 safe successes and
  521 safe timeouts.
- Reach / Iso-Low: 300k simulated transitions completed; 0 safe successes and
  2,952 safe timeouts. This is retained as a negative baseline, not hidden.
- Reach / EA: 214k/300k transitions at the latest snapshot, about 10.3
  transitions/s, with 5,247 safe successes and 482 safe timeouts. Training is
  still running on A100 GPU0.

The full Piper deployment plan, dry-run boundaries, and physical-motion
authorization constraints remain unchanged. Checkpoints stay server-side.
