# Piper simulation deployment preparation

Piper remains simulation-only; no physical connection or motion has been
enabled. The official Piper asset receipt and model-family identity are stored
beside this report.

- Reach / Iso: 300k simulated transitions completed; final 30-episode safe
  success was 0.900 nominal, 0.300 under new geometry, and 0.833 at execution
  scale 0.85.
- Reach / Iso-Low: 300k simulated transitions completed; final safe success was
  0 across all three conditions, with timeouts retained as a negative baseline.
- Reach / EA: 300k simulated transitions completed on A100 GPU0 at about 10.26
  transitions/s, with 8,215 training safe successes and 482 safe timeouts.
  The final 30-episode simulation evaluation reports nominal safe success
  1.00 (0 violations), new-geometry safe success 0.333 (20 violations), and
  execution-scale-0.85 safe success 0.833 (5 violations). No collision was
  recorded in these evaluation conditions.

The final EA receipt, training receipt, evaluation JSON, and latest status are
stored in `piper-progress-20261113/`. The run used 29,311.6 seconds of elapsed
training time, 6,000,000 training physics substeps, 6,318,898 training
distance queries, and zero physical-robot interactions.

The full Piper deployment plan, dry-run boundaries, and physical-motion
authorization constraints remain unchanged. Checkpoints stay server-side.
