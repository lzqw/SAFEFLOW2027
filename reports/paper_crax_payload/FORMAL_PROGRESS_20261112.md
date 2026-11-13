# CRAX/Payload live progress — 2026-11-12

This is a compact, read-only progress snapshot. It intentionally contains no
model checkpoints; large artifacts remain on the A100/4080 hosts.

The latest queue receipt (11:22 UTC) supersedes the earlier table below for
process status: **6/9 formal runs are now complete**, EA/51028 is at about
3.49M/10M steps, Stable-Iso/51029 is at about 2.27M/10M steps, and the final
EA/51029 job will be appended automatically. The latest queue and observer
receipts are in `formal-latest-20261112-1122/`.

## Current state

- PointGoal formal matrix: **5 of 9 registered runs completed**; 2 runs are
  active and the remaining 2 will be appended by the finite queue. The frozen
  protocol is unchanged: three methods (`sac_lag`, `stable_iso`, `ea`), three
  training seeds, 10M actual transitions per run, five checkpoint opportunities.
- Completed training runs: SAC-Lag seeds 51027/51028; Stable-Iso seeds
  51027/51028; EA seed 51027. No held-out or independent test has been run.
- Active A100 runs at snapshot time: EA seed 51028 (about 1.21M/10M steps,
  about 1.1k transitions/s) and SAC-Lag seed 51029 (just launched). GPU use is
  high; no foreign process was stopped.
- Payload on RTX 4080: EA seed 41028 at 278k/300k transitions, about 25.3
  transitions/s. Its recent training window reports 0.89 success, 10,837 safe
  successes, 874 violations, and 514 safe timeouts. This is training telemetry,
  not the final strict selector result.
- Piper remains simulation-only. Reach/Iso-Low development is running on A100;
  no physical robot connection or motion has been attempted.

## Available PointGoal endpoint evidence

The following are 50-episode evaluations from completed formal checkpoints;
they are descriptive until the complete common validation and selection gate is
run:

| method / seed | final checkpoint | at-least-one goal | mean episode cost | budget-compliant fraction | budget-compliant return |
|---|---:|---:|---:|---:|---:|
| EA / 51027 | 10,000,128 | 1.00 | 20.82 | 0.64 | 16.72 |
| Stable-Iso / 51027 | 10,000,128 | 0.76 | 28.71 | 0.86 | 1.33 |
| Stable-Iso / 51028 | 10,000,128 | 1.00 | 27.44 | 0.48 | 24.88 |

The two completed SAC-Lag runs have valid 10,000,384-transition native
checkpoints and receipts; their common-validation metrics are intentionally not
substituted with a different evaluation protocol.

## Next gate

The queue continues without restart. After all nine runs have receipts, the
pre-registered common validation (same five checkpoints and validation seeds)
will be executed, followed by model selection and only then the held-out test.
Therefore this snapshot is a progress report, not a final deployment claim.

Authoritative queue/observer JSON snapshots for this report are stored beside
this file in `formal-latest-20261112-1051/` and
`4080-inspection-20261112-1051/`.
