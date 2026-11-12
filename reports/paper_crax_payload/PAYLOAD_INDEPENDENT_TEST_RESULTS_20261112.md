# Payload locked independent test — 2026-11-12

After the complete common-validation freeze, the exact selected policy bytes
were evaluated on fresh layout seeds 71027–71126, 100 episodes per condition.
The test did not modify selection. One held-out radius layout per candidate was
not constructable after the environment's bounded attempts; it is retained as
an explicit `reset_failure` and counted conservatively in the violation count,
not silently resampled.

| selected candidate | nominal safe / violation | radius .075 | payload shape | action scale .80 | offset candidate |
|---|---:|---:|---:|---:|---:|
| SAC-Lag / 41028 / 300k | 99% / 1% | 86% / 14% | 95% / 5% | 94% / 6% | 90% / 9% |
| Stable-Iso / 41028 / 120k | 95% / 5% | 93% / 7% | 94% / 6% | 96% / 4% | 13% / 86% |
| EA / 41028 / 240k | 99% / 1% | 73% / 27% | 90% / 10% | 98% / 2% | 74% / 26% |

The SAC-Lag candidate is the only common-validation risk-qualified deployment
candidate and remains the strongest held-out candidate: it preserves 90% safe
success under the offset condition, while Stable-Iso and EA show substantial
offset degradation. These are fixed-policy independent-test results, not a
claim of an alternative-route proof; route qualification remains a separate
diagnostic.

The compact receipt and queue evidence are in
`payload-independent-test-20261112-1201/`. Full per-episode JSON remains on the
4080 archive.
