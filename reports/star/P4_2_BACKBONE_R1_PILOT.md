# P4.2 B0-R1 seed-2028 pilot

Status: failed the preregistered feasibility hard stop; safety work remains paused.

## Immutable run result

The pilot used training source `2c03c7ad07b36eaebab71affa8624080b6e4406f`, seed 2028,
global safety off, local geometry off, target-flow reward backups on, and actor update interval 2.
It completed exactly 50,000 environment transitions, 48,000 outer updates, 21,000 actor updates,
and a 50,000-transition replay. All four required checkpoints exist, including replay-bearing
`latest_full.pt` and `final_full.pt`. Training took 2,094.801 seconds and the recorded end-to-end
receipt cost was 0.8642066 CNY.

Task behavior recovered strongly. The external fixed-random-flow evaluation used seeds
12027--12046, 20 episodes at each endpoint:

| endpoint | return | episode success | goals | stall | step displacement | path length |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -3.31098 | 0.15 | 3 | 0.0000 | 0.02023 | 20.2073 |
| 50k | 14.23341 | 1.00 | 133 | 0.0000 | 0.01200 | 11.9864 |

The reward-TD series at 5k intervals was
`[0.00880, 0.00869, 0.00650, 0.00559, 0.00524, 0.00668, 0.00636, 0.00782,
0.00987, 0.01136]`. Its final four values rise strictly and the final/historical-minimum ratio is
2.1673. Therefore the frozen `reward_td_monotonic_divergence` hard stop is true. The machine
decision is `continue_backbone_repair_and_pause_safety`; seeds 2027 and 2029 were not launched.
`final_p4_verdict` remains null.

## Read-only failure decomposition

All following measurements restore the final full checkpoint without collecting a transition or
updating the saved checkpoint. They are diagnostics, not promotion metrics.

On 4,096 evenly spaced replay transitions and 32 stochastic target-flow draws, target-action
sampling standard deviation was 0.00670, 64.2% of the all-draw-mean TD residual 0.01045. Four
target samples reduced the observed one-sample TD from 0.01206 to 0.01084. This is meaningful
noise reduction but cannot alone explain the late two-fold rise.

The expected-target residual is chronological: the oldest replay decile has TD 0.00750, while the
newest decile has TD 0.01640. A frozen-policy/frozen-replay 2x2 test then varied target samples
(1/4) and reward updates (1/2). At 1,000 outer updates, `4 samples + 2 updates` reduced uniform TD
to 0.00940 and newest-decile TD to 0.01289. This identifies policy-distribution tracking capacity,
not numerical blow-up, as the remaining failure.

Finally, reward batches were decoupled from safety-category stratification and tested with a
10,000-transition recent window:

| target samples | reward updates | recent fraction | uniform TD | oldest TD | newest TD |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.00 | 0.00977 | 0.00760 | 0.01333 |
| 4 | 2 | 0.00 | 0.00833 | 0.00692 | 0.01208 |
| 4 | 2 | 0.50 | 0.00867 | 0.00761 | 0.00805 |
| 4 | 2 | 0.75 | 0.00947 | 0.00823 | 0.00754 |

The 50% mixture best balances current-policy tracking and full-replay retention. The 75% mixture
slightly improves the newest decile but degrades old and aggregate coverage, so it is rejected.
The selected R2 candidate is therefore: four stochastic target samples, two reward-critic updates
per outer update, and a reward/actor batch drawn as 50% uniform replay plus 50% recent 10k replay.
The target actor and actor interval 2 are retained. This choice must be sealed in a new protocol
before any R2 training begins.

## Server evidence

The full run and checkpoints remain under
`/root/autodl-tmp/SAFE/runs/active/p4.2-r1-2c03c7a-b0-repair-50k-pilot/seed-2028`.
Compact evaluation and diagnostic evidence remains under
`/root/autodl-tmp/SAFE/artifacts/p4.2/backbone-r1-2c03c7a/pilot-seed-2028`.
SHA-256 seals and exact compact values are recorded in
`reports/star/p4_2_backbone_r1_pilot.json`; no checkpoint is committed to Git.
