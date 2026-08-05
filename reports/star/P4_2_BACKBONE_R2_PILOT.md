# P4.2 B0-R2 seed-2028 pilot

Status: failed the preregistered feasibility hard stop; safety work remains paused.

## Immutable run result

The pilot used training source `2332a550cc0a366c436c9a116b72b7bfc918bef2`, seed 2028,
global safety off, local geometry off, target-flow reward backups on, actor update interval 2,
four stochastic target samples, two reward-critic updates per outer update, and an independent
reward replay batch split equally between uniform replay and the most recent 10,000 transitions.
It completed exactly 50,000 environment transitions, 48,000 outer updates, 96,000 reward-critic
updates, 21,000 actor updates, and a 50,000-transition replay. All four required checkpoints exist,
including replay-bearing `latest_full.pt` and `final_full.pt`. Training took 2,434.097 seconds and
the recorded end-to-end receipt cost was 1.0036803 CNY.

The external fixed-random-flow evaluation used seeds 12027--12046, 20 episodes at each endpoint:

| endpoint | return | episode success | goals | stall | step displacement | path length |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -3.31098 | 0.15 | 3 | 0.0000 | 0.02023 | 20.2073 |
| 50k | 12.95416 | 0.95 | 122 | 0.0500 | 0.01099 | 10.9813 |

The reward-TD series at 5k intervals was
`[0.00505, 0.00512, 0.00384, 0.00293, 0.00265, 0.00343, 0.00496, 0.00680,
0.00851, 0.00923]`. Its final four values rise strictly and the final/historical-minimum ratio is
3.4826. Therefore the frozen `reward_td_monotonic_divergence` hard stop is true. The machine
decision is `continue_backbone_repair_and_pause_safety`; seeds 2027 and 2029 were not launched.
No continuation beyond 50k or formal four-environment run was launched, and `final_p4_verdict`
remains null.

R2 reduced the absolute final logged TD by 18.7% relative to R1 (0.01136 to 0.00923), but did not
repair the registered tail trend. Strong task behavior therefore cannot promote this candidate.

## Read-only failure decomposition

All following measurements restore the final full checkpoint without collecting a transition or
changing a saved checkpoint. They are diagnostics, not promotion metrics.

On 4,096 evenly spaced replay transitions and 32 stochastic target-flow draws, expected-target TD
was 0.00952 and target sampling standard deviation was 0.00708, or 74.4% of that residual. Four
target samples gave TD 0.00997 versus 0.00952 for the 32-draw reference. The online actor produced
TD 0.00965 and target mean 0.288915, while the target actor produced TD 0.00952 and target mean
0.288895. The target means differ by only 0.000021, so target-actor lag is not the remaining cause.

The 50% recent-replay mixture did repair the R1 chronology imbalance. At the R2 endpoint, the
oldest replay decile has expected-target TD 0.00786, the newest has 0.00911, and the maximum is
0.01321 in the 30k--35k decile. The newest transition cohort is no longer uniquely underfit.

A frozen-policy/frozen-replay continuation then applied the exact R2 update rule for another 2,000
outer updates. Newest-decile TD fell from 0.00917 to 0.00797, but aggregate TD rose from 0.00962 to
0.01038 and mean absolute Q rose from 0.32987 to 0.36406. A 75% recent mixture worsened aggregate
TD further to 0.01126. More target samples, more repetitions of the same critic update, or more
recent replay are therefore not supported as R3 repairs.

An isolated target-timescale scan then retained the exact R2 critic update and changed only the
reward target critic Polyak coefficient. After 2,000 frozen-replay outer updates:

| reward target Polyak | uniform TD | mean abs Q | relative target gap |
|---:|---:|---:|---:|
| 0.00000 | 0.00795 | 0.32892 | 0.10213 |
| 0.00025 | 0.00811 | 0.33143 | 0.07932 |
| 0.00050 | 0.00823 | 0.33376 | 0.06378 |
| 0.00100 | 0.00846 | 0.33812 | 0.04652 |
| 0.00250 | 0.00930 | 0.34962 | 0.02935 |
| 0.00500 | 0.01060 | 0.36225 | 0.02117 |

Freezing the target minimizes the short counterfactual residual but creates 10.2% parameter lag;
0.00025 and 0.00050 likewise trade a small TD reduction for substantially more lag. Coefficient
0.001 is the first nonzero candidate that keeps relative target lag below 5% while reversing the
TD and Q-scale drift. It is a factor-five slowdown from the shared 0.005 coefficient.

A final four-way closed-loop diagnostic restored actor updates at interval 2 and optionally the
energy dual while holding replay fixed. With coefficient 0.001, final TD was 0.00835 with both
actor and dual frozen, 0.00841 with actor active and dual frozen, and 0.00835 with both active. The
corresponding 0.005 actor-plus-dual control ended at TD 0.00975 and mean absolute Q 0.37987, versus
0.34254 for 0.001. Actor/energy feedback is therefore not the causal difference in this window.

The selected R3 candidate changes only the reward target critic Polyak coefficient from 0.005 to
0.001. Cost-target and target-actor coefficients remain 0.005, and every R2 sampling, replay, UTD,
and actor-cadence choice remains unchanged. This candidate still requires implementation and a
new frozen protocol before any training run.

## Server evidence

The full run and checkpoints remain under
`/root/autodl-tmp/SAFE/runs/active/p4.2-r2-2332a55-b0-repair-50k-pilot/seed-2028`.
Compact evaluation and diagnostic evidence remains under
`/root/autodl-tmp/SAFE/artifacts/p4.2/backbone-r2-2332a55/pilot-seed-2028`.
SHA-256 seals and exact compact values are recorded in
`reports/star/p4_2_backbone_r2_pilot.json`; no checkpoint is committed to Git.
