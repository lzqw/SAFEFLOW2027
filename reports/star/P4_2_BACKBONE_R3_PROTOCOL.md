# P4.2 B0-R3 reward-target timescale protocol

Status: preregistered before any B0-R3 protocol training result.

## Parent decision and scope

B0-R2 seed 2028 completed with strong task behavior and failed the unchanged reward-TD hard
stop. Its immutable decision is `continue_backbone_repair_and_pause_safety`; seeds 2027 and 2029
were not launched. The R2 training source is
`2332a550cc0a366c436c9a116b72b7bfc918bef2`, and the compact parent report is
`reports/star/p4_2_backbone_r2_pilot.json`.

B0-R3 training source is exactly `b3c490309d465b89a5ae5a6ed235835587074abb`. No Bellman
probability critic, dynamic geometry, candidate probing, continuation beyond 50k, or formal
four-environment run may start while the R3 three-seed gate is closed. Short implementation and
checkpoint-recovery smoke tests are not protocol runs and cannot enter any scientific decision.

## Frozen diagnosis

The read-only R2 endpoint diagnostics are sealed by SHA-256 in the parent report. They established:

- target and online actors produce essentially identical expected reward targets, ruling out
  target-actor lag as the residual cause;
- the R2 recent-replay mixture repairs newest-cohort underfit, while more repetitions of the same
  critic update increase aggregate TD and absolute Q scale;
- with actor, energy dual, and replay frozen, reward target Polyak 0.005 raises uniform TD from
  0.00963 to 0.01060 over 2,000 outer updates, whereas 0.001 lowers it to 0.00846 with 4.65%
  relative target-parameter lag;
- under restored actor interval-2 and energy-dual updates, coefficient 0.001 ends at TD 0.00835
  versus 0.00975 for 0.005; freezing actor or the energy dual changes the 0.001 result by less than
  0.00007.

These are post-hoc fixed-replay diagnostics, not promotion metrics. They select one intervention
but do not alter endpoint evaluation or the registered reward-TD rule.

## Frozen R3 intervention

R3 changes exactly one training mechanism: the reward target critic uses Polyak coefficient
0.001 after each outer update. The target cost critic and target flow actor retain coefficient
0.005. This is an independent reward-target timescale, not a change to network capacity, reward
scale, actor cadence, or UTD entitlement.

All R2 choices remain fixed: target-flow reward backups, actor interval 2, four stochastic reward
target samples, two reward-critic updates per outer update, independent reward/actor batches, and
the 50% uniform plus 50% recent-10k replay mixture. Global safety and local geometry remain off.
Environment reward/done semantics, actor objective, energy dual, network widths, learning rates,
random/terminal-utility warmup, evaluation seeds, and 0/50k checkpoints are unchanged. The
manifest and training summary must record reward/cost/actor target coefficients 0.001/0.005/0.005,
and full checkpoints must record 96,000 reward-critic updates and 48,000 outer updates.

## Pilot and hard stops

Run seed 2028 from scratch to exactly 50k. It may count toward the final three seeds. External
evaluation uses fixed stochastic flow bases and seeds 12027--12046 for 20 episodes at checkpoint
0 and 50k. Do not launch seeds 2027 or 2029 if any unchanged pilot hard stop is true:

- a required artifact is non-finite or structurally invalid;
- endpoint episode success is zero;
- endpoint stall is at least 0.5;
- the last four 5k reward-TD values rise strictly and final/historical-minimum is at least 2.

The machine decision is then `continue_backbone_repair_and_pause_safety`; otherwise it is
`launch_remaining_r3_repair_seeds`.

## Three-seed promotion gate

If the pilot passes, run seeds 2027 and 2029 from scratch with the identical source and config.
The three-seed gate requires all five unchanged criteria:

- pooled fixed-random-flow episode success strictly rises from 0 to 50k;
- pooled return strictly improves;
- pooled stall does not rise;
- pooled mean step displacement does not decline;
- no seed triggers the reward-TD divergence rule above.

Only all-five pass yields `proceed_to_bellman_probability_critic`. Any failure yields
`continue_backbone_repair_and_pause_safety`. `mean_start_end_displacement` remains diagnostic;
the movement gate remains consecutive simulator `mean_step_displacement`. No seed may continue
beyond 50k under this protocol, and `final_p4_verdict` remains null.
