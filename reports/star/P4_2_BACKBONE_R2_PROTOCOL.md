# P4.2 B0-R2 backbone tracking protocol

Status: preregistered before any B0-R2 training result.

## Parent decision and scope

B0-R1 seed 2028 completed, recovered task behavior, and failed the frozen reward-TD hard stop.
Its immutable decision is `continue_backbone_repair_and_pause_safety`; seeds 2027 and 2029 were
not launched. The R1 training source is `2c03c7ad07b36eaebab71affa8624080b6e4406f`, and the
compact parent report is `reports/star/p4_2_backbone_r1_pilot.json`.

B0-R2 training source is exactly `2332a550cc0a366c436c9a116b72b7bfc918bef2`. No Bellman
probability critic, dynamic geometry, candidate probing, continuation beyond 50k, or formal
four-environment run may start while the R2 three-seed gate is closed.

## Frozen diagnosis

The read-only final-replay diagnostics are sealed by SHA-256 in the parent report. On 4,096
evenly spaced transitions:

- stochastic target-flow sampling standard deviation was 0.00670, 64.2% of the expected-target
  TD residual; four target samples reduced one-sample observed TD by about 10%;
- expected-target TD was 0.00750 in the oldest replay decile and 0.01640 in the newest, locating
  the remaining error in current-policy tracking;
- at 1,000 frozen-policy outer updates, four samples plus two reward updates reduced aggregate TD;
- adding a 50% recent-10k reward mixture balanced oldest/newest TD at 0.00761/0.00805, whereas 75%
  recent degraded aggregate and old-replay coverage.

These are post-hoc capacity diagnostics, not promotion metrics. They select the intervention but
do not change the registered endpoint or reward-TD gates.

## Frozen R2 intervention

R2 changes only reward-backbone tracking:

1. each reward Bellman target averages four independent stochastic target-flow actions, applying
   `min(Q1,Q2) - alpha_E * energy` before the expectation;
2. each outer update performs two reward-critic gradient updates but one target-network Polyak
   update;
3. reward-critic and actor state batches are separated from safety-category stratification;
4. each separated batch is 50% uniform resident replay and 50% samples from the newest 10,000
   ring-buffer transitions.

R1's target flow actor and actor interval 2 are retained. Global safety and local geometry remain
disabled. Environment reward/done semantics, actor objective, energy dual, network widths,
learning rates, outer UTD entitlement, random/terminal-utility warmup, evaluation seeds, and
0/50k checkpoints are unchanged. The manifest must record all R2 fields, and full checkpoints
must record 96,000 reward-critic updates in addition to 48,000 outer updates.

## Pilot and hard stops

Run seed 2028 from scratch to exactly 50k. It may count toward the final three seeds. External
evaluation uses fixed stochastic flow bases and seeds 12027--12046 for 20 episodes at checkpoint
0 and 50k. Do not launch seeds 2027 or 2029 if any pilot hard stop is true:

- a required artifact is non-finite or structurally invalid;
- endpoint episode success is zero;
- endpoint stall is at least 0.5;
- the last four 5k reward-TD values rise strictly and final/historical-minimum is at least 2.

The machine decision is then `continue_backbone_repair_and_pause_safety`; otherwise it is
`launch_remaining_r2_repair_seeds`.

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
