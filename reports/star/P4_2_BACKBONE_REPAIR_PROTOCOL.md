# P4.2 B0-R1 backbone repair protocol

Status: preregistered before any B0-R1 training result.

## Parent decision

The immutable B0/B1 decision produced by `scripts/star/summarize_p4_2_backbone.py` is
`repair_backbone_and_pause_safety`. Its training source is
`f4371320a1683bc2130cd687213ffbac2d7da9d9`; `final_p4_verdict` remains null. No Bellman
probability critic, dynamic geometry, probing, 100k continuation, or four-environment formal run
may start while this repair gate is closed.

## Diagnosis and intervention

Offline decomposition of all three B0 full checkpoints found:

- failure-stratified replay samples had lower reward TD than normal samples, so replay
  stratification is not the primary instability;
- the flow-energy subtraction was approximately 0.003--0.005 at 50k and is too small to explain
  the late TD rise;
- time-limit transitions were approximately 0.1% of replay and are not the primary contributor;
- online-versus-target reward-value lag was approximately 0.011, the same order as the final
  reward TD, while reward backups used the changing online flow actor.

B0-R1 therefore makes exactly two coupled stability changes:

1. reward Bellman backups use a Polyak target flow actor;
2. the online actor and energy dual update once per two critic updates.

Global safety and local geometry remain disabled. Reward semantics, replay sampling, energy
objective, UTD entitlement, random warmup, network widths, optimizer learning rates, and the
0/50k endpoint protocol are unchanged.

## Movement construct correction

The failed parent report is retained unchanged. Its endpoint displacement field measured the
straight-line distance from episode start to episode end. That value confounds locomotion with a
successful multi-goal path that returns near its start.

B0-R1 additionally records every consecutive simulator displacement and gates on
`mean_step_displacement`. The old `mean_start_end_displacement` remains in every artifact as a
diagnostic. This metric definition is frozen before B0-R1 results.

## Budget and decision rules

The first run is seed 2028 at 50k. It is a feasibility pilot but uses the final frozen
configuration and may later count as one of the three required seeds. Do not launch the other two
seeds if the pilot is non-finite, has zero endpoint success, has endpoint stall at least 0.5, or
triggers the unchanged reward-TD divergence rule.

If the pilot clears those hard stops, run seeds 2027 and 2029 from scratch to 50k. The three-seed
gate in `scripts/star/summarize_p4_2_backbone_repair.py` requires all of the following:

- pooled fixed-random-flow success rate strictly rises from 0 to 50k;
- pooled return strictly improves;
- pooled stall does not rise;
- pooled mean step displacement does not decline;
- no seed has a strictly rising final four 5k reward-TD values with final/historical-minimum at
  least 2.

Only all-five pass yields `proceed_to_bellman_probability_critic`. Otherwise the decision is
`continue_backbone_repair_and_pause_safety`. External evaluation uses the fixed seeds
12027--12046 and fixed stochastic flow bases, 20 episodes per checkpoint. No run may continue
beyond 50k under this protocol.
