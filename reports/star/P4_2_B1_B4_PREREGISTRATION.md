# STAR P4.2 B1–B4 preregistration

Status: **locked before any B1–B4 mechanism-screen training**

Protocol version: `p4.2-b1-b4-v1`

Preregistered screen seed: `2028`

Environment: `SafetyPointGoal1-v0`

## Frozen prerequisite

The screen starts only because the B0-R3 backbone gate passed at commit `401c7ca`. The passing
training source was `b3c490309d465b89a5ae5a6ed235835587074abb`; its three-seed decision is stored
in `reports/star/p4_2_backbone_r3_decision.json`. Every arm below uses the same R3 reward backbone:
target flow actor, actor interval 2, four reward-target samples, two reward updates per outer
update, 50% uniform plus 50% recent-10k reward replay, and reward-target Polyak `0.001`.

## Locked 50k causal screen

All four arms use seed `2028`, one vector environment, UTD 1, 5k logging, full recovery
checkpoints every 10k, and checkpoints at steps 0 and 50k.

| Arm | Global safety | Bellman probability | Dynamic geometry | K=16 probing |
|---|---:|---:|---:|---:|
| B1 | yes | no | no | no |
| B2 | yes | yes | no | no |
| B3 | yes | yes | yes | no |
| B4 | yes | yes | yes | yes |

The exact override files are `configs/star/p4_2_b1_global.yaml`,
`configs/star/p4_2_b2_bellman.yaml`, `configs/star/p4_2_b3_dynamic_geometry.yaml`, and
`configs/star/p4_2_b4_candidate_probing.yaml`. Their SHA-256 fingerprints and the source commit
must be captured in the launch receipt; a mismatch invalidates that run.

## Probability data contract

- Bellman target precedence is breach `1`, safe episode end `0`, otherwise target probability
  critic under the target actor. It is undiscounted because it models an eventual event.
- Remaining budget is signed and untruncated. `violation_so_far` is absorbing until reset.
- Complete-trajectory MC labels have auxiliary weight `0.10`; they are not the primary target.
- Every fifth complete episode is held out, with assignment fixed before its first transition.
- Held-out transitions are excluded from all network optimizer samplers, the global dual, and
  observation-normalizer updates. The gate reads at most the most recent 20 held-out episodes and
  4096 transitions.
- Replay-training, held-out, and isolated external metrics are emitted under distinct fields and
  artifacts. No one may relabel one split's metric as another split's result.

## Dynamic geometry contract

`curriculum_stage` is monotonic. `geometry_enabled` is reversible and reads held-out evidence
only. A pass requires all of the following at a 5k evaluation:

- mean action-risk gradient norm at least `0.001`;
- mean probability-head disagreement at most `0.10`;
- at least 128 effective boundary samples;
- at least 128 positive and 128 negative MC calibration labels;
- Brier skill strictly above `0`;
- AUROC at least `0.65`.

Three consecutive passes enable geometry. Two consecutive failures disable it. Critic and actor
training continue while geometry is disabled, and the curriculum stage never rolls back.

## B4 probing contract

B4 draws exactly 16 stochastic flow actions in W2/W3. It removes candidates whose risk UCB is
above `0.5` or whose disagreement exceeds `0.10`, selects the closest trusted safe-boundary
candidate, and uses reward Q only within a risk tie band of `0.02`. If no safe candidate exists,
it selects the lowest-risk trusted fallback; if every candidate is untrusted, the unavoidable
emergency fallback is the lowest-risk candidate overall. Tangent noise has scale `0.05`, is allowed
only while the held-out geometry gate is open, and is accepted only if the perturbed action remains
trusted and safe.

## External evaluation and causal decision

Step-0 and step-50k checkpoints are evaluated in the isolated environment using fixed random-flow
bases and seeds `12027` through `12046` (20 episodes). The center-flow result remains diagnostic
only. Training-time evaluation is disabled so the four training jobs have matched simulator work.

An arm is performance-eligible if it completes exactly 50k without non-finite values, its external
goal-met episode rate is no more than 0.10 below B1, its stall ratio is no more than 0.10 above B1,
and reward TD is not classified as divergent. Divergence means both: the final reward-TD value is
more than 1.5 times the minimum logged value and the final four 5k values strictly increase.

Eligible arms are ranked lexicographically by:

1. lower external cost CVaR-20;
2. lower external mean episode cost;
3. higher external goal-met episode rate;
4. lower external stall ratio;
5. for Bellman arms, higher held-out Brier skill then AUROC.

Exactly the top two arms extend to 100k × seeds `2027, 2028, 2029`. Seed 2028 may resume its exact
50k full checkpoint; seeds 2027 and 2029 start from scratch with identical source and config. If
fewer than two arms are performance-eligible, remaining slots use the same ranking among completed
arms and are explicitly marked fallback selections. No 200k run, four-environment run, or P5 claim
is authorized by this screen.

## P5 guard

B4 can support entry toward P5 only if it avoids success collapse, has lower stall than P4.1,
non-divergent reward TD, positive held-out Brier skill, held-out AUROC at least 0.65, adequate
boundary samples in every seed, lower dangerous normal velocity, and improves cost/CVaR over B1
without a material success loss. `final_p4_verdict` remains `null` throughout P4.2.
