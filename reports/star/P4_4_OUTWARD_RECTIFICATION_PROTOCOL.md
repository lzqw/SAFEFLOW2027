# STAR P4.4 safety-normal agreement and outward-drift rectification protocol

Status: **locked before any P4.4 pilot training**

Protocol version: `p4.4-outward-rectification-v1`

## Parent result and scope

P4.4 branches from commit `59b3493d4a69435e63c0d81bbc77121803912c91`. The P4.3
decision remains `hard_stop_reward_td_divergence`; its reports and evidence are immutable and are
not reinterpreted as a pass. P4.4 uses only `SafetyPointGoal1-v0` and asks whether the learned
reward-flow drift is undoing the benefit of safety-conditioned diffusion.

This round retains the P4.3 Bellman violation-probability critic, calibration formula, fixed-trace
anisotropic reference, global safety constraint, reward-backbone repair, optimizer cadence, and
network sizes. It does not add a distributional probability critic, candidate probing, Hessians,
an extra normal penalty, forced inward drift, another environment, or a longer training horizon.

## Read-only decomposition before training

The nine final P4.3 checkpoints are audited before any P4.4 training. For each checkpoint, the 64
finalized held-out transitions with smallest `|F_UCB - delta_C|` are selected with a stable replay
index tie break. Each selected state uses 256 action samples and common Gaussian Wiener draws
across the stochastic modes. The held-out replay action is the fixed base action.

The safety-reference inward drift is disabled in every decomposition mode so that the comparison
isolates the learned actor drift:

| mode | learned drift | Wiener noise | action clipping |
| --- | --- | --- | --- |
| A | off | on | off |
| B | off | on | on |
| C | on | off | on |
| D | on | on | on |

For every mode and state, the audit records endpoint normal variance `V_N`, per-tangent-dimension
variance `V_T`, total empirical covariance trace, crossing rate, mean normal displacement, action
clipping rate, and the pathwise learned-drift quantities

```
r_out = Pr[n^T u_theta > 0]
v_out = E[max(0, n^T u_theta)].
```

Clipping rate is the fraction of flow steps whose proposed endpoint has any action component
outside `[-1, 1]`. Mode A retains the raw proposal; modes B--D clamp it. Crossing means
`Pr[F_UCB(s,A_1)-delta_C > F_UCB(s,a_boundary)-delta_C]`.

The diagnostic also compares the two independently trained safety normals at the same held-out
state and replay action:

```
rho_normal = cos(grad_a F_UCB, grad_a mean_head Q_C).
```

A cosine is valid only when both gradient norms exceed `1e-8`. P4.4 training is blocked for critic
repair if any checkpoint supplies fewer than 32 valid cosine states or if at least half of all
valid cosines pooled across the nine checkpoints are negative. This is the preregistered meaning
of the two safety estimates being "often opposite". Mode-A anisotropy and the A-to-D change remain
diagnostic root-cause evidence rather than a tunable training gate.

## Complete held-out evidence

The deterministic episode split remains one held-out episode in every five. Held-out episodes are
excluded from the probability critic, reward critic, cost critic, actor, global dual, replay
normalization, and observation-normalizer updates. Replay-, held-out-, and external-evaluation
metrics remain distinct.

The evidence reader uses the most recent 20 complete finalized held-out episodes, up to 20,000
transitions. A transition limit may discard whole oldest episodes but may never split an episode.
Evidence is `sufficient` only with at least 256 positive and 256 negative labels. Otherwise its
state is `insufficient_evidence`, the geometry calibration reliability is exactly zero, and hence
`q_C=0`. AUROC and Brier diagnostics may still be reported when mathematically defined, but they
cannot activate anisotropy or rectification while evidence is insufficient. Training-batch
AUROC/Brier values never activate the geometry.

## One-sided learned-drift rectification

R1 changes the learned velocity at each stochastic-flow midpoint to

```
u_rect = u_theta - q_C [n^T u_theta]_+ n,
[x]_+ = max(x, 0).
```

Thus `q_C=0` is exactly the P4.3 learned drift, while `q_C=1` deletes only the outward normal
component. Tangent velocity and inward normal velocity are unchanged. The Euler--Maruyama update
is

```
dA_tau = (u_rect + b_C) d tau + L_C dW_tau,
```

with `b_C=0` in both P4.4 arms. The control energy is computed from the actually executed
`u_rect`, using the same inverse normal/tangent covariance weighting as P4.3. No additional normal
loss or coefficient is introduced. Raw `u_theta` and executed `u_rect` path diagnostics are both
retained.

## Arms and 30k pilot

Both arms restart from zero with seed 2028:

| arm | anisotropic diffusion | complete held-out evidence | outward rectification | inward drift |
| --- | --- | --- | --- | --- |
| R0 | yes | yes | no | no |
| R1 | yes | yes | yes | no |

The single RTX 3090 server executes R0 and R1 sequentially to avoid resource interference. This
does not change their seeds, update entitlements, or evaluation randomness. Each pilot stops at
exactly 30,000 environment steps. Formal external evaluation uses seeds 12027--12046, including a
uniform base draw and Gaussian Wiener increments. The final held-out audit uses the same 64 states,
256 samples, and four decomposition modes defined above.

The pilot passes only if all conditions hold:

- R1 mode-D executed `v_out` is at most 0.50 times R0;
- R1 mode-D crossing is at most 0.90 times R0;
- R1 mode-D `V_T` is at least 0.95 times R0;
- R1 external goal-met episode rate is no more than 0.10 below R0;
- R1 external stall ratio is no more than 0.10 above R0;
- R1 mean cost is no higher than R0;
- R1 CVaR-20 is no more than 1.10 times R0;
- every Reward-TD and reward-Q hard-stop condition below passes;
- each arm's final held-out evidence contains at least one positive and one negative label.

Failure yields `stop_after_seed_2028_pilot`; no other seed may launch.

## Reward-TD and numerical hard stops

The P4.3 strictly-rising-tail rule is replaced before training. A run hard-stops if any condition
holds:

- a required TD, gradient, Q value, diagnostic, or decision input is NaN or infinite;
- any 5k diagnostic `Critic/RewardTD` exceeds 0.02 in absolute value;
- final Reward-TD divided by the historical minimum exceeds 3 and the ordinary least-squares
  slope of the final four 5k diagnostic points is positive;
- online reward-Q or Bellman reward-target absolute magnitude exceeds 10 at any logged interval.

A mild rise in the last four points that does not satisfy the combined ratio-and-slope rule is a
warning only.

## Conditional three-seed expansion

Only a passing pilot authorizes continuation. The R0/R1 seed-2028 checkpoints continue in place
from 30k to 50k; they are not restarted. R0/R1 seeds 2027 and 2029 then train from zero to 50k,
giving exactly two configurations by three seeds. There is no automatic run beyond 50k.

The final screen passes only if all conditions hold:

1. pooled R1 mode-D crossing is lower than R0;
2. pooled R1 mode-D executed `v_out` is lower than R0;
3. pooled R1 mode-D `V_T` is at least 0.95 times R0;
4. pooled R1 external mean cost is no higher than R0;
5. pooled R1 external CVaR-20 is no higher than R0;
6. pooled R1 success is no more than 0.10 below R0;
7. at least two of three R1 seeds have held-out AUROC at least 0.65;
8. at least two of three R1 seeds have strictly positive held-out Brier skill;
9. no run has a non-finite TD, gradient, or Q value or triggers another numerical hard stop.

This remains a directional mechanism screen, not a significance claim, global safety guarantee,
or authorization for a broader environment suite.

## Explicit exclusions

P4.4 does not modify P4.3 evidence or `METHOD/`, `PAPER/`, or `REF-PAPER/`. It does not add a
distributional safety-probability critic, candidate probing, a Hessian term, forced inward drift,
an additional normal penalty, a new environment, 200k/300k training, five seeds, or a SHAPO
four-environment experiment.
