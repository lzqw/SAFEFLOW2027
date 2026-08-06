# STAR P4.3 safety-conditioned stochastic-flow protocol

Status: **locked before any P4.3 v2 formal training**

Protocol version: `p4.3-safety-anisotropy-v2`

Revision provenance: the v1 engineering queue was stopped after only S0 seed 2027 reached its
5,000-step diagnostic and before any S1/S2 run began.  Inspection showed that requiring positive
Brier skill inside the continuous reference weight could make anisotropy identically zero even
with useful discrimination.  That engineering artifact is retained separately but is excluded
from every v2 decision.  All nine formal runs start from zero under this sealed v2 protocol.

## Question and causal arms

This screen asks whether a single stochastic flow policy can preserve exploration tangent to a
learned safety boundary while reducing risk-normal exploration.  It is not a comparison between
multiple policies and it does not use an action selector.  All three arms train the same Bellman
violation-probability ensemble so that S0 -> S1 changes only the reference covariance:

| arm | stochastic flow | safety anisotropy | inward drift | candidate probing |
| --- | --- | --- | --- | --- |
| S0 | isotropic | no | no | no |
| S1 | isotropic-to-anisotropic continuously | yes | no | no |
| S2 | isotropic-to-anisotropic continuously | yes | yes | no |

The shared backbone is B0-R3 plus the B1 global safety constraint: target flow actor, four reward
target samples, two reward-critic updates per update, 50/50 uniform/recent reward replay, and reward
target Polyak coefficient 0.001.  SafetyPointGoal1-v0 is the only environment.  Training seeds are
2027, 2028, and 2029; every arm stops at 50,000 environment steps.  Formal external evaluation uses
the 20 fixed seeds 12027--12046 and includes both the uniform base draw and the stochastic-flow
Wiener increments.  No run is extended automatically.

## Frozen stochastic reference

The unified safe-risk tolerance is `delta_C = 0.20` and
`m_C(s,a) = F_UCB(s,a) - delta_C`, with ensemble UCB coefficient 0.5.  At every flow midpoint,

```
n = grad_a F_UCB / max(||grad_a F_UCB||, eps)
P_N = n n^T
P_T = I - P_N
q_C = g_B r_grad r_ens r_cal
g_B = exp(-m_C^2 / (2 sigma_B^2))
r_grad = ||grad_a F_UCB|| / (||grad_a F_UCB|| + 0.005)
r_ens = exp(-std_head(F) / 0.05)
r_cal = sqrt(clip((AUROC - 0.5) / 0.25, 0, 1)
             * exp(-BrierScore / 0.25))
```

`sigma_B = 0.10`.  Missing or non-finite held-out evidence, invalid negative Brier score, or
chance-or-worse AUROC sets `r_cal = 0`; otherwise Brier error attenuates reliability smoothly.
Weak gradients, ensemble disagreement, and poor calibration therefore return the policy
continuously toward the isotropic reference.  There is no positive-Brier-skill geometry gate.

The per-dimension isotropic variance is `nu_0 = 0.04`, `nu_min = 0.004`, and the total trace is
fixed at `B = d * nu_0`:

```
nu_N = (1 - q_C) nu_0 + q_C nu_min
nu_T = (d nu_0 - nu_N) / (d - 1)
D_C = nu_T P_T + nu_N P_N
L_C = sqrt(nu_T) P_T + sqrt(nu_N) P_N
```

The midpoint Euler--Maruyama update is

```
A[k+1] = clamp(A[k] + dt * (u_theta + b_C) + sqrt(dt) * L_C epsilon[k], -1, 1)
```

with `epsilon[k] ~ N(0,I)`.  S0 fixes `q_C=0`.  S1 uses the covariance above and `b_C=0`.
S2 additionally uses

```
b_C = -0.20 q_C sigmoid(m_C / 0.05) n.
```

The actor pays only for its residual learned velocity, not the inward reference drift:

```
E = 0.5 integral (||P_T u_theta||^2 / nu_T
                  + ||P_N u_theta||^2 / nu_N) dt.
```

The old isotropic energy is multiplied by `1 / nu_0 = 25` in S0.  To retain the exact B0-R3
constraint scale at initialization, the target energy is therefore changed from `0.5*d` to
`12.5*d` and the initial energy multiplier from 0.10 to 0.004 in every arm.  This is a fixed
algebraic reparameterization, not a hyperparameter sweep.

## Preregistered measurements

Each final full checkpoint is audited on held-out replay only.  The 64 held-out transitions with
smallest `|m_C|` are fixed (stable replay-index tie break); at least 32 must lie within
`|m_C| <= sigma_B`.  For every state, 256 actions are sampled with preregistered local RNG seeds.
The audit reports both the complete policy distribution and a conditional distribution that fixes
the base action to the held-out near-boundary action and varies only Wiener increments.

For the boundary normal evaluated at the held-out action, the audit reports empirical `V_N`,
`V_T`, Gaussian plug-in `H_N`, `H_T`, total trace, mean normal displacement, predicted unsafe
rate, and

```
crossing_rate = Pr[m_C(s, A_1) - m_C(s, a_boundary) > 0].
```

Task metrics are return, cost, cost CVaR-20, goal-met episode rate, and stall ratio.  Critic and
optimization diagnostics include held-out AUROC/Brier skill, boundary counts, reward TD error,
reference trace error, reliability components, and inward normal drift.

## Frozen causal decision

Protocol validity requires all nine runs to reach exactly 50,000 steps with finite diagnostics,
the exact arm flags above, no probing, identical non-arm settings, and empirical reference trace
relative error at most `1e-5`.

The covariance mechanism passes only if pooled conditional audits satisfy all of:

- S1 `V_N <= 0.85 * S0 V_N`;
- S1 `V_T >= 1.05 * S0 V_T` and `H_T > S0 H_T`;
- S1 crossing rate is at least 0.02 lower than S0 or at most 0.90 times S0;
- the analytic matched-normal-variance isotropic-low counterfactual has lower total Gaussian
  entropy than the safety-anisotropic reference;
- every S1/S2 seed supplies at least 32 in-band boundary states, and at least two of three seeds in
  each arm have final held-out AUROC >= 0.65 and positive Brier skill.

The drift mechanism passes only if pooled S2 has lower mean normal displacement and no higher
crossing rate than S1.  Its pooled mean cost and CVaR-20 must not exceed S1, goal-met episode rate
may be no more than 0.10 below S1, and stall ratio may be no more than 0.10 above S1.

S1 and S2 each retain task functionality only if goal-met episode rate is no more than 0.10 below
S0 and stall ratio no more than 0.10 above S0.  Reward-TD divergence, any non-finite training
quantity, or a functionality failure is a hard stop.  With three training seeds these are directional
mechanism gates, not a claim of statistical significance or a final P5 result.

## Theory scope

For a locally Gaussian reference with fixed trace `B` and risk-normal variance cap
`n^T D n <= nu_N < B/d`, maximizing differential entropy is equivalent to maximizing
`log det D`.  The optimum uses the full normal cap and equal tangent eigenvalues
`(B-nu_N)/(d-1)`, exactly the covariance above.  First-order risk variation obeys
`Var[dF] = ||grad_a F||^2 nu_N dt`; tangent variance does not enter.  At the same normal-risk
variance, the anisotropic reference has `(d-1)/2 * log(nu_T/nu_N) > 0` more Gaussian entropy
than an isotropic-low reference.  These are local first-order statements; they do not constitute a
global safety guarantee.

The FLAC basis is Lei Lv et al., *FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge
Matching*, arXiv:2602.12829.  The paper explicitly formulates the iterative policy as an SDE from a
uniform or Gaussian prior and relates drift energy to divergence from a Brownian reference; its
published implementation uses an isotropic energy, identifying anisotropic/state-dependent energy
as future work.
