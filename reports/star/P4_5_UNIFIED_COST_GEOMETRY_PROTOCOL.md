# STAR P4.5 unified cost geometry protocol

Status: **locked before any P4.5 training**
Base: `076e8d389ecb6fb4050f1ca2db78997ae2b6342e`
Protocol SHA-256: `8abebc101021ec71a00fdf6d2c5580b5f97dada710e58de00936a75490e85c11`
Environment: `SafetyPointGoal1-v0`

P4.5 replaces P4.4's probability-normal execution rectification with one cumulative-cost critic
geometry used consistently by global safety, anisotropic diffusion, and the optional actor
direction loss. It completes all six `2 configurations x 3 seeds x 100k` runs unless an
engineering failure occurs. Reward-TD trends and temporarily poor scientific metrics are recorded
but never stop training.

## Locked geometry

Let the existing distributional ensemble's mean expected cumulative cost be

\[
Q_C(s,a,b,h)=\mathbb E_{\text{head},\tau}[Q_{C,\tau}(s,a,b,h)].
\]

The local normalized cost score and budget margin are

\[
r_C=Q_C/C_{\max},\qquad m_C=r_C-b,
\]

where `b` is the remaining-budget ratio already present in every STAR network context and
`C_max=10` for PointGoal. The safety normal is

\[
n_C=\frac{\nabla_a Q_C}{\|\nabla_a Q_C\|_2+10^{-8}}.
\]

The detached, scale-stable safety-band weight is

\[
w_C=\exp\left[-\frac12(m_C/0.10)^2\right]
\frac{\|\nabla_a r_C\|}{\|\nabla_a r_C\|+0.005}
\exp[-\operatorname{Std}_{\text{head}}(Q_C/C_{\max})/0.05].
\]

It contracts normal variance to at most `0.1` of the `0.04` reference variance while exactly
preserving total covariance trace. Probability-critic AUROC, Brier, skill, and calibration remain
diagnostics only: they do not affect behavior actions, replay boundary scoring, covariance,
direction loss, or any policy switch.

P4.4 execution rectification and forced inward drift are disabled. C1 instead adds

\[
\mathcal L_{\mathrm{dir}}=
\frac{\mathbb E[w_C[n_C^\top u_\theta]_+^2]}
{\mathbb E[w_C\|u_\theta\|_2^2]+10^{-8}},
\qquad \lambda_{\mathrm{dir}}=1.
\]

Normals and weights are detached, so the loss remains first-order in the raw learned midpoint
velocity. C0 and C1 differ only in whether this term is added.

## Runs and stopping policy

The paired arms are C0 (Q-cost anisotropy only) and C1 (the same policy plus direction loss).
Seeds are `2027`, `2028`, and `2029`. Every run targets exactly 100,000 environment steps. The
server schedule is four concurrent single-environment runs for seeds 2027/2028, then two for seed
2029.

Only NaN/Inf, CUDA OOM, corrupted checkpoints, environment/process failure, or absolute Q values
above `1e6` may stop one run. Scientific curves never gate subsequent runs.

Training logs are written every 5k. Lightweight checkpoints are preserved at
`0/10k/25k/50k/75k/100k`; full replay recovery is refreshed every 10k and retained at 100k.

## Fixed evaluation and evidence

Formal isolated evaluation occurs at `0/25k/50k/75k/100k`, using the same seeds `12027` through
`12046` and the same local CPU uniform-base/Wiener random stream for both arms. Trajectory seeds
are fixed in advance to `12027`, `12036`, and `12046`.

Performance evidence includes return, goal success, mean cost, CVaR-20, violations, training
cumulative cost, stall, displacement, path length, action norm, cost per successful episode, and
cost per unit path length. Cost-critic evidence includes TD, online/target Q scales, action-gradient
norm, empirical cost-to-go MAE, Spearman correlation, and five fixed equal-count risk bins.
Probability diagnostics remain separate.

Every formal checkpoint also records normal/tangent variance, their ratio, dangerous normal
velocity, outward ratio, directional energies, crossing, clipping, and safety-band coverage. Each
100k full checkpoint receives a 64-state x 256-action audit with common randomness:

| mode | learned drift | Wiener noise | clipping |
| --- | --- | --- | --- |
| A | off | on | on |
| B | on | off | on |
| C | on | on | on |

Four learning-curve figures and the three fixed-seed C0/C1 trajectory figures are produced. The
trajectory figures include hazards, goal, cost points, action direction, Q-cost normal, and tangent.

## Final interpretation

The report always exposes every arm-by-seed raw result. A safety trend requires C1 to lower final
mean cost, CVaR-20, and dangerous normal velocity in at least two paired seeds and in the pooled
three-seed result. Task performance is maintained when pooled success and mean return each fall by
no more than `0.10` absolute.

- **effective**: the safety trend is consistent and task performance is maintained;
- **safety benefit with performance cost**: the safety trend is consistent but task performance is
  not maintained;
- **no stable effect**: the safety trend is not consistent.

Large checkpoints, replay, and raw videos remain on the server. GitHub receives code, configs,
CSV/JSON evidence, figures, checkpoint SHA-256 identities, and the complete report. P4.4 reports
and the user-owned `METHOD/`, `PAPER/`, and `REF-PAPER/` trees remain untouched.
