# Arm safe entropy: implementation and development changes

Starting branch commit: `a251b7af4864212c7191e2ff80393d0f62936805` (P4.7 code plus all20 native results and
diagnosis). New branch: `codex/arm-safe-entropy-v1`. The user's dirty source
worktree and all historical navigation results are preserved. No rollback,
navigation resume or historical provenance repair is a prerequisite.

New independent arm runtime follows the existing bounded midpoint flow,
target-flow reward backups (4 stochastic target samples), two reward critic
updates, actor interval2, target actor/reward Polyak .001 and uniform+recent
reward/actor replay. Gaussian SAC uses conventional squashed-normal sampling,
log-probability correction, automatic temperature and twin Q backups. Its
implementation is visible in models.py, not labelled as an unmodified SB3 run.
No second behavior actor or action candidate reranking is introduced.

Before the first ability run, two explicit design corrections:

1. Remove independent uniform base actions from Brownian arm flow. Their
   variance 1/3 per axis exceeds the .04 reference variance and could mask the
   very endpoint directional covariance being tested. Initial base is zero;
   the SDE still samples stochastic actions and all behaviors/targets/evaluations
   use that one policy. Raw proposals are clipped to action bounds and clipping
   is logged; there is no separate safety action projection.
2. Match reference energy units: E_D=E_kin/.04,
   alpha_D=.04*alpha_kin and target_D=target_kin/.04. Initial alpha_kin=.02,
   target_kin=.25 per action coordinate; alpha_D=.0008 and total energy
   target=18.75. The legacy target12.5 per coordinate exceeds the flow's bounded
   kinetic maximum2 per coordinate at max velocity2, so it is not carried over.
   The original log-space automatic energy-dual update is retained.

Ability implementation has geometry_source=none, no global cost optimization
and no obstacle. Separate obstacle paths now implement two sigmoid cost Q
heads for the binary episode event, projected completed-episode EMA dual
(budget .10, initial0, lr.05, EMA.05, cap100), and an unnormalized
`alpha_E*E - Q_R + lambda_C*Q_C` actor objective, without the legacy
division by `1+lambda_C`. All actor/reward samples are uniform+recent,
independent of any safety-prioritized state distribution (none used initially).
Random-warmup violations remain counted, but dual updates begin only once
learning has started, to avoid saturating the multiplier on short random
failure episodes before any learned policy exists. These are initial safety
settings, not a retrospectively improved result: no obstacle learner had run
under the earlier draft dual settings.

Two64-unit local-margin heads and strictly normalized first-order action
gradients are implemented. Active q=sigmoid(mean margin/.02)*positive cosine
agreement, detached; near-zero head/mean gradients disable geometry. The
legacy gradient/numeric-disagreement factors are diagnostic-only. No learned
policy calls the oracle. This path has not yet produced trained-method evidence.
Oracle training integration, controls and the full matrix are still pending.
Directional loss, inward drift and safety projection stay off in arm-v1.

One necessary initial-policy reference check measured covariance diagonal
[.00996,.05410,.05447] against [.010,.055,.055], using a fixed known normal
at one state. It also checked zero-gradient handling and geometry stop-gradient.
This is an engineering check, not evidence about a trained policy or task gains.

One2400-transition engineering flow run and a200-new-transition save/resume
were completed. The first restoration exposed CUDA RNG byte tensors moved
by map_location; restore now explicitly moves generator-state bytes to CPU.
Failed preflight output is retained and was not relabelled a scientific run.
Engineering training total so far2600; all large replay stays on the server.

Minimum validation only: three task clips and reset/cost/true-terminal checks,
unit-normal/trace/stop-gradient checks when geometry is implemented, and one
short learner save/resume. No historical matrix test suite or disaster-recovery
framework is added. Latest replay/checkpoints stay on A100, compact evidence
goes remote→host→GitHub.
