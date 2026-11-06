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
division by `1+lambda_C`. Flow actor/reward samples are uniform+recent,
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
An exact controller-space finite-difference oracle is integrated as a selectable
training geometry. Full Oracle training is not launched: its measured geometry
query rate (138 state-actions/s in one bounded batch) is not an end-to-end
learner rate and would make the repeated flow-target calls expensive. Controls
and the full matrix are still pending.
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

## Coherent simulator task v2

Fresh EE forward kinematics replaces Bullet's lagging default link cache, and
every action labels the complete 40 ms interval even when a violation occurs
inside it. All obstacle learners restart on the same v2 task; old v1 pilots
remain intact with explicit engineering-stop markers. SAC-Lag's v1 100k
evaluation was 0/30 success and 0/30 violation. Do not explain that learning
failure causally by the API bug without evidence. On v2, SAC-Lag2027 at100k
obtained30/30 safe successes and0/30 violations; it is still an interim,
single-seed result, not a directional-method gain.

## Controller-Jacobian acceleration prototypes: not yet qualified for training

Retain actual interval-distance planes for all21 frames and10 protected links.
Numerical controller derivatives construct local first-order models, not a
learned safety head and not an online candidate-action selector. All reference
geometry is detached. Compare these approximations with the exact simulator
backend, preserving invalid gradients and mismatch rows:

| Prototype | Mean / maximum margin error | Valid on both / exact valid | Mean / minimum cosine |
|---|---:|---:|---:|
| Zero-action linearization | 0.373 / 2.911 mm | 15 / 26 | 0.977 / 0.872 |
| Nearby numerical action anchor, spacing0.5 | 0.091 / 0.893 mm | 24 / 26 | 0.888 / -0.052 |

Each comparison used the same bounded12-state/48-action probe, not a trained
policy evaluation. The first model missed11 exact-valid gradients. The second
missed2 and added2 spurious-valid gradients, but one direction was slightly
opposed to the exact normal. Neither has established adequate direction quality
for an Oracle upper-bound experiment. Both had zero violation-sign disagreements
in this small probe, which does not certify safety or repair the cosine issue.
The local model's113k/s fully cached arithmetic and383/s reused-state/new-action
rate exclude learner work and must not be advertised as training throughput.
Native details are in engineering/oracle-linearization-v2.json and
engineering/oracle-local-anchor-v2.json. The exact backend remains the truth
diagnostic; no approximate-Oracle training result is claimed.

Flow sampling can optionally return per-midpoint reference normal, weight and
eigenvalues. This prepares the required conditional mechanism analysis, without
changing default policy sampling or fabricating any mechanism result. Reference
and endpoint covariances must both be projected onto a shared normal across
methods, and must be estimated separately within each fixed state.
