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

## Controller-Jacobian approximation: accuracy limits and an explicit training path

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
opposed to the exact normal. Neither establishes an error-free Oracle upper
bound. Both had zero violation-sign disagreements
in this small probe, which does not certify safety or repair the cosine issue.
The local model's113k/s fully cached arithmetic and383/s reused-state/new-action
rate exclude learner work and must not be advertised as training throughput.
Native details are in engineering/oracle-linearization-v2.json and
engineering/oracle-local-anchor-v2.json. The exact backend remains the truth
diagnostic; no trained-method efficacy result is claimed.

The more relevant common visited pool subsequently compared256 actual replay
state/actions,128 near-boundary and128 safe-interior, without dropping any case.
Of199 exact-valid normals,195 were valid in both models,4 were missed,4 extra
model-valid normals appeared. Mean cosine was0.97147, near-boundary mean0.96531,
near-boundary10th percentile0.96562; minimum near-boundary cosine was-0.32428.
Mean margin error0.0895mm, maximum0.8815mm, sign disagreements0/256. This supports
using a **labelled privileged-controller approximation as an experimental
diagnostic**, not declaring its occasional wrong direction solved. In particular,
a negative result with this backend alone cannot rule out a perfect-direction
mechanism. Exact simulation remains the independent truth comparison.

To make this diagnostic runnable, discard only distance planes provably unable
to win the lower envelope throughout their numerical action cell (a plane's
lower bound exceeds the smallest global upper bound). The linearized model is
unchanged, not replaced with a new margin fit. Variable-sized plane sets are
cached and only padded per GPU batch. In the200-transition resumed short run,
46,900 cached state/anchors used11.0MB numeric plane storage rather than157.6MB
for all210 planes. Full training's explicit40M-entry cap is an entry bound, not
a claim that Python overhead or late-run memory is zero; inspect actual memory.

An engineering run collected2008 transitions (8 post-warmup learner updates),
then restored learner/replay/RNG and collected200 more;2008+200=2208 new train
transitions, not4216. Restored counters reached208 learner updates,416 reward
updates and104 actor updates; backend identity is checked on resume. Both
runs saved and evaluated successfully; their two-episode engineering outputs
are not included in development efficacy tables. The resumed200 transitions
took100.22s before final evaluation (~2.0 transitions/s) and114.23s including
evaluation/save (~1.75/s), with32 CPU oracle workers. This small replay/cold-cache
window does not establish full300k throughput. Do not quote the warmup-inflated
first-run FPS as an Oracle training speed. Compact receipts are retained under
engineering/oracle-training-v2-{smoke,resume}/.

## Implemented mechanism controls (not yet trained)

`flow_iso_low` requires an explicit scalar variance measured from LearnedDir's
mean normal reference and records the source run; no arbitrary variance is
silently selected. It matches normal noise, not total trace. Its noise/energy
pair uses alpha_D=nu_low*alpha_kin and target_D=target_kin/nu_low. No geometry
head is needed for the scalar control.

`flow_random_aniso` uses the same local-margin q rule and eigenvalue/trace
formula as LearnedDir but an independent random unit normal, sampled once per
generated action and held throughout all midpoint steps. Each separately
trained method has its own margin heads, so q values need not numerically match
across trained checkpoints; measured reference eigenvalues must be reported,
not assumed identical merely from the formula. No new behavior policy exists.
One algebra-only check confirms fixed-in-action random normals, matching trace,
paired inverse energy, and Iso-Low rescaling. Its illustrative0.015 variance is
not an experimental selection. Do not launch the control matrix before useful
directional evidence or populate mechanism results from this engineering check.

Flow sampling can optionally return per-midpoint reference normal, weight and
eigenvalues. This prepares the required conditional mechanism analysis, without
changing default policy sampling or fabricating any mechanism result. Reference
and endpoint covariances must both be projected onto a shared normal across
methods, and must be estimated separately within each fixed state.

The analysis runner now has two explicit modes: interim diagnosis may inspect
currently available foundation/directional snapshots with their unequal ages
visible; final-mechanism mode requires both2027/2028 fixed300k results from
Iso/Oracle/Learned plus both mechanism controls. It balances256 distinct real
visited states across contributing runs and clearance regions, uses128 actions
per state, never pools states for covariance, and preserves undefined normals.
Oracle policy sampling uses the SAME approximate/exact backend as its training;
independent truth risk/normal queries use exact simulation. Random-Aniso is not
assigned its unused learned-head direction as an active aligned normal. The
final CSV links native episode evaluation, training violation counts, source
commits, backend and valid comparison counts. The final mode was checked only
for rejection of an incomplete cohort; full trained-cohort analysis is pending.

Counterfactual Oracle intervals are model-information queries, not extra
on-policy replay transitions: they never train the learned margin heads.
Their counts are explicitly separate from collected-training-transition budgets
in Oracle receipts. Evaluation transitions are recorded separately as well.
Neither model-query compute nor privileged geometry should be omitted when
describing the Oracle diagnostic's information/computational advantage.

The5080 extension shares only read access to its existing PyTorch/CUDA install;
new simulation dependencies live in a separate venv under LZQW. The new A/B
seed2028 jobs have the same physics, observations, rewards, action scale, replay,
UTD and update ratios as seed2027; later code adds metadata/control options but
leaves their default behavior unchanged. Torch2.7.1+cu128 on5080 differs from
Torch2.7.1+cu118 on A100 and is recorded, not claimed to be identical hardware.
