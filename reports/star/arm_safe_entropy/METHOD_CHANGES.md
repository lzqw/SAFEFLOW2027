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

## Development/main separation (main has not started)

The runner now labels ability, obstacle-development, mechanism-control and main
phases explicitly. Seeds2030/2031/2032 are rejected for development/engineering;
main is restricted to the specified three methods,1M transitions and100-episode
evaluation. `main_plan.json` is explicitly **not frozen**, so it cannot launch
main runs. A later genuinely frozen config must record the method code commit,
common settings and evaluation schedule. Default main layout seeds start62027,
disjoint from development52027; changing a frozen schedule on resume is rejected.
No main seed has been trained or used to select a parameter by these checks.

Resume checks preserve training-seed identity; legacy checkpoints require their
native run manifest when seed metadata is absent. Formal resumes must be from
the same main seed/config, not a development checkpoint. This adds ordinary
run-config checks, not a historical evidence repair or disaster-recovery system.

The compact collector reads phase identity from native manifests, retains every
development checkpoint (including weak ones), and writes only completed final1M,
100-fixed-layout evaluations to MAIN_RESULTS.csv. It refuses duplicate formal
method/seed rows or mixed frozen configs, rather than metric-selecting a run.
Main results remain header-only:0/9 completed. Only lightweight protocol parsing
and pure argument/config checks were performed; no new simulation validation,
formal training, extra seed or long matrix was launched.

On5080 a single replacement queue waited for SAC-Lag2028's complete300k
receipt, launched LearnedDir2028 once, and exited. It keeps the existing learner count,
uses explicit>8GiB free VRAM/disk and CPU headroom, and permits shared busy GPU
utilization instead of assuming exclusivity. Its source267e848 leaves the
selected learner optimizer and task settings unchanged.

Mechanism output now also records the actual path-reference normal/tangent
eigenvalues, separately from covariance projected onto the shared safety normal.
This distinction is essential for Random-Aniso: averaging random directions
can make common-axis reference covariance isotropic even while each path has
anisotropic eigenvalues. Matching must be checked from these measured spectra,
not asserted from a shared formula alone. Full control matching and trained
mechanism outcomes remain pending. Oracle curves explicitly label the
controller-Jacobian backend so the figure alone does not imply an exact upper bound.

Oracle2028 now shares A100 GPU0 with Oracle2027 after an actual free-capacity
inspection. It uses the same controller approximation/settings and original300k
budget, source3fd8bbf; only save cadence is1000 rather than10000 transitions.
Both have productive updates and recovery checkpoints. LearnedDir2028's separate
path has likewise produced finite local-margin/reward/cost/actor updates; unused
oracle flags carried by the generic queue do not instantiate an oracle for that
method (`geometry_source=learned_margin`, `oracle_backend=null` in its manifest).
No held-out directional-method advantage or full mechanism validation is claimed.

## Fixed100k direction diagnosis and engineering recovery (2026-11-06 14:30)

The matched Iso/Learned seed2028 diagnostic explicitly loads `policy_100000.pt`
and restricts replay to the first100k samples. The256-record interim pool has252
distinct states and177 valid shared exact normals. Near-boundary endpoint normal
variance decreases27.2% versus Iso, tangent3.7%, with mean normal cosine.9074.
Safe-interior tangent also decreases26.4%; this negative finding is retained.
Native100k success is28/30 versus Iso30/30, training violations112 versus107.
Therefore the diagnosis supports reference-to-action transmission, not an
overall method gain or readiness for formal freezing. No parameter revision
or new training configuration follows from this partial result.

Learned2028's5080 process then disappeared at196k, with a recoverable190k full
checkpoint. No Python traceback or queried kernel OOM/Xid evidence establishes
the cause. A100 GPU1 was chosen for recovery because4080 was busy and5080 disk
was tight. Source65de8b0 preserves learning settings; checkpoint cadence is now
1000. Seed/method/task/optimizer/replay/RNG identities are retained. New physical
episodes are explicit, not claimed exact continuation. The original190k..196k
tail is preserved in raw logs, with6000..7000 lost actual transitions and at
least one lost violation. Unknown unlogged safety cost is not invented as zero.
The collector labels resumed training counters as retained-lineage-only via a
small sidecar; no new disaster-recovery framework or historical repair was added.

## Mechanism-control measurement plan (not launched)

`mechanism_control_plan.json` predeclares how the Iso-Low scalar will be measured
from both fixed300k LearnedDir seeds, not selected from control returns or the
more favorable100k seed. A shared256-distinct-state calibration pool uses128
near-boundary/128 safe-interior states and128 samples per state. The global
scalar matches the exact-common-normal reference variance averaged over valid
near-boundary states, with equal weighting of the two seeds; it does not match
total trace or every state's variance. The scalar remains null until measured.
Random-Aniso's nominal fixed-trace/eigenvalue rule is implemented, but own-data
head learning means numerical spectra are not yet proven matched; identical
formula is not reported as achieved empirical matching. Both controls remain
behind the useful-signal stage decision. This plan starts no training and adds
no parameter sweep. Formal seeds2030/2031/2032 remain untouched.

## Cross-seed100k failure diagnosis (2026-11-06 15:35)

Learned2027's first100k evaluation stalled safely (0/30 success,0 violations),
unlike Learned2028's28/30. All four Iso/Learned100k policies were therefore
compared on one256-distinct-state visited pool, with114 valid near-boundary and
66 valid interior shared exact normals,128 actions/state. No new training or
parameter revision was used. Interim pool sampling now also excludes duplicate
observations; the old252-distinct-state pair analysis remains unchanged.

Near-boundary Learned2027 endpoint normal/tangent change is-23.3%/-46.4%
relative to Iso2027; Learned2028 is-33.8%/+9.7% relative to Iso2028.2027's
reference q=.1543 and native eigenvalues.035371/.042315 do NOT yield common-axis
normal contraction (projected variance.039797 versus Iso.04).2028's stronger
q=.3448 does (common normal.033231). The shared source-action cosines.7778/.8336
do not prove alignment at all flow midpoints. This distinguishes native
spectrum changes from actual risk-axis allocation, without claiming a complete
causal diagnosis. Different source pools change tangent conclusions, including
the earlier negative2028 interior result; both are retained. Full curves and
Oracle results are needed before a targeted revision or formal freeze.

Future main/mechanism exports also carry the training-counter scope and native
resume-lineage caveat when present. The development recovery's unknown lost
cost must not silently become an exact count in another table. Current main
and final-mechanism files remain header-only; this metadata propagation is not
a new evaluation, completed formal run, or repair of the missing tail.

## First completed Learned seed and qualitative delivery (16:30)

Learned2028 completed its fixed300k target,30/30 safe successes and0/30
violations/collisions, return11.4718. Restored-lineage updates are298k learner,
596k reward and149k actor; the new segment collected110k transitions. Its206
training violations are retained-lineage-only and do not remove the unknown
lost tail. Iso2028's final return11.4770 and zero violations remain the matched
baseline. This single completed seed is not a formal freeze decision.

The qualitative renderer loads only the same behavior actor/local geometry,
without critic reranking, filtering, a second behavior policy or training.
Two fixed300k policies were each rendered once on layout52049 with fresh common
sampling seed902027, and3x video slowdown. Both succeeded safely (Iso18 control
steps, Learned17), totaling35 qualitative transitions and zero training
transitions. They are explicitly not a replay of the earlier200k failure or a
replacement for native30-episode evaluation or future100-episode primary results.
No repeated success search
or environment revalidation was performed.

## Learned2027 reaches its second fixed evaluation (17:00)

Without any parameter change after the100k stall, Learned2027 at200k obtained
30/30 safe successes,0/30 violations/collisions, return11.4764,183 training
violations. Matched Iso2027 has30/30 safe successes,0/30 violations, return11.4376,
195 training violations. The12-count (6.2%) reduction is exact for this
uninterrupted seed, but small and interim, not a final population-risk claim.
The300k budget continues. Oracle2027/2028 remain live at58k/54k and have no100k
held-out results yet. No scientific revision, extra seed or formal freeze was
introduced in response to the new checkpoint.

## Completed Learned comparison and final-policy diagnosis (19:00)

Learned2027 completed300k with30/30 safe successes,0/30 violations/collisions,
return11.4663 and223 training violations, exactly Iso2027's final training
cost. Its small200k cost advantage did not persist. Learned2028 also completed
with30/30 safe successes, but its206 counter remains incomplete for actual
collection because of the lost tail. No stable final safety advantage or
formal-freeze condition is inferred from task mastery alone.

A fixed300k four-policy analysis uses256 distinct shared visited states,
128 actions/state and211 valid exact common normals (115 near,96 interior).
Near-boundary endpoint normal/tangent changes versus each Iso seed are
-2.1%/+5.4% for2027 and-10.5%/-11.2% for2028; interior tangent also contracts
for both. Normal cosines.7951/.8108 and margin RMSE1.46/1.35mm do not establish
accurate pathwise geometry. This preserves the distinction between reference
eigenvalues, covariance along a shared actual risk axis and final actions.
It is not the final control-inclusive mechanism table; no extra training was
collected. Exact simulator geometry is diagnostic only, never used by the
Learned actor during its own training or evaluation.

The previously declared calibration rule yields Iso-Low variance.0358835181
from the two final Learned seeds' common-normal reference means, equal weighting
by seed and valid near-boundary states. The value is measured, not tuned to
returns, and matches neither total trace nor every state's noise. The plan
still forbids automatic launch: useful final benefit is not established.
Oracle2027/2028 remain active at78k; their first held-out results are pending.

## Internal-point geometry and a bounded offline activation probe (19:50)

A16-state/four-path/four-midpoint diagnostic exposes the actual geometry-use
locations without changing actor sampling. The exact teacher is invoked only
after Learned paths are generated. Its valid-point cosine averages.4510/.6151,
versus.7266/.8182 at source actions of the same16 states. Teacher-inactive
points103/91 out of256 per seed still receive mean learned q=.2807/.1024.
Value error remains small (2.31/1.39mm); good scalar fitting is not sufficient
evidence of a correct action derivative. Active/inactive predicted norm ranges
overlap, so no unsupported numerical-floor increase is applied.

One fixed5000-update offline ReLU/SiLU comparison uses actual replay labels,
identical initial parameters/batches/masks per seed and excludes exact probe
observations. Teacher data are evaluation-only. SiLU's axis-cosine-squared and
value RMSE do not improve over matched fresh ReLU students in this budget;
signed-cosine improvement alone is not meaningful for a sign-invariant
diffusion axis. Both are less fitted than the native online heads, so this is
not a universal activation claim. The candidate is not promoted: no new
policy training, parameter sweep or main-seed use follows. The default ReLU
implementation remains numerically unchanged. Extra midpoint trace output is
diagnostic-only and does not alter behavior or path energy.

These512 midpoint rows (416 unique finite-difference queries,2912 cloned
controller intervals) and20000 offline auxiliary optimizer steps are reported
separately from actual environment transitions:
both probes add zero. Offline head files remain on the server. A read-only
half-hour resource watcher observes the two existing Oracle PIDs and three
GPUs while analysis proceeds; it performs no automatic interventions or new
launches and exits after both watched completion receipts exist.

## First Oracle100k results retained for both seeds (21:50)

Controller-Jacobian Oracle2027 obtains30/30 safe successes,0/30 violations,
return11.3601,104 training violations versus Iso137 (24.1% fewer). Oracle2028
obtains0/30 safe successes,0/30 violations, return-2.1947,66 training violations
versus Iso107; this is safe stalling, not a beneficial safety/performance
tradeoff. Neither the successful seed nor its opposite is used to select a
new default prematurely. Both retain their full300k budgets and all native
episode records. The approximation's privileged model access and earlier
accuracy limitations still apply. The direction mechanism is not declared
proved across seeds and the formal plan is not frozen.

## R1 online revision: rho.25 to.50 only (23:19 launch)

The supplied conservative-Oracle branch motivates two fresh100k short runs,
seeds2027/2028, with gentler normal suppression. All other scientific settings
remain unchanged, including the privileged controller approximation, fixed
trace.12, paired energy, task, UTD, replay, dual and learning rates. Each short
run completes its predeclared100k budget; original Oracle seeds continue300k.
This adds200k actual transitions, projected upper bound2,951,808 for the first
stage. No automatic extension, primary control launch or formal freeze follows.

Both run on A100 GPU1, source767357e, with2 envs/32 model workers. GPU0 retains
the original pair. The4080 is busy with foreign work and5080 disk is tight;
neither is cleared or altered. PIDs454684/454738 are real native processes,
but initial2000-step warmup rows alone are not a post-warmup completion check.

Rho is now explicitly passed, saved and restored in native learner/full state,
manifests and diagnostic loaders. Legacy defaults remain.25; mismatched rho
cannot silently reinterpret weights. The main frozen-config check also requires
the actual rho. The one CPU-only algebra/state fixture checks positive trace.12,
paired energy and rho.5 restore/rejection with zero new environment steps.
Result columns/plots distinguish R1 from the original method/seed instead of
averaging or replacing it. The earlier SiLU prototype remains unpromoted.

At23:34 both real R1 full checkpoints load at4000 transitions, rho.5 in outer
and learner state,2000 learner updates,4000 reward updates and1000 actor
updates. Finite native loss/energy diagnostics confirm post-warmup progress;
this supersedes the startup-only caveat without inventing an early evaluation.
Both remain on the original100k revision budget; the original pair continues
300k. Only compact data are mirrored through the host to GitHub.

## Frozen-weight endpoint-anchor intervention, not an online revision (23:50)

The internal-gradient diagnosis motivated one bounded sampler intervention:
compute local geometry at the SAME actor's deterministic endpoint and hold that
detached reference through the stochastic path. Actor/head parameters, q rule,
rho and inverse-covariance energy formula remain unchanged; paired Gaussian
increments isolate the reference-evaluation scheme. The anchor is never
executed, scored against alternatives or used as a second behavior policy.

On the existing256-state/128-actions pool, Learned2027/2028 near-boundary final
normal variance changes+0.64%/+1.06%, tangent-0.19%/+0.04% versus native sampling.
No useful improvement is established, so it is not promoted to training. The
six samplers (two Iso, two native Learned, two counterfactual Learned) retain
separate records; nothing overwrites native results. There are zero new
environment steps, teacher queries or optimizer updates. Common axes are from
the original pool actions, not newly queried anchor truth. This does not test
retrained-anchor performance, episode risk, or the complete mechanism controls.
The four existing Oracle jobs continue; no further short run is launched.
