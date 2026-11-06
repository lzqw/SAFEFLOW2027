# Arm safe entropy — development in progress

Start date: 2026-11-06. Starting commit `a251b7af4864212c7191e2ff80393d0f62936805`; independent branch
`codex/arm-safe-entropy-v1`. This is a new simulated Panda task, not a resumed
navigation matrix. Old outcomes remain unchanged. Current success targets are
development aims, not fabricated results or scientific stopping gates.

## Execution order

1. Three environment clips and minimal semantics checks.
2. Seed2027 unobstructed SAC / reward-only flow, initially100k transitions each;
   optionally200k if learning trends justify continuing.
3. SafeReach obstacle A/B first, then Oracle/Learned direction once foundations work.
   Two development seeds2027/2028,300k each, evaluation100k/200k/300k on common30 layouts.
4. At most two targeted revisions, four extra short runs at most; no broad sweep.
5. If useful: Iso-Low and Random-Aniso controls and fixed-state conditional action
   covariance diagnostics; freeze before formal new seeds2030/2031/2032.
6. Only after first-task evidence: payload extension and two perturbation families.

First-stage training cap is3M actual transitions including ability and engineering
smokes; no vector-env step-count multiplication ambiguity. Later larger stages
are conditional on useful signal, not automatically launched by the queue.

Initial resources: A1002x80GB idle,128 CPU threads; other workloads occupy
4080/5080 and are not interrupted. Python environment isolated in
`/home/lzqw/SAFEFLOW2027_ARM/env`; server root initially about60GiB free.
First platform is panda-gym/PyBullet, for real link-distance labels and exact
same-controller counterfactual geometry. Simulation FPS is not learner FPS.

## Current evidence and recovery (through 2026-11-07 05:30 CST)

All four v2 SAC-Lag/Flow-Iso development runs completed their fixed300k budgets.
Each final evaluation obtained30/30 safe successes and0/30 violations; this is
not evidence of zero population risk. The direction comparison is still open.
Both LearnedDir seeds have also completed300k with30/30 safe successes and0/30
violations/collisions. Thus six of the original eight v2 development runs
are complete; the two Oracle approximations remain live.

| Method | Seed | Fixed300k return | Training violations |
|---|---:|---:|---:|
| SAC-Lag | 2027 | 11.4191 | 219 |
| SAC-Lag | 2028 | 11.4195 | 152 |
| Flow-Iso | 2027 | 11.4569 | 223 |
| Flow-Iso | 2028 | 11.4770 | 274 |
| LearnedDir | 2027 | 11.4663 | 223 |
| LearnedDir | 2028 | 11.4718 | 206, retained lineage only |

**The completed Learned comparison does not yet demonstrate a stable gain.**
Learned2027's final training cost223 is exactly Iso2027's223: the12-count
advantage at200k did not persist through300k. All four Iso/Learned final
evaluations have30/30 safe successes and0/30 violations, with almost identical
returns. Learned2028's206 counter still excludes an unknown lost tail. Do not
turn the earlier100k geometry signal or200k cost gap into a final Pareto claim.
Learned2027 completed300k actual transitions without interruption in17173.49s,
with298k learner,596k reward and149k actor updates. No bad seed was restarted or
discarded, and no result was selected from its intermediate checkpoints.

At05:30 Oracle2027 completed its fixed200k evaluation on the common30 layouts:
30/30 safe successes,0/30 violations/collisions, return11.4436, and171 training
violations. The matched-budget seed2027 comparison is:

| Method, seed2027 at200k | Safe success | Evaluation violations | Return | Training violations |
|---|---:|---:|---:|---:|
| SAC-Lag | 30/30 | 0/30 | 11.4292 | 162 |
| Flow-Iso | 30/30 | 0/30 | 11.4376 | 195 |
| OracleDir, privileged controller approximation | 30/30 | 0/30 | 11.4436 | 171 |
| LearnedDir | 30/30 | 0/30 | 11.4764 | 183 |

Oracle's count is24 lower than Iso (12.3%) at matched safe success, but9 higher
than SAC-Lag. This is a second positive milestone for one development seed,
not an evaluation-risk improvement, a two-seed gain, or a final300k result.
Learned2027's own earlier200k training-cost advantage disappeared by300k, so
interim reductions must not be promoted into a frozen-method claim. Oracle2028
has not yet reached200k; its retained100k evaluation remains0/30 successes.
All four live PIDs were observed directly at05:30: original Oracle2027/2028
at200k/180k, R1 at69k/80k. The compact capture contains30 native evaluations,
with formal rows still0/9. No budget, configuration or queue expansion changed.
A100 host memory remained34GiB available and disk60GiB free; this observation
does not promise future capacity or authorize interference with foreign jobs.

At19:00 Oracle2027/2028 were both live at78k, not yet at their first held-out
100k evaluation. Available A100 RAM95GiB and disk60GiB show no present storage
or memory pressure. Final Iso/Learned inputs are backed up on the4080's4T disk;
original server files are retained. Main remains unfrozen and no new training
configuration, control matrix or formal seed was launched.

Learned2028's completed receipt records298k learner updates,596k reward updates,
149k actor updates, and110k newly collected transitions after the190k restore.
The resumed segment took6763.35s. Its final evaluation is from the fixed300k
policy, not the best checkpoint. Both it and Iso2028 have0/30 final violations
and almost identical returns; there is no demonstrated final evaluation-risk
advantage. The206 retained-lineage training violations omit at least one
known lost-tail violation and an unknown unlogged tail, so no exact actual-cost
percentage improvement is claimed. Earlier100k and200k failures remain visible.

At16:30 the remaining process PIDs270049/297991/329031 were alive and updating:
Oracle2027/2028 at52k/49k, Learned2027 at178k. The latter's recent training
success was99/100, but its next held-out node remains200k. Completing one seed
does not establish cross-seed stability or authorize formal freezing. No
scientific early stop, replacement seed or new training was introduced.

Two [qualitative final-policy clips](qualitative/final300k-seed2028-layout52049/)
use the fixed300k Iso/Learned2028 policies, layout52049 and fresh common sampling
seed902027. Each was rendered once: Iso reached safely in18 steps, Learned in17,
both cost0. The videos are3x slow playback, not extra training or primary
evaluations. This layout was chosen because Learned's native200k evaluation
had failed there; the new draws are **not** a reconstruction of that episode.
One successful clip per method is not comparative efficacy evidence. Original
three environment demonstration clips and all quantitative failures remain.

At17:00 Learned2027 reached its fixed200k evaluation:30/30 safe successes,
0/30 violations/collisions, return11.4764 and183 training violations. Iso2027
at the same200k budget had30/30 safe successes,0/30 violations, return11.4376
and195 training violations. This seed therefore has a small, exact training
count reduction of12 (6.2%) at matched task success, not an evaluation-risk
advantage or a final two-seed gain. Its earlier100k failure is learning delay,
not evidence that it never learns; both checkpoints remain in the curve.
It continues through the full300k budget without a hyperparameter change.

All three unfinished process PIDs remained alive at17:00: Learned2027 at209k,
Oracle2027/2028 at58k/54k. The Oracle methods have not yet reached their first
held-out100k node, so their low training costs/success rates are not a result
table or a scientifically justified early-stop reason. Controls remain untrained
and the formal plan remains unfrozen. No extra training matrix was launched.

LearnedDir2028 at its fixed100k evaluation reached28/30 safe successes,
0/30 violations, return10.4282 and112 training violations. Iso2028 at100k had
30/30 safe successes,0/30 violations and107 training violations. Thus there is
no demonstrated task-preserving safety improvement at this matched milestone.
The two safe timeouts and the earlier SAC2028 failure remain in native evidence.

At the15:00 process check all four unfinished runs were alive. The compact
capture at15:02 recorded Oracle2027/2028 at35k/29k, Learned2027 at88k and the
resumed Learned2028 at224k. No process was restarted or added at this check.
The4080 was free again (5% GPU utilization,1.6TiB free disk); no unneeded
training was launched merely to fill it.

Learned2028's new fixed200k evaluation obtained29/30 safe successes and1/30
clearance violations, with no contact collision. The failed common layout was
seed52049, episode length18; return averaged10.7823. Iso2028 at the same200k
milestone obtained30/30 safe successes and0/30 violations, return11.4408.
Learned's retained-lineage training violation counter was167 versus Iso244,
but this is **not** an exact actual-collection cost reduction: Learned's lost
tail includes at least one additional violation and an unknown unlogged tail.
The new evaluation is valid for the resumed policy but does not establish a
Pareto gain, population safety, or the efficacy of a completed two-seed method.
No scientific early stop or parameter change follows from one30-episode point.

### Frozen-weight geometry-anchor intervention: not adopted

One sampling-only intervention tested whether moving local-geometry queries
from stochastic flow midpoints to the same actor's deterministic endpoint,
then holding that detached reference through the path, improves allocation.
It uses the same saved300k parameters, learned heads/q rule and paired latent
Gaussian increments; no second behavior network, candidate scoring/ranking,
projection or executed anchor action is introduced. Native sampling and both
Iso seeds are retained in the same diagnostic, on the existing256-state pool
with128 actions/state and the same common exact source-action axes.

| Learned seed, near boundary | Native endpoint normal | Anchor-CF endpoint normal | Native endpoint tangent | Anchor-CF endpoint tangent |
|---|---:|---:|---:|---:|
| 2027 | .012302 | .012380 | .012434 | .012410 |
| 2028 | .012777 | .012913 | .013102 | .013107 |

The normal changes are+0.64%/+1.06%, tangent-0.19%/+0.04%: no material selective
improvement is demonstrated. These are paired frozen-sampler measurements,
not a retrained policy, new success/violation evaluation or a claim about
geometry truth at the anchors. No new teacher query, actor update or environment
transition was collected. New Monte Carlo samples are not substituted for
earlier native results. This negative intervention does not prove that every
retrained anchor scheme fails, but provides no basis to promote one now.

See [intervention data](diagnostics/frozen-geometry-anchor-cf-v2/diagnostic.json)
and [covariance comparison](diagnostics/frozen-geometry-anchor-cf-v2/conditional-covariance-flow_learned-flow_learned_anchor_cf.png).
It stays outside primary/mechanism result tables. No additional online
revision follows: the original pair and the already declared R1 pair continue
their budgets while the next fixed evaluations are pending.

### R1: one bounded normal-suppression revision, original budgets unchanged

The first supported online revision changes **only rho_N:.25 to.50** in the
Oracle controller-approximation condition. Oracle2027 has a positive100k
signal, while Oracle2028 still stalls through116k; this tests the task-loss /
over-strong-normal-suppression branch of the supplied plan. This is not a claim
that the approximation is perfect or that rho explains all failures. The
negative offline activation candidate is not installed. No geometry, reference
trace, energy target, actor/critic learning rate, dual, reward or task change
is bundled into R1.

Both development seeds receive fresh, predeclared100k short runs,30 fixed-layout
evaluation episodes at the final100k node, no best-seed/checkpoint selection.
The original rho.25 Oracle runs retain their complete300k budgets. R1 is two
of the at-most-four additional short policy runs, not a replacement matrix or
formal/control launch. The upper-bound first-stage total becomes2,951,808
actual training transitions including engineering/lost-tail overhead, below3M.
See `configs/star/arm_safe_entropy/revision_r1_oracle_rho050.json`.

The00:00 half-hour watchers confirmed all four real PIDs alive. Logged original
progress was135k/130k, and R1 progress9k/10k. Across23:30..00:00 the logged
aggregate increased by about24k transitions (11k original,13k R1); coarse log
granularity, differing cache phases and workloads make this an operational
capacity observation, not a controlled speedup benchmark or a full-run ETA.
Both original targets remain300k and both R1 targets100k. No new evaluation
node has been reached, so recent training success is not substituted for one.
Only these four already declared jobs continue; no additional online trial is
launched in response to the negative frozen-anchor intervention.

At03:00 both half-hour watchers again observed the four actual native PIDs
alive: original Oracle2027/2028 at170k/158k and R1 at45k/53k. Original2028's
recent training success recovered to96/100, but this does not replace its
retained0/30 held-out100k result or predict the pending200k evaluation. No new
held-out node has been reached; native evaluation count remains29, formal
rows0/9. All four predeclared budgets remain intact. There is no new parameter
change, short-run addition, control matrix or formal launch during this wait.

At04:00 both read-only half-hour watchers again confirmed all four native
training PIDs alive. Original Oracle2027/2028 reached182k/167k, and the bounded
R1 pair reached55k/66k. Original recent training successes were100/100 and98/100;
R1's were13/100 and26/100. These are training-window observations, not fixed
held-out evaluations or grounds to select, stop or extend a seed. The compact
capture still contains29 native evaluations and0/9 formal rows. Original
targets remain300k and R1 targets100k; no new trial or diagnostic was added.
Whole-card A100 utilization was68%/59%, while4080 and5080 were75%/50%; these
shared-device figures are not project-specific utilization or a throughput
benchmark. Foreign workloads were left untouched. The next comparison awaits
the original200k and R1 final100k nodes; original Oracle2028's failing100k
result and all earlier negative Learned evidence remain visible.

At23:15 the4080 was97% foreign-GPU-utilized with2.3GiB free VRAM, and5080 had
only9.7GiB disk free. Both R1 learners therefore started at23:19 on the idle
A100 GPU1 (92GiB available host RAM,60GiB disk free), two learners on that card;
the original pair stays on GPU0. Each uses32 controller workers and2 envs.
At23:21 native PIDs454684/454738 were alive with2000 warmup transitions each;
these initial rows are not evidence of productive post-warmup learning yet.
No foreign task was stopped, and no new environment installation was needed.

At23:34 both R1 native checkpoints reached4000 steps with rho.5 in both full
and learner state,2000 learner updates,4000 reward updates and1000 actor
updates each. Actor losses.2864/.3520 and energies22.67/18.41 were finite.
This confirms actual post-warmup learning and saving, not just startup rows.
There is no held-out R1 result yet. The live matrix, observed at05:30, is:

| Live Oracle group | Seed2027 progress | Seed2028 progress | Predeclared target per seed | A100 GPU |
|---|---:|---:|---:|---:|
| Original rho.25 | 200k | 180k | 300k | 0 |
| R1 rho.50 | 69k | 80k | 100k | 1 |

Source767357e records rho in native manifests, full checkpoints, policy state
and analysis/export metadata, rejects reinterpreting rho.25 checkpoints as.5,
and keeps main-freeze identity explicit. A CPU-only contract fixture verified
rho.5 actor/target checkpoint restoration, rejection of mismatched rho,
positive fixed-trace covariance and the paired inverse-covariance path energy
on a constant-velocity example. It collected zero environment transitions;
no environment demo or historical suite was repeated. A separate read-only
half-hour watcher follows both R1 PIDs without restarting them automatically.

### First Oracle100k evaluations, not final Oracle results

Both controller-Jacobian Oracle approximations have reached their first fixed
100k evaluation. They use privileged controller/robot geometry and are not an
error-free exact-model upper bound. The native records are retained for both
seeds, including stalling:

| Training seed | Iso100k safe success / training violations | Oracle100k safe success / training violations | Oracle return |
|---|---|---|---:|
| 2027 | 30/30 /137 | 30/30 /104 | 11.3601 |
| 2028 | 30/30 /107 | 0/30 /66 | -2.1947 |

Both Oracle evaluations had0/30 violations/collisions. For2027 the training
count is24.1% below Iso at matched perfect success (Iso return11.3359), a useful
single-seed signal. For2028 the lower training count accompanies safe stalling,
not a task-preserving improvement. This is not a stable two-seed Oracle gain,
and does not establish a final comparison. The earlier Learned2027 stall later
resolved, so neither an early failure nor a lower cost alone justifies stopping
the original300k budgets or selecting the favorable seed.

The21:30 watcher confirmed both actual Oracle PIDs alive; the21:49 native
capture records109k progress for both. They continue the original budgets.
No formal/control run is launched from these mixed interim results. Current
code, configs, compact native evaluations, diagnostics and reports are delivered
on branch `codex/arm-safe-entropy-v1`; full checkpoints/replay remain on servers.

### Internal-flow geometry probe and one bounded offline candidate

The final policies were probed on16 randomly fixed near-boundary states from
the existing pool, four generated paths per state and four actual flow
midpoints per path. Learned sampling was unchanged. Only after those paths
were generated did an independent exact-simulator finite-difference teacher
evaluate their margins/normals; it was never a Learned behavior callback or
training target. The512 queried midpoint rows contain416 unique state/actions,
requiring2912 cloned controller intervals (7 per finite difference), and zero
new training transitions. This is not another policy evaluation or a256-state
covariance estimate.

| Learned seed | Same16 source-action cosine | Internal-midpoint cosine | Internal margin RMSE | Teacher-gradient-inactive points |
|---|---:|---:|---:|---:|
| 2027 | .7266 | .4510 | 2.31mm | 103/256 |
| 2028 | .8182 | .6151 | 1.39mm | 91/256 |

Midpoint cosine uses153/165 valid comparisons respectively. Under the teacher's
finite-difference resolution and1e-7 validity floor, inactive gradients receive
no normal. Learned still applies mean q=.2807/.1024 on those inactive points.
Predicted gradient norm distributions on active/inactive teacher points overlap,
so simply raising a numerical norm threshold is not an established repair.
These observations show why good margin values and source-action cosine do
not guarantee correct geometry where the SDE actually uses it. The small
selected-state probe and changing valid-gradient support do not establish a
complete causal diagnosis or disjoint-layout generalization.

One ReLU-versus-SiLU local-head prototype was then fitted offline, using only
actual retained replay interval-margin labels. Each activation received fresh
identical weights within each seed, the same draws/bootstrap masks and exactly
5000 fitting updates; both heads stayed width64. The16 probe observations were
excluded from fitting (2/4 matching native replay rows). Exact teacher labels
were evaluation-only. No actor, critic, dual or native policy was changed.

| Offline activation | Seed2027 axis cosine squared | Seed2027 RMSE | Seed2028 axis cosine squared | Seed2028 RMSE |
|---|---:|---:|---:|---:|
| ReLU | .3827 | 9.81mm | .4294 | 5.87mm |
| SiLU | .3166 | 24.92mm | .3773 | 20.90mm |

For diffusion the normal sign is immaterial; a higher signed cosine alone is
not a successful axis repair. Within this deliberately small, fixed offline
budget, SiLU is not supported: axis agreement and value fitting did not improve.
Both fresh offline students also underperform the fully trained native heads;
this is not proof about SiLU after unlimited fitting. No epoch was selected or
extended to obtain a favorable outcome. The default remains ReLU; the optional
activation is used only by the isolated comparison, not a promoted online
method revision. Offline weights stay on A100 and do not replace run artifacts.

See [internal-point records](diagnostics/final300k-learned-path-geometry-v2/diagnostic.json)
and [fixed-budget comparison](diagnostics/offline-margin-relu-silu-v2/comparison.json).
Both analyses have zero new environment training transitions. Oracle2027/2028
remain live (87k/88k at19:50). A read-only resource watcher now records the three
GPUs and actual Oracle PIDs on half-hour clock boundaries; it never restarts
jobs, changes parameters or stops foreign work. Formal/control launch gates
remain unmet pending useful evidence and a supported targeted repair.
The20:00 watcher observation confirmed both Oracle PIDs alive at89k, before
their first fixed100k evaluation. No process was restarted on an observation
timeout or because another GPU became free.

### Fixed300k policy geometry and control calibration (not the full mechanism table)

The completed Iso/Learned2027/2028 policies were compared on one new shared
pool of256 distinct visited states (128 near-boundary,128 interior),128 samples
per state, with211 valid exact common normals:115 near-boundary and96 interior.
This uses the fixed final300k weights and full retained replay, not a favorable
earlier checkpoint. Undefined normals and the2028 resume-cost caveat remain in
the native JSON. It collected zero new training transitions.

| Near-boundary, final300k | Reference normal | Reference tangent | Endpoint normal | Endpoint tangent |
|---|---:|---:|---:|---:|
| Iso2027 | .040000 | .040000 | .012629 | .011886 |
| Learned2027 | .036149 | .041925 | .012366 | .012533 |
| Iso2028 | .040000 | .040000 | .014345 | .014981 |
| Learned2028 | .035618 | .042191 | .012838 | .013297 |

Tangent is per dimension. Relative to its Iso seed, Learned2027's endpoint
normal decreases2.1% and tangent increases5.4%; Learned2028's normal decreases
10.5% but tangent also decreases11.2%. In the interior both seeds contract
tangent variance (-8.1%/-13.4%). Native reference trace remains.12; near-boundary
path-average q=.2609/.2584 and raw learned-axis eigenvalues are approximately
.0322/.0439. The shared true-axis reference allocation and the final sampled
actions are not the same object. The final result does **not** establish
stable selective normal suppression with tangential exploration preserved.

Same-state/action near-boundary learned/exact normal cosine is.7951/.8108;
common-pool margin RMSE is1.46/1.35mm. Low value error does not prove accurate
action gradients or correct transmission through every internal flow point.
The current evidence does not isolate which issue is causal. Marginal Gaussian
entropy proxies, clipping and counterfactual interval risks remain in the JSON;
those mixed-pool risks are not on-policy episode violation rates.

The predeclared equal-seed, valid-near-state common-normal reference average
gives an Iso-Low scalar of**.0358835181**, not total-trace matching. It is now
recorded in `mechanism_control_plan.json`, without looking at control returns.
Measurement is not a successful method result or permission to launch another
long matrix: the useful-signal gate remains unmet. Random-Aniso numerical
matching, trained controls, Oracle conclusions and formal freezing remain open.
MAIN_RESULTS and MECHANISM_RESULTS remain header-only, rather than relabeling
this incomplete directional cohort as the complete mechanism experiment.

See [fixed300k native data](diagnostics/paired300k-bothseeds-v2/diagnostic.json),
[all-four covariance](diagnostics/paired300k-bothseeds-v2/conditional-covariance.png)
and [normal agreement](diagnostics/paired300k-bothseeds-v2/normal-agreement.png).
Earlier100k pools and their positive/negative findings below are preserved.

### Both-seed matched100k diagnosis (15:35, not a final mechanism result)

Learned2027's first100k held-out evaluation obtained0/30 safe successes and0/30
violations, return-1.6895, training violations74. This is safe stalling, not a
task-preserving safety win against Iso2027's30/30 successes/137 training
violations. By15:30 its ongoing116k run had begun recent training successes
(25/100 completed episodes); this is not a replacement held-out evaluation.
All four unfinished process PIDs were alive at that check. Full budgets remain.

One additional analysis on the free4080 compared BOTH Iso and Learned seeds
at the same100k policy milestone and replay prefix. The shared pool has256
distinct visited states,128 near-boundary and128 safe-interior;180 valid exact
common normals comprise114 near-boundary and66 safe-interior. All four policies
use this same pool/axis and128 independent action samples per state. Large
inputs remain on the4T disk. This is zero new training, not a new parameter run.

| Near-boundary conditional variance | Ref normal | Ref tangent | Endpoint normal | Endpoint tangent |
|---|---:|---:|---:|---:|
| Iso2027 | .040000 | .040000 | .017134 | .020694 |
| Learned2027 | .039797 | .040101 | .013150 | .011083 |
| Iso2028 | .040000 | .040000 | .016647 | .014681 |
| Learned2028 | .033231 | .043385 | .011014 | .016099 |

Tangent is per dimension. Learned2027 loses23.3% normal but46.4% tangent
endpoint variance versus its Iso seed: it does not preserve tangential
exploration here. Learned2028 loses33.8% normal while gaining9.7% tangent.
Mean same-state/action normal cosine near the boundary is.7778/.8336 for
Learned2027/2028; signed cosines and undefined cases are retained. Since the
diffusion uses an axis, cosine sign alone is not a safety-performance verdict.

The distinction between eigenvalues along the learned axis and covariance
along the COMMON true risk axis matters. For2027, near-boundary path-average
q=.1543 and the raw reference eigenvalues are.035371/.042315, but the common-axis
normal variance is.039797, almost isotropic. For2028 q=.3448, raw eigenvalues
.029655/.045173, and common-axis normal variance.033231. Thus nonzero q and a
changed native spectrum do not by themselves prove risk-aligned exploration.
The fixed source-action cosine is not a measurement of true geometry at every
internal flow midpoint, so this does not establish the precise causal error.

Results depend on the visited-state pool: the earlier2028-only pair pool below
showed tangent-3.7% near the boundary and-26.4% inside, whereas this expanded
four-policy pool shows+9.7% near and+6.8% inside. Both native analyses remain
visible, not selectively replaced. The more consistent observation is2028's
near-boundary normal reduction; global tangent preservation and cross-seed
efficacy remain unproven. Do not tune from a favorable subset of these states.

See [both-seed native data](diagnostics/paired100k-bothseeds-v2/diagnostic.json),
[cross-seed covariance](diagnostics/paired100k-bothseeds-v2/conditional-covariance-flow_learned.png)
and [cross-seed normal agreement](diagnostics/paired100k-bothseeds-v2/normal-agreement-flow_learned.png).
This localizes the next question to geometric activation/transmission stability,
not a blind weight increase. Oracle held-out results and complete Learned
learning curves are still pending; no scientific revision is launched yet.

### Earlier two-policy100k pool, not a final mechanism result

An interim analysis used the actual100k policy files for Iso2028 and LearnedDir2028,
and only their first100k replay prefixes. Later full snapshots provided storage,
not later policy weights or later training data. The shared pool contains256
records. The raw JSON retains an old generic `limitations` sentence about
different ages; its explicit requested checkpoint and per-policy metadata are
authoritative here: policy milestones and replay prefixes are matched, full
snapshot ages differ. The generic sentence is corrected for future analyses.
The shared pool contains256
records (252 distinct states);177 have valid exact common normals:101 near-boundary
and76 safe-interior. Undefined normals are retained, not replaced by an axis.
Each policy contributes128 action samples per state; covariance is conditional
on each state, never pooled across states.

| Near-boundary conditional variance | Iso | LearnedDir |
|---|---:|---:|
| Reference normal | .040000 | .031820 |
| Reference tangent, per dimension | .040000 | .044090 |
| Endpoint normal | .018166 | .013223 |
| Endpoint tangent, per dimension | .018275 | .017601 |

Near-boundary endpoint normal variance falls27.2%, while tangent falls3.7%.
Mean learned/exact normal cosine is.9074 in that region. This supports actual
transmission of the directional reference to final sampled actions, not just
a changed reference matrix. However, in safe-interior states tangent variance
also falls26.4% (normal13.7%), an important limitation. Margin RMSE is2.59mm near
the boundary and7.30mm inside on this visited-state pool, not a held-out fitting
benchmark. Mixed-pool one-interval risk queries are not on-policy episode risk.

See [conditional covariance](diagnostics/paired100k-seed2028-v2/conditional-covariance.png),
[normal agreement](diagnostics/paired100k-seed2028-v2/normal-agreement.png) and
[native diagnostic](diagnostics/paired100k-seed2028-v2/diagnostic.json).
This collected no new training transitions. It does not populate the final
MECHANISM_RESULTS table, prove superiority to Iso-Low, or justify freezing yet.

### Engineering interruption and environment choice

The5080 LearnedDir2028 process disappeared without a receipt; its last log was
196k/166 violations, while the usable full checkpoint is190k/165 violations.
There is no project Python traceback, and the queried kernel journal has no
OOM/Xid match; the cause remains unknown. The old status file saying running
is stale. Original files are preserved with an engineering-interruption note.

At14:25 A100 had61GiB free disk and97GiB available RAM; its GPU1 had ample
VRAM. The4080 had1.6TiB free on `/home/carla/4T`, but its GPU was96% utilized
by other work. The5080 root had only13GiB free. Accordingly, the existing
LearnedDir2028 run was resumed on A100 GPU1 from190k, targeting the original
300k, not a new scientific run or changed hyperparameters. Learner, optimizer,
replay and RNG were restored; physical episodes restart, so this is not bitwise
trajectory continuation. The new run saves every1000 transitions.

The discarded tail contains at least6000 actual transitions and one known
violation; the conservative transition upper bound is7000, with an unknown
unlogged violation tail. Resumed counters describe the retained checkpoint
lineage only, not the total actual collected safety cost. This caveat is
explicit in `resume_lineage.json` and the development CSV counter-scope field.
Projected full first-stage collection including this engineering overhead is
at most2,751,808 transitions, still under3M. Do not conceal the lost tail or
count a repeated transition as free.

Oracle2027/2028 and LearnedDir2027 remain active on A100. No foreign task was
stopped; no controls, formal seeds, payload training or robustness expansion
was launched. Configuration remains unfrozen. The4080 4T disk remains the
preferred large-analysis/archive location when needed; no large replay is
copied to GitHub or this host. The remainder below is an earlier execution log,
not a claim that its old PIDs/statuses are current.

## Earlier execution log (2026-11-06 13:10 CST)

Initial eight-run coverage is now2 completed,5 active,1 dependency-queued;
this is launch coverage, **not** completion of the scientific comparison:

| Method | Seed2027 | Seed2028 |
|---|---|---|
| SAC-Lag | Completed300k, A100 | Completed300k,5080 |
| Flow-Iso | Active, A100 | Active,5080 |
| OracleDir controller approximation | Active, A100 GPU0 | Active, same A100 GPU0 |
| LearnedDir | Waits for Flow-Iso2027 completion | Active,5080 |

The three required clips and minimum semantics checks passed:

| Fixed-layout demonstration | Control steps | Reached safely | Binary cost |
|---|---:|---:|---:|
| Unobstructed reach | 16 | yes | 0 |
| Straight path, obstacle present | 2 | no | 1 |
| Collision-free detour | 33 | yes | 0 |

Primary v2 videos and native step records are in [environment_v2](environment_v2/).
Historical [environment_v1](environment_v1/) evidence remains unchanged; the
v1 detour took34 steps. v2 fixes a stale EE pose cache and completes every full
control interval before returning terminal. All new obstacle comparisons use
v2; geometry, reward, action scale and layout distribution are unchanged.
True horizon is terminal (no bootstrap), and cost is binary. Geometry/energy
and one flow save/resume check passed; see [method changes](METHOD_CHANGES.md).
No all-history validation or new disaster-recovery system was introduced.

Both first ability learners have completed:

- SAC seed2027, source0eba984, A100 GPU0: completed100k training transitions,
  **30/30 successes at25k,50k,75k and100k**, fixed common layout seeds52027–52056,
  stochastic policy evaluation. Native end-to-end runtime913.35s,98k learner
  updates and98k reward/actor updates; full replay remains on A100.
- Reward-only flow seed2027, sourcefc6b3c0, A100 GPU1: completed100k transitions,
  **30/30 successes at25k,50k,75k and100k**; runtime2126.68s,98k learner updates,
  196k reward updates and49k actor updates.

These are **unobstructed ability** results, not obstacle safety or a formal
seed comparison. The full100k budgets were retained despite early successes;
80% is not a programmatic early-stop rule. Both remain the sole behavior
policy of their learner. [DEVELOPMENT_RESULTS.csv](DEVELOPMENT_RESULTS.csv)
labels interim checkpoints explicitly; formal/mechanism/robustness result files
remain header-only. A zero-shot API compatibility check evaluated both100k
ability policies on unobstructed v2: each again succeeded30/30, with no retraining.

One scheduling comparison uses actual productive post-warmup progress:
single SAC5k–10k:111.30 transitions/s; with both learners running,
SAC35k–45k:106.15/s and flow4k–8k:46.35/s, aggregate about152.50/s.
The workloads and sample windows differ: this is an actual heterogeneous
queue capacity comparison, **not** an algorithm speed comparison or a
controlled same-workload scalability claim. Initially keep two independent learners,
one per currently free A100; later use additional genuinely available host capacity.
Do not equate low VRAM use with permission to
duplicate scientific seeds or interrupt foreign jobs.

### Obstacle learning

The v1 SAC-Lag2027 pilot obtained0/30 successes and0/30 violations at100k:
safe stalling, not task mastery. For a coherent Oracle/baseline interface,
both v1 pilots were stopped and preserved after the API correction described
above: SAC last logged124k (unlogged conservative bound125k), Flow-Iso14k
(upper bound15k). Original statuses/progress, checkpoints and stop markers
remain intact; no completion receipt was invented. This is not evidence that
the API bug caused poor v1 learning. v1 and v2 efficacy must not be pooled.

Matched v2 baselines use sourcece0d175 for seed2027 and source3c1b407 for
seed2028, with300k budgets. The latter adds runtime metadata/options without
changing the selected baseline optimizer/physics settings:

| Method | Training seed | Latest native evaluation | Safe success | Violation | Status |
|---|---:|---:|---:|---:|---|
| SAC-Lag | 2027 | 300k,30 episodes | 30/30 | 0/30 | Completed; fixed final checkpoint |
| SAC-Lag | 2028 | 300k,30 episodes | 30/30 | 0/30 | Completed; retained earlier100k failure |
| Safe-FLAC-Iso | 2027 | 200k,30 episodes | 30/30 | 0/30 | Continues to300k |
| Safe-FLAC-Iso | 2028 | 200k,30 episodes | 30/30 | 0/30 | Continues to300k |

For seed2027 at a matched100k budget, SAC's return was11.1936 and training violations97;
Flow's return11.3359 and training violations137. SAC200k return11.4292,
training violations162. SAC300k return11.4191, total training violations219,
native runtime4190.35s and298k learner/reward/actor updates. For seed2028 at100k,
SAC return-1.8642/training violations32 versus Flow return11.3626/violations107.
The lower violation count of a stalled policy is not a task-preserving safety
improvement. Both Flow development seeds have now learned at the first fixed
evaluation; SAC's seed difference is retained, not hidden or treated as a final
300k failure. At200k SAC2028 reached30/30 successes, return11.3890 and training
violations119. Its earlier0/30 outcome remains in the table/curve, not discarded.
Flow2027 at200k retained30/30 successes, return11.4376 and training violations195.
SAC2028 finally completed300k with30/30 successes, return11.4195 and training
violations152 (native2595.74s,298k learner/reward/actor updates). Flow2028 at200k
also retained30/30 successes, return11.4408 and training violations244.
This is evidence of later learning, not a basis for an early scientific stop
or a premature multi-parameter change. These per-seed checkpoints do not prove zero population risk or
a directional-method gain. Flow's earlier
stalling was followed by genuine learning, so it was not a reason to terminate
the run or change several parameters prematurely. Foundation evidence now
supports moving to the two direction variants while retaining full A/B budgets.
The native [learning curves](figures/SafeReach-Obstacle-v2-learning.png)
separate held-out evaluations from recent training statistics; missing
evaluations are not plotted as zeros.

### Oracle and pending work

The exact simulator-replica oracle is integrated and a bounded interface check
passed: max margin discrepancy3.7253e-8m, minimum normal cosine1.0 among six
valid comparisons; live state unchanged. Exact queries are too costly to
launch the full repeated-target training path without acceleration work.
Two explicit controller-Jacobian approximations were examined. Nearby action
anchors reduced mean margin error to0.091mm but direction cosine averaged0.888,
with one slightly opposed normal and2 missed/2 spurious-valid gradients.
**This approximation is not qualified as an accurate Oracle upper bound.**
No Oracle/Learned training gain is claimed; all mismatch rows are retained.
See [method changes](METHOD_CHANGES.md) for the limited probes and their scope.

On256 actually visited common state/actions, the local model's mean cosine
was0.97147 (near-boundary0.96531),195/199 exact-valid directions were valid in
both, with4 misses and4 extras. Minimum near-boundary cosine remained-0.32428;
all rows, not just successful ones, are retained. Mean margin error0.0895mm,
sign disagreements0/256. The runnable training variant is explicitly a
**privileged controller-linearization approximation**, not an error-free
model bound. A negative result from it alone cannot disprove a perfect-direction
mechanism. Exact simulator finite differences remain the independent diagnostic.

Pruning provably inactive distance planes compresses the same local linear
model, rather than changing its predictions. Oracle2008-transition smoke and
200-new-transition restoration completed; actual resumed learner throughput
was about2.0 transitions/s before final evaluation, not the cached arithmetic
rate. The small-window end-to-end rate is not a full-run completion guarantee.
Both tests are labelled engineering, not included as trained-method results.

One dependency queue has launched and one remains waiting (sourcee792e45):

| Diagnostic | Status / dependency | A100 GPU | New budget |
|---|---|---:|---:|
| OracleDir, controller-linearization backend, seed2027 | Active PID270049;10k logged, checkpoint saved | 0 | 300k |
| OracleDir, same backend, seed2028 | Active PID297991;3k logged, checkpoint saved | 0 | 300k |
| LearnedDir, seed2027; no oracle calls by its actor | Queue PID252096 waits for Flow-Iso2027 to complete300k | 1 | 300k |

Queue251920 completed its one launch and exited; child270049 and its32
single-threaded controller workers were inspected alive. The3k→4k productive
window took313.91s (~3.19 transitions/s), not the warmup-inflated headline FPS.
This is an early cached-model window, not a guaranteed late-run rate or learned
Oracle efficacy; no held-out directional result exists yet. Each
queue rechecks shared GPU/CPU/disk headroom and launches only one run, not the
entire matrix. Later controls/formal/payload stages are not blindly enqueued.

At12:56 A100 CPU load was below3, GPU0 utilization3%, available RAM108GiB.
The original matrix's Oracle2028 therefore started on the same mostly-waiting
GPU0, source3fd8bbf,32 CPU model workers, with checkpoints every1000 transitions
because model-derived geometry is slow. This changes save cadence, not the
optimizer, task, reference or300k budget. Both scientific learners were checked
alive and updating; the extra run is not a new seed/configuration. Partly
overlapping productive windows were Oracle2027 8k→10k in556.33s (~3.59/s) and
Oracle2028 2k→3k in334.24s (~2.99/s). These small, different cache/data windows
are not a controlled scalability benchmark or guaranteed full-run throughput.
After initialization, available RAM was98GiB and CPU load about6.8, retaining
headroom; no other workload was stopped. At12:56 the4080 still showed95% foreign
GPU usage and was left alone.

### Additional shared capacity:5080

At12:14 the4080 remained98% utilized by foreign workloads, so it was left
untouched. The5080 had spare capacity and approximately14GiB disk free. A new
project lives under `/home/autodrive/LZQW/SAFEFLOW2027_ARM`, with its own venv;
it read-only reuses the old SAFE conda environment's working PyTorch2.7.1+cu128
instead of duplicating CUDA wheels. Panda-gym3.0.7, PyBullet3.2.7, Gymnasium1.2.3,
NumPy1.24.3 and SB32.7.0 are installed in the new venv, not the old project.
CUDA tensor execution and package versions were checked; no three-video
revalidation or historical suite was repeated. Space remains about14GiB.

The existing matrix's seed2028 A/B runs use source3c1b407 there:

|5080 learner|PID|Latest mirrored progress|Budget|
|---|---:|---:|---:|
|SAC-Lag2028|Completed; former3890030|Final300k evaluation|300k|
|Flow-Iso2028|3890092|200k evaluation available|300k|
|LearnedDir2028|3895948|9k, actual margin/critic/actor updates|300k|

Both have productive optimizer updates, not just live startup processes. The
first post-warmup7k→8k SAC and6k→7k flow windows yielded approximately100/s
and83/s respectively; these are short throughput observations, not full-run
ETAs or a controlled hardware benchmark. A foreign Isaac workload returned
while setup was underway and remains running (~3GiB VRAM at the observation).
Each new learner used~0.36GiB VRAM. The observed98% whole-card utilization
includes that foreign job and must not be attributed entirely to this project.
All sharing continues without stopping anyone else's work. Five scientific
learners are currently active across A100+5080, without increasing budgets or
claiming unobserved seed2028 success. Native runtime versions and GPU names
are recorded in new run manifests; the5080 CUDA version differs from A100's11.8.

After SAC-Lag2028 finished its full300k budget, replacement queue3893413
(source267e848) launched LearnedDir2028 once, childPID3895948, and exited.
The child was checked live, with finite margin/critic/actor updates through9k.
It has not produced a100k held-out result yet; a small margin loss is not proof
of an accurate normal or a directional gain. This preserves two project learners on that shared card,
not a third concurrent addition. The explicit16GB-card checks are free VRAM>8GiB,
CPU headroom and disk>8GiB; shared utilization up to100% is allowed, rather than
waiting for an exclusive idle card contrary to the user's sharing choice. The
project directory is about490MB; observed total disk free fluctuated12–14GiB.
No foreign files were removed. All eight original development runs are now
completed, live or explicitly queued; no control/formal/payload expansion is implied.

One interim foundation diagnostic used256 shared genuinely visited states
(128 with current-state clearance0–20mm above the safety threshold,128 farther
inside),128 independent actions per state, and the exact common normal at
each source replay action. Normal-valid states199/256; undefined directions
remain null, never replaced by a convenient axis. SAC120k and Flow60k had
mean conditional action covariance traces0.05645 and0.04903 respectively.
This does not support diagnosing Flow as effectively deterministic from these
states. Flow hard-clipping averaged7.18% per path-coordinate-step on this
mixed state pool; this differs from on-policy training clipping and includes
states visited by SAC. The policies are at different training ages: neither
relative efficacy nor a directional entropy benefit follows. Actual per-state
covariances, Gaussian marginal entropy proxies, reference covariances and true
interval risk queries are retained in
[foundation-sampling-v2](engineering/foundation-sampling-v2/diagnostic.json).
This diagnostic collected zero new training transitions and does not populate
the still-pending final mechanism table.

The same analysis implementation now supports labelled interim directional
diagnosis and a strict final-mechanism mode. Final mode requires the two-seed
fixed300k directional/control cohort,256 distinct jointly visited states,
shared exact normals,128 samples/state, reference/endpoint projections,
normal validity counts, same-action normal cosines and legacy cost-weight
factor distributions. It refuses an incomplete foundation-only cohort before
creating output. The complete mechanism analysis has **not** been executed;
the formal mechanism table remains empty. Server-resident replay may be
streamed between servers for this analysis; it is not copied onto this host.

Budget bookkeeping retains ability200k, flow engineering smoke/resume2600,
oracle engineering smoke/resume2208 and v1 obstacle conservative upper bound140k.
A full matched eight-run v2 matrix would bring the projected first-stage total
to2,744,808 transitions, below3M.
This is a ceiling projection, not a command to launch uninformative runs.

Cost-Q/dual, learned-margin and detached-geometry components are implemented.
Flow obstacle learning now has a positive first held-out result. Longer-run
stability, Oracle approximation limits, directional-method evidence, trained
controls, formal runs, payload and robustness remain open. Mechanism controls
are implemented and algebra-checked, but have not been trained or assigned an
arbitrary Iso-Low variance. The full
requested objective is not complete. A100 project storage has about61GiB free
at the current check. Large checkpoint/replay files remain there; compact native
JSON, progress and videos are copied through this host to the new GitHub branch.

See [task definition](TASK_DEFINITION.md), [method changes](METHOD_CHANGES.md)
and configs/star/arm_safe_entropy/development_v2.json for current settings.
All primary evaluations will use fixed final checkpoints, common layouts and
per-training-seed statistics. An unsuccessful experiment will be reported as
such, not replaced by a claim that finishing a budget proves effectiveness.
