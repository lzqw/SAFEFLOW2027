# Arm safe entropy — frozen main running on A100/4080;5080 excluded by user

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

## Current frozen-main execution — 2026-11-09 21:12 CST

Five native main trainers are directly confirmed alive and past random warmup;
four actual waiting queue processes complete the nine-run frozen cohort. No
formal1M run is complete yet: MAIN_RESULTS.csv has0/9 final rows. All native
configs record fresh initial_steps=0, the same frozen v3 method/task and the
requested100k/300k/600k/1M evaluation milestones,100 episodes each.

| Training seed / host | SAC-Lag | Flow-Iso | LearnedDir |
|---|---|---|---|
| 2030 / A100 | running104k,PID857625 | running77k,PID857626 | running43k,PID857627 |
| 2031 /4080,4T | queued after Iso,PID518589 | running278k,PID518590 | running147k,PID518591 |
| 2032 /4080,4T | queued after new Iso,PID551582 | queued after2031 SAC,PID551580 | queued after2031 Learned,PID551581 |

These are observed progress/queue states, not extrapolated evaluation values.
The2032 jobs are queued fresh1M runs on4080; none has started yet. Cross-seed
dependencies release a resource slot only: they do not change a training seed,
transfer weights or alter the frozen configuration. At most two learners share
the4080; A100 has the two-card assignment recorded in
[main_launch_plan.json](main_launch_plan.json). Other-project loads are not
paused or killed. A read-only1800s watcher now follows only A100/4080 and the
actual native/queue PID identities, including future queued-child handoffs.

**5080 is no longer used for training, per the user's instruction.** Its Iso/
Learned2032 PIDs disappeared at about35k/19k before any fixed evaluation.
Logs contain finite diagnostics but no Python traceback; filtered kernel logs
did not establish OOM/Xid/segfault. The cause remains unknown, not an algorithm
failure claim. The waiting SAC2032 queue PID371633 was identity-checked and
stopped; it must not silently restart there. Original files remain, and both
native recovery checkpoints (30k/10k) were copied to4080's4T archive. No foreign
process was changed. [Engineering evidence](engineering/main5080_interruption_20261109.json)
retains both attempted runs and at least88 logged training violations.

To keep primary formal training counts complete, the replacement2032 group
will start fresh on4080 with identical seeds/parameters, not resume an unknown
lost tail or select a favorable checkpoint. The original attempted35k/19k
segments remain engineering interruptions, with approximately54k–56k actual
transitions consumed in addition to the9M primary scientific cohort. This
overhead and its incomplete cost tail are explicitly separate; no exact9M
total including failed deployment is claimed. No scientific outcome was
available from either interrupted run before this resource repair.

The first available main milestones are still **interim**, not final:
seed2031 Iso/Learned100k both obtain100/100 safe successes and0/100 violations,
with163/125 training violations (Learned23.3% lower at this same seed/budget).
SAC2030 at100k has9/100 safe successes and0/100 violations, mostly safe stalling;
do not compare it to a different seed as a matched method ranking. Main
parameters remain frozen despite these observations; all runs retain1M.

The shared formal-layout preparation succeeded on500/500 fixed condition/seed
pairs with23008 validity transitions and0training/model-evaluation episodes.
The initial30-proposal reset limit failed before any formal training began;
only the evaluation wrapper's same-seed proposal continuation was extended to
300. Exactly five radius8cm cases used a second30-proposal batch; every other
case, including all100 nominal layouts, passed within the original limit.
No safety threshold, level, fixed seed or failed policy episode was removed.
[Complete layout proof](formal_layout_validity/layout_validity.json) and the
[initial stopped preparation](engineering/formal_initial_reset_limit_20261109/)
remain available. Frozen actor/task math is95ab831; initial formal runtime is
498535a and the resource-slot queue extension isf059d4d. The full old negative
v2 robustness data and the positive one-seed v3 screen are both retained below.

## Final relative-layout screen and freeze decision — 2026-11-09 19:54 CST

All three predeclared runs completed exactly300000 new transitions each,
**900000 total**, with native completion receipts and all nine scheduled
30-episode evaluations retained. No restart, lost tail, scientific early stop,
extra method or budget extension occurred in this screen. The watcher found
all three trainer PIDs absent with their normal receipts and exited normally.
Final policies are the fixed300k checkpoints, not the best intermediate result.

| Final300k, development seed2027 | Safe success | Violation/contact-collision episodes | Return | Training violations | Time-to-goal |
|---|---:|---:|---:|---:|---:|
| SAC-Lag | 30/30 | 0/30 /0/30 | 11.4594 | 190 | .8160s |
| Flow-Iso | 30/30 | 0/30 /0/30 | 11.4759 | 279 | .7813s |
| LearnedDir | 30/30 | 0/30 /0/30 | 11.5105 | 188 | .7200s |

All time-to-goal means cover30 successes. Learned retains a **32.6% lower
training violation count than Iso**, with matched final success and observed
episode risk; its return/time are also modestly better on this common set.
Against SAC the count advantage is only2 (1.1%), so do not claim a strong
SAC-relative safety gain. This is one development training seed, not cross-seed
confirmation, statistical significance or proof of zero population risk.
The100k failure and earlier stalls remain in the curves and native JSON.

This resolves the user-requested B screen without another task/parameter
search. **Freeze SafeReach-RelativeLayout-v3 and the unchanged algorithm for
the new-seed formal comparison.** It is not a9M extension of the saturated v2
task. The original v2 zero-shot comparison remains negative, and no v3
Iso-Low-dominance or robustness advantage is claimed without measurement.
The actor, local geometry and task source files match95ab831 exactly; later
source changes are evaluation/queue/recording interfaces, not a new loss,
critic, selector, drift, projection, dual or activation.

The frozen protocol is SAC-Lag/Iso/Learned x2030/2031/2032, fresh initialization,
1M actual training transitions each (9M total), fixed100k/300k/600k/1M
evaluation points and100 episodes per condition. Independent main layouts are
92027..92126. The five conditions keep nominal/radius7cm/radius8cm/balanced
position3cm/position6cm; no stronger conditions are introduced. Native nominal
records are reused, not rerun and selected; four stress conditions use frozen
policies without adaptation. One shared physical-layout proof is required
before those new formal comparisons. At this freeze handoff it has **not yet
run**, and formal training has **not yet started**. Planned dispatch assigns
one full training-seed group per host for within-seed method/hardware parity:
A1002030,4080/4T2031,5080/LZQW2032, respecting shared resources and leaving
foreign jobs untouched. Actual launch evidence will be reported separately.

The full three-run final input corpus is on4080's4T disk under
`analysis_inputs/relative-layout-screen-v3-20261109/`; A100 originals remain.
No checkpoint/replay is uploaded to GitHub. The combined development CSV now
contains56 native evaluations (47 original plus9 v3 screen); formal main rows
remain0/9 and the original40 zero-shot robustness rows are retained.

## Screening checkpoint history — 2026-11-09 18:24–18:25 CST

The scheduled half-hour watcher confirmed both SAC/Iso native PIDs absent with
normal300k completion receipts; Learned's PID remains alive at226k. The compact
capture reached300k/300k/226k (SAC/Iso/Learned). No process was restarted,
configuration changed, seed replaced or budget expanded.

**All three methods now have their predeclared fixed100k evaluation** on
common layouts82027..82056:

| Method at100k | Safe success | Violation/contact-collision episodes | Return | Training violations |
|---|---:|---:|---:|---:|
| SAC-Lag | 19/30 | 0/30 /0/30 | 6.4996 | 108 |
| Flow-Iso | 0/30 | 0/30 /0/30 | -1.9416 | 142 |
| LearnedDir | 12/30 | 1/30 /0/30 | 3.0036 | 56 |

Iso's30 failed episodes all end safely at the100-step true horizon: this is
safe stalling, **not** a safety advantage over SAC. SAC's11 non-successes
also end at that horizon. Mean time-to-goal among SAC's19 successes is2.1074s;
Iso's successful time-to-goal is undefined, not zero. Learned's mean is2.2467s
among12 successes, with17 safe horizon failures and one first-interval
clearance failure on layout82034. This failure has no physical contact and is
retained, not removed as an inconvenient layout. Successful coverage is
explicit, not an all-episode speed claim.

Learned's12 successes versus Iso's0 and56 training violations versus142 are
an **early, one-seed task-acquisition signal** (+40 percentage points success,
60.6% fewer training violations at the same100k budget). But it also has one
evaluation violation versus Iso's0; this is not a clean evaluation-risk Pareto
gain. SAC still has higher100k success19/30 with0/30 violations. The signal
does not establish a final300k gain, cross-seed stability or readiness for9M.
Each run retains its original300k endpoint, with no scientific early stop;
SAC and Iso have reached that endpoint; Learned continues unchanged.

### Two completed screens and matched200k foundation results

| Fixed checkpoint | Safe success | Violation/contact-collision episodes | Return | Training violations | Successful time-to-goal |
|---|---:|---:|---:|---:|---:|
| SAC-Lag200k | 30/30 | 0/30 /0/30 | 11.4412 | 164 | .8773s,30 successes |
| Flow-Iso200k | 30/30 | 0/30 /0/30 | 11.4818 | 225 | .7720s,30 successes |
| LearnedDir200k | 30/30 | 0/30 /0/30 | 11.4587 | 155 | .8240s,30 successes |
| SAC-Lag300k, final | 30/30 | 0/30 /0/30 | 11.4594 | 190 | .8160s,30 successes |
| Flow-Iso300k, final | 30/30 | 0/30 /0/30 | 11.4759 | 279 | .7813s,30 successes |

Iso recovered from its retained100k safe stall to full safe success at200k;
the early failure is not its final outcome. Its training count225 versus
SAC's164 at matched200k is higher, despite slightly higher return/shorter
successful time. Learned's fixed200k now also has30/30 safe successes and0/30
violations/contact collisions. Its155 training violations versus Iso's225
is **31.1% lower at matched evaluation success and risk**; versus SAC's164 it
is5.5% lower. This is a useful one-development-seed, matched-budget safety-
exploration signal on the independently varied layout task. Its return/time
are close to, not uniformly better than, both baselines. The result does not
establish population zero risk, cross-seed stability or the still-pending
fixed300k Learned outcome. No favorable milestone is selected as the final
result; the earlier100k clearance failure remains in the native evidence.
SAC's native receipt records300000 newly collected transitions,298000 learner,
reward and actor updates each, in5793.14s. Iso's records300000 new transitions,
298000 learner/596000 reward/149000 actor updates in9120.29s. Iso's final
training count279 exceeds SAC's190; matched final success does not imply
matched training safety. These are two completed screening runs, not completion
of all0.9M or formal-seed results. Learned's226k current training count165 and
recent success1.0 are not substituted for its pending fixed300k evaluation.

The completed SAC/Iso final policies, full-resume checkpoints and compact native
evidence were copied by successful SSH stream to
`/home/carla/4T/SAFEFLOW2027_ARM/analysis_inputs/relative-layout-screen-v3-20261109/`,
under the original `sac-lag-seed2027-v3` and `flow-iso-seed2027-v3` run names.
The4T mount has1.5TiB free. A100 originals remain intact; no large checkpoint
was written to the host or uploaded to GitHub. The active Learned checkpoint
was not copied mid-write or interrupted.

There are55 native development/ability/control/screen evaluations in the
combined CSV, including eight new v3 checkpoints; the original47 remain
intact. Robustness rows remain40 and formal completed rows0/9. Screening
completion is2/3. The active Learned recent10k-transition speed is17.3/s;
approximate remaining time is1.2h,
conditional on unchanged resource sharing. This is a throughput
projection, not a completion promise. The next scheduled native-PID check is
approximately18:54. The [v3 learning figure](figures/SafeReach-RelativeLayout-v3-learning.png)
retains all three native curves without extrapolated evaluation points.

While waiting, local **protocol-only preparation** adds the requested future
100k/300k/600k/1M main milestones, explicit frozen-task-version matching and
time-to-goal/minimum-distance reductions from the same native episodes. The
old100k/200k/300k screening schedule is unchanged; existing trainers remain on
immutable95ab831. The planned independent main layouts start92027. The main
plan still has frozen=false and null task/method freeze; no formal run,
additional rollout or algorithm module was introduced. This removes an
execution-interface gap without treating current interim results as a freeze.

The preparation-only formal perturbation runner also reuses the exact native
nominal episodes at all four milestones, rather than rerunning nominal and
choosing a stochastic outcome. Its candidate protocol retains the same five
condition magnitudes,100 independent shared layouts per condition,50/50
lateral signs, and one shared layout-feasibility artifact. On v3, lateral means
perpendicular to each independently sampled start-goal line and the offset is
included in shared reset admissibility. A native main config must embed the
identical frozen protocol before any of this can execute. No main model,
formal layout proof or new formal episode has yet been generated.

## Stage decision — zero-shot obstacle evaluation, 2026-11-09

The new user instruction supersedes the earlier pending-stage question: keep
the algorithm unchanged, evaluate frozen300k policies first, and use the
predefined A/B continuation. The complete2000-episode evaluation is now done,
with no training/adaptation updates. All four methods and both training seeds
are retained on nominal, radius7cm/8cm, and balanced lateral offsets3cm/6cm;
each condition has50 common layout seeds72027..72076. Positive and negative
offsets each have25 episodes. These levels were fixed before policy outcomes.

| Method | Seed2027 perturbed safe success / violation | Seed2028 perturbed safe success / violation |
|---|---:|---:|
| SAC-Lag | 74.5% /25.5% | 85.0% /15.0% |
| Flow-Iso | 84.0% /16.0% | 66.5% /33.5% |
| Iso-Low | 57.0% /43.0% | 72.5% /27.5% |
| LearnedDir | 56.0% /44.0% | 67.5% /32.5% |

These are equal-weight results across the four **predeclared perturbed**
conditions,200 episodes per method/training seed; nominal is reported
separately, not silently included. Nominal success is50/50 except Iso-Low2027
at49/50. Learned is1/5 percentage points below Low and28 points below/1 point
above Iso across the two seeds: **no stable robustness advantage**. Every
unsuccessful policy episode in this comparison is a clearance violation,
not safe stalling; contact collisions are still recorded separately.
Both Learned policies fail50/50 at radius8cm, although a safe physical route
has been demonstrated on every layout. SAC2028 succeeds29/50 and Iso2027
18/50 there. The zero-shot failure is not evidence of an impossible task.

[All40 condition rows](ROBUSTNESS_RESULTS.csv), [per-episode native results](robustness/obstacle-v1-20261109/),
and the [fixed-condition figure](robustness/obstacle-v1-20261109/obstacle-robustness.png)
retain safe success, episode violations, successful time-to-goal with its
success coverage, horizon-capped time including failures, return and minimum
clearance. Horizon-capped time is not failed-episode duration: failures receive
the full4s horizon. This avoids reporting early collision as fast arrival.

The unchanged native stochastic actors and Learned margin heads were loaded
from their exact final policies with frozen parameters; no learner update is
called. Geometry, reward, controller, observations and termination are the
same as nominal. The evaluator source is81dc9d5. Fixed25-row action batches
retain inactive rows only for random-stream stability; **no physical step**
is taken after an episode ends. Actual layouts are saved and checked against
the common layout set, with no discarded failed policy episodes.

The original environment's safe-start rejection remains active. At radius8cm,
40/50 reset seeds accept a different shared translation from nominal to avoid
an unsafe initial robot pose; the7cm/position conditions retain the nominal
translations. Every method and both training seeds use exactly the same
accepted layout within each condition. Thus the8cm result is a comparison on
a common **valid stressed distribution**, not a strictly identical-start paired
radius intervention. This limitation is visible in the saved layouts and is
not repaired by removing failures or rerunning conditions until favorable.

Initial direct-ascent constructive paths proved219/250 condition/layouts but
violated on the first interval for31 radius8cm layouts. This happened before
any policy evaluation. The [initial evidence](engineering/robustness_initial_detour_20261109/)
is preserved; only these31 were retried with a retreat-first overhead route,
proving250/250 safe feasible paths. No perturbation level, layout, reward or
policy was changed to produce a favorable comparison. The completed invocation
collected32299 **evaluation**,1240 additional **validity**, and0 **training**
transitions; earlier constructive-check transitions remain in their original
artifact. Runtime19.48s excludes the earlier failed constructive-path invocation.

**Decision B applies.** Do not launch the9M nominal formal matrix. The next
authorized stage is a single independently varied relative-layout task,
SAC-Lag/Iso/Learned, development seed2027,300k each (0.9M total), all fresh
initializations, unchanged algorithm. Its explicit configuration is
`configs/star/arm_safe_entropy/relative_layout_screening_v3.json`.
This is task screening, not a new formal result or a promoted algorithm module.
New formal seeds2030..2032 remain untouched. No stronger perturbations or
additional conditions are added to search for a favorable robustness result.

### Relative-layout screen actually started, not merely queued

Immutable training source95ab831 runs all three fresh models on the previously
idle A100, keeping method/hardware/software comparable for this single-seed
screen. SAC-Lag/Iso share physical GPU0; Learned uses GPU1. No foreign process
was paused or killed. At14:54:34 the read-only watcher directly confirmed:

| New v3 screening run | Native PID | Actual transitions | Fixed target |
|---|---:|---:|---:|
| SAC-Lag seed2027 | 746585 | 6000 | 300000 |
| Flow-Iso seed2027 | 746586 | 4000 | 300000 |
| LearnedDir seed2027 | 746587 | 3000 | 300000 |

All have passed the2000-step random warmup; these are not efficacy results or
substitutes for the fixed100k/200k/300k common30-episode evaluations. Each keeps
2 environments, actual-transition counting, original UTD/batch/actor/reward
update ratios and native10k full checkpoints. Before launch, a single fixed
retreat/overhead route completed all30 new common layouts safely; all30 direct
EE line segments intersect the inflated sphere (geometric indicator, not an
extra straight-line physics evaluation). This bounded check used1210 validity
transitions,0training transitions, no learned policy and no parameter search.
The [actual layout evidence](relative_layout_screening/environment/layout_validity.json)
records independent relative poses and unchanged task units/costs. A read-only
1800-second watcher follows these exact PIDs/receipts, with no automatic restart
or additional runs. Completion of this0.9M screen remains pending.

## Completed development/mechanism result — fixed endpoints, not a superiority claim

All eight original development runs, both bounded100k R1 runs and all four
300k mechanism-control runs are complete. The final common-state analysis is
also complete. There are47 native evaluation records in
[DEVELOPMENT_RESULTS.csv](DEVELOPMENT_RESULTS.csv),30 conditional diagnostic
rows in [MECHANISM_RESULTS.csv](MECHANISM_RESULTS.csv), and **0/9 formal main
results**. Those47 records are the original cohort; the same CSV separately
retains the new v3 screening phase as its native evaluations become available.
No original development/control trainer remains active; the new v3
screen above is a separate authorized stage. No bad seed or
final failure was replaced, and no scientific early stopping was used.

### Final nominal task results, common30 layouts per policy

| Method | Seed | Safe success | Violation episodes | Return | Training violations |
|---|---:|---:|---:|---:|---:|
| SAC-Lag | 2027 | 30/30 | 0/30 | 11.4191 | 219 |
| SAC-Lag | 2028 | 30/30 | 0/30 | 11.4195 | 152 |
| Flow-Iso | 2027 | 30/30 | 0/30 | 11.4569 | 223 |
| Flow-Iso | 2028 | 30/30 | 0/30 | 11.4770 | 274 |
| OracleDir, privileged approximation | 2027 | 30/30 | 0/30 | 11.4664 | 210 |
| OracleDir, privileged approximation | 2028 | 30/30 | 0/30 | 11.4182 | 231 |
| LearnedDir | 2027 | 30/30 | 0/30 | 11.4663 | 223 |
| LearnedDir | 2028 | 30/30 | 0/30 | 11.4718 | 206, retained lineage only |
| Iso-Low | 2027 | 30/30 | 0/30 | 11.4724 | 213 |
| Iso-Low | 2028 | 30/30 | 0/30 | 11.4491 | 171 |
| Random-Aniso | 2027 | 28/30 | 2/30 | 10.0831 | 238 |
| Random-Aniso | 2028 | 30/30 | 0/30 | 11.4823 | 206 |

All final evaluations have0/30 actual contact collisions; the two Random2027
failures are clearance violations on layouts52039/52048, both at17 steps.
Task success equals safe success for these native evaluations. Thirty common
layouts do not establish zero population risk or statistical superiority.
All policies above use fixed final300k checkpoints, not selected milestones.

Iso-Low's aggregate384 training violations versus Iso's497 is22.7% lower,
but seed2028 had **0/30 success at100k**, so this is not a clean engagement-
adjusted safety effect or a claim of learning-curve dominance. Learned's clean
seed2027 count223 ties Iso and exceeds Iso-Low's213; seed2028's retained206
already exceeds Low's171. Its unknown lost tail still prevents an exact pooled
Learned training-cost comparison. The lost segment contained at least one
known violation; it must never be repaired by treating206 as an exact actual
cost. Oracle's441 versus Iso497 (11.3% lower) is modest, privileged and also
affected by seed2028's early safe stall; SAC's aggregate371 is lower still.
Random's favorable100k result did not persist: its final2027 policy fails2/30.

### Final mechanism: a local allocation signal, not the full evidence chain

The [native final diagnostic](diagnostics/final300k-full-control-cohort-v2-20261109/diagnostic.json)
uses all10 final flow policies, both seeds for each of Iso/Oracle/Learned/Low/
Random, on one shared pool of256 distinct recorded states,128 near-boundary
and128 safe-interior states, with128 sampled actions per state. The exact
interval-gradient axis is evaluated at the source replay action and reused
across methods. There are195 valid common normals:113 near,82 interior.
Undefined normals remain null and are excluded only from directional projections.
Covariance is estimated separately within each state, never from pooled actions.

| Near-boundary final actions | Seed | V_N | V_T, per tangent dimension | H_T, 2D Gaussian proxy |
|---|---:|---:|---:|---:|
| LearnedDir | 2027 | .012775 | .012652 | -1.6780 |
| Iso-Low | 2027 | .013051 | .011916 | -1.7875 |
| LearnedDir | 2028 | .012985 | .013696 | -1.7632 |
| Iso-Low | 2028 | .013396 | .011943 | -1.8283 |

Relative to Low, Learned's final normal variance is2.1%/3.1% lower while
tangent variance is6.2%/14.7% higher. H_T is higher by .1095/.0651 nats.
This is a useful **two-seed conditional allocation signal**: unlike global
noise reduction, the directional sampler retains more tangent exploration.
It is not proof of higher safe success: both methods have30/30 final successes,
and Learned has more recorded training violations. Against Iso, Learned's
terminal near-boundary V_T changes+6.8%/-16.1%; tangent preservation relative
to the full-noise baseline is still not consistent across seeds.

The full-pool one-interval violation probabilities are .05472/.04117 for
Learned and .06781/.05066 for Low (19.3%/18.7% lower). These are cloned
single-interval counterfactual probabilities at the shared states, **not
episode violation rates**. The separate CSV `local_risk_increment` is the
actual next-interval margin minus the source replay-action margin, in metres,
not this probability. Near-boundary means are -.002190/-.002494 for Learned
and -.001825/-.002137 for Low. Clipping rates and both marginal Gaussian
entropy proxies are retained for every region in the CSV. H_N+H_T is not
claimed to equal the joint entropy, and no entropy proxy is a safety guarantee.

Control matching is approximate, not hidden. Low's scalar .035883518056867805
was fixed before training from the earlier shared Iso/Learned calibration pool;
on the new10-policy pool Learned's common-normal reference V_N is
.035310/.035335 versus Low's .035884, a1.6%/1.5% difference. Low matches an
average normal-noise scale, not trace or every state. Random retains the same
fixed-trace .12 eigenvalue formula but its independently learned q heads yield
raw near-boundary normal eigenvalue means .030650/.030505, versus Learned's
.032179/.032282. Thus this trained Random control does **not** provide an
exact numerical-spectrum-matched isolation of normal alignment. Its common-
safety-axis reference V_N/V_T are approximately .04/.04, as expected for
random orientations; this must not be confused with its own eigenvalues.

Learned's signed normal cosine on the common near-boundary reference actions
is .833/.834; Oracle's privileged controller approximation is .935/.935 with
108 valid geometry comparisons, not a perfect upper bound. The earlier actual
flow-midpoint Learned cosines .451/.615 remain a counterexample to treating
reference-action alignment as on-path reliability. Earlier pool-dependent
diagnostics, negative offline ReLU/SiLU fitting and negative frozen-anchor
sampling interventions are retained, not superseded by selectively positive
numbers from the final pool.

Focused figures: [Learned versus Low covariance](diagnostics/final300k-full-control-cohort-v2-20261109/conditional-covariance-flow_learned-flow_iso_low.png),
[Oracle versus Learned covariance](diagnostics/final300k-full-control-cohort-v2-20261109/conditional-covariance-flow_oracle-flow_learned.png),
[signed normal agreement](diagnostics/final300k-full-control-cohort-v2-20261109/normal-agreement-flow_oracle-flow_learned.png),
and the [complete nominal learning curves](figures/SafeReach-Obstacle-v2-mechanism-learning.png).
Display subsets are explicit; all10 policies and all30 region rows remain in
the native JSON/CSV. No extra training was used for these figures or analysis.

### Stage boundary and delivery

The first-stage conservative upper bound remains2951808 actual transitions,
including ability, pilots, R1 and the possible interrupted tail. It is closed.
The four separately requested controls added exactly1.2M; they are not an
unreported hyperparameter sweep. Final analysis source is
`645e19ecc61aea1f42831ded9c56e8f6cb32e23d`; it samples frozen policies and
performs no optimizer updates or new training transitions. All server originals
remain intact, with the10-policy final input corpus on4080's4T disk. Only
compact results, diagnostic samples, plots and source are delivered to GitHub;
checkpoint/replay files are not uploaded.

Analysis ran once on A100 GPU0 with32 simulator workers using the existing
`diagnose_foundation --final-mechanism --checkpoint-steps300000` path. The
native JSON records every actual run path and checkpoint identity. Iso2028's
final input was staged from the4T corpus; its optional historical progress.csv
is absent there, while its required full checkpoint, final policy, config,
evaluation and receipt are present. No historical log was reconstructed, no
training was restarted, and the analyzer does not use that progress file.

**The task is learned and local directional allocation is observable, but the
chain “more tangent exploration -> better safe task performance” is not yet
established.** The formal1M x3methods x3held-out-seeds stage remains unfrozen
and unstarted, rather than being silently launched from a local diagnostic.
The subsequent user instruction and completed zero-shot comparison select
bounded relative-layout screening as described at the top; the earlier stage
question is no longer pending. No additional parameter approval is needed
within that authorized stage. Payload
and the two zero-shot perturbation families remain conditional, unimplemented
downstream work, not completed deliverables or hidden scope removal.

The half-hour control watcher exited normally when all four native receipts
were present. At the November9 resource check all three hosts were reachable;
foreign GPU tasks were left untouched. There is no remaining control-training
ETA: that bounded stage is done. Further ETA depends on the stage decision,
not on pretending completed jobs are still running.

## Historical execution log — observations below are dated, not current status

### Evidence and recovery as of 2026-11-07 19:08 CST

All four v2 SAC-Lag/Flow-Iso development runs completed their fixed300k budgets.
Each final evaluation obtained30/30 safe successes and0/30 violations; this is
not evidence of zero population risk. The direction comparison is still open.
Both LearnedDir seeds have also completed300k with30/30 safe successes and0/30
violations/collisions. Both original Oracle runs have now also completed300k.
All eight original v2 development runs and both bounded R1 runs are complete.
This is completion of the development budgets, not completion of the requested
mechanism evidence or proof that the learned directional method is effective.

| Method | Seed | Fixed300k return | Training violations |
|---|---:|---:|---:|
| SAC-Lag | 2027 | 11.4191 | 219 |
| SAC-Lag | 2028 | 11.4195 | 152 |
| Flow-Iso | 2027 | 11.4569 | 223 |
| Flow-Iso | 2028 | 11.4770 | 274 |
| OracleDir, privileged controller approximation | 2027 | 11.4664 | 210 |
| OracleDir, privileged controller approximation | 2028 | 11.4182 | 231 |
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

### Completed development: what is and is not established

Oracle2028's native fixed300k result is30/30 safe successes,0/30
violations/collisions, return11.4182 and231 training violations. Its receipt
records300000 newly collected transitions,298000 learner,596000 reward and
149000 actor updates, elapsed94199.16s. The69574190 cloned controller intervals
are privileged approximate-model queries, not training environment transitions.
At15:30 neither original Oracle PID remained and both completion receipts were
present; the read-only watcher exited normally. Compact evidence now contains
35 native evaluations and0/9 formal rows. No original or R1 trainer remains live.

1. **The simulated task is learned.** All eight original final300k evaluations
   have30/30 safe successes and0/30 violations/collisions on the common layouts.
   Earlier failures remain in the curves;30 episodes do not prove zero risk.
2. **Oracle direction has a modest training-count signal, not a proven entropy
   mechanism.** Counts210/231 versus Iso223/274 are5.8%/15.7% lower, aggregate
   441 versus497 (11.3%). Final success is matched, but evaluation violations
   are tied at zero. Oracle2028 learned later, so fewer training violations may
   partly reflect delayed risky task engagement; counts are not episode rates
   or a clean causal decomposition. SAC's aggregate371 is lower than Oracle's
   441. Geometry is privileged and approximate, not an error-free upper bound.
3. **Learned direction has not reliably retained the gain.** Its clean2027
   training count equals Iso's223, and2028's206 is only a retained-lineage
   counter with an unknown lost tail. No exact pooled cost improvement is
   reported. Final evaluation success and risk do not distinguish it from Iso.
4. **Tangent preservation is not yet established.** On the fixed shared final
   pool, Learned's near-boundary terminal tangent variance changed+5.4%/-11.2%
   against Iso for the two seeds. Reference eigenvalues and fixed trace alone
   do not establish the final-action claim. Iso-Low/Random-Aniso training and
   the full common-state mechanism comparison are still outstanding.
5. **The bounded revision is mixed, and the offline candidates were negative.**
   R1 rho.50 removed2028's100k stall but its Iso-relative training counts improved
   in one seed and worsened in the other. The offline activation comparison
   and frozen-anchor intervention were not promoted into trained policies.

### Next bounded stage, not an automatic formal expansion

The first-stage conservative training upper bound is2951808 transitions,
including both R1 runs and the possible interrupted tail; this stage is now
closed. No additional weight search or extra development trial is added.
The user-requested **Iso-Low and Random-Aniso controls** are the next separate
mechanism stage: two seeds2027/2028 per method,300k per seed,1.2M transitions
total, fixed100k/200k/300k common30-episode evaluations. These are the four
explicitly requested mechanism runs, not four new hyperparameter candidates.

The completed learnable foundation and modest privileged-direction training
signal justify this limited comparison of directional allocation versus global
noise reduction; they do not justify claiming Learned superiority or freezing
formal settings. The predeclared Iso-Low scalar remains0.035883518056867805,
measured from the existing shared final pool, not tuned using control returns.
Random-Aniso keeps a direction fixed within one action generation; its actual
reference spectrum must still be measured on the common states, not assumed
identical across independently trained geometry heads. The plan is evidence-
reviewed. At15:30 no control had yet been launched; the bounded stage started
at15:36 as described below. Main remains
unfrozen,0/9; payload and robustness stages are not silently omitted or declared
complete. They remain downstream of resolving the first-task method evidence.

### Mechanism-control execution, no efficacy result yet

The A100 has102GiB available host RAM,60GiB disk free and ample GPU memory at
the stage handoff. Both original and R1 pairs are terminal with native receipts;
no foreign task was stopped. The existing single-run queue was extended only
to accept the two requested controls, require the evidence-reviewed plan and
enforce the already measured Iso-Low scalar/artifact. It still launches one
300k run per explicit invocation, never a main matrix or a tuning sweep.

Source645e19ecc61aea1f42831ded9c56e8f6cb32e23d was pushed and deployed as an
immutable snapshot before launch. On A100 GPU1, Iso-Low2027 PID564938 and
Random-Aniso2027 PID564939 are actually running, each with2 envs and unchanged
UTD/reward/actor update ratios. Their2028 queues PIDs564933/564929 respectively
wait for the same-method2027 completion receipt, maintaining two learners at
most. Every run has a fixed300k budget and100k/200k/300k common30-episode
evaluations, with full checkpoints every10k. No scientific early stop, automatic
restart, method addition or budget extension is enabled.

At15:38 native logs reached5k/3k transitions, past the2k random warmup, with
finite actor losses.5909/-.1802 and energies18.03/44.33. This is actual SGD,
not just a queued intent or warmup count. Neither run is yet at its scheduled
10k full checkpoint or100k evaluation. No control efficacy or complete mechanism
table is fabricated from these startup rows. Compact manifests/progress are
mirrored server-to-host; checkpoints/replay stay on the server. The read-only
half-hour watcher follows actual queue/native PIDs and all three GPUs, without
restarting a process on a transient observation failure.

At16:00 the watcher directly observed both native trainer PIDs and both waiting
queue PIDs alive. Iso-Low2027 reached42k and Random-Aniso2027 reached24k; these
are training progress, not fixed held-out results. Both have now saved native
full checkpoints. One [CPU restore check](engineering/mechanism_native_cpu_restore_20261107.json)
loaded the actual40k/20k checkpoints without interrupting either job: all learner,
target and optimizer state matched exactly after restore, and replay contents
matched at40000/20000 rows. Update counts were38000/18000 learner,
76000/36000 reward and19000/9000 actor, with the original reference variance
and rho preserved. This supersedes the startup-only checkpoint caveat above.
It performs zero optimizer/environment steps and is not a claim of bitwise
physical-simulator continuation. No environment suite or extra scientific
diagnostic was repeated. Native evaluation count remains35, formal rows0/9;
both100k control evaluations remain pending.

### First control100k evaluation: Iso-Low2027, not a final mechanism result

The16:54 compact capture contains Iso-Low2027's fixed100k evaluation on the
same30 layouts:30/30 safe successes,0/30 violations/collisions, return11.4132,
and140 cumulative training violations. Its fixed variance was measured before
training and was not selected from this result. The seed2027 matched100k
comparison is:

| Method at100k, seed2027 | Safe success | Evaluation violations | Return | Training violations |
|---|---:|---:|---:|---:|
| Flow-Iso | 30/30 | 0/30 | 11.3359 | 137 |
| OracleDir, privileged controller approximation | 30/30 | 0/30 | 11.3601 | 104 |
| LearnedDir | 0/30 | 0/30 | -1.6895 | 74 |
| Iso-Low, measured normal-noise level | 30/30 | 0/30 | 11.4132 | 140 |

Iso-Low has3 more training violations than Iso (2.2%), not a demonstrated
global-noise-reduction safety benefit. Learned's lower count at this milestone
coincided with complete safe stalling, so it is not a successful directional
comparison either. These are one-seed interim results, not final300k or
cross-seed mechanism conclusions. Random-Aniso's100k evaluation remains pending.
The capture records ongoing control progress136k/76k; both targets remain300k,
with both2028 queues still waiting. Native evaluations total36; formal rows0/9.

A separate [nominal-method/control learning figure](figures/SafeReach-Obstacle-v2-mechanism-learning.png)
is generated by `plot_development --include-controls`. It retains per-seed,
per-version curves and the interrupted/resumed Learned2028 segments, with no
smoothing, extrapolation or fabricated evaluation zeros. It compares the
predeclared nominal methods and controls; the original development figure
continues to retain both R1 runs separately. This plotting-only extension
changes no training code, reference setting or evaluation protocol. Queued
methods without native progress are not drawn as measured zero-step outcomes.

### Both controls now have a first checkpoint; direction specificity remains open

Random-Aniso2027's fixed100k evaluation obtained30/30 safe successes,0/30
violations/collisions, return11.3747 and128 training violations. This is9 below
Iso2027's137 (6.6%) at matched success, despite the independent random direction.
Learned2027's same100k policy had0/30 success and74 violations; its low count
cannot be called a task-preserving gain. The random control's positive interim
count difference is retained, not discarded or treated as an aligned-direction
success. This one-seed result means an Iso-relative gap alone is insufficient
evidence for correct normal alignment; actual spectra, learning histories and final outcomes
still need the complete two-seed comparison.

Iso-Low2027 has also reached fixed200k:30/30 safe successes,0/30
violations/collisions, return11.4748 and188 training violations. At the same
budget Iso2027 had195 and Learned2027 had183, all30/30 safe success. Iso-Low's
7-count reduction against Iso (3.6%) and Learned's5-count edge over Iso-Low
are small one-seed interim differences, not a final directional-exploration
advantage. Oracle's171 and SAC's162 at200k remain in the original comparison.
No evaluation-risk advantage is established by the tied zero-violation samples.

The17:30 watcher directly observed both trainer PIDs and both waiting queue
PIDs alive at199k/109k; the subsequent compact capture reached202k/111k and
contains38 native evaluations, formal rows0/9. Both controls retain full300k
budgets. No scientific early stop, seed substitution or parameter adjustment
was introduced in response to these mixed outcomes.

For the eventual required final mechanism analysis, the two completed original
Oracle300k inputs were copied to the existing4080 corpus at
`/home/carla/4T/SAFEFLOW2027_ARM/analysis_inputs/final300k-calibration-20261106/`.
Each added run includes its96MiB full checkpoint, final policy and native
config/evaluation/receipt/progress. The SSH-to-host-to-SSH stream completed
successfully; no large checkpoint was stored on the host or uploaded to GitHub,
and A100 originals were not removed. The4T mount has1.6TiB free. This only
prepares inputs alongside the existing Iso/Learned data; it adds no environment
steps, teacher queries, duplicate environment checks or new scientific probe.

### First final control: no clean-seed Learned advantage over global low noise

Iso-Low2027 completed its fixed300k budget with30/30 safe successes,0/30
violations/collisions, return11.4724 and213 training violations. Its native
receipt records300000 collected transitions,298000 learner,596000 reward and
149000 actor updates in10280.69s. At the same final budget Iso2027 and
Learned2027 each had223 violations, so this global low-noise control has10
fewer (4.5%) than either, with matched final success. Oracle2027 had210 and
SAC2027 had219. These are one-seed outcomes, not statistical significance,
population-risk claims or the completed two-seed entropy-mechanism result.
Learned has not beaten global low noise in this clean final-seed comparison.

Random-Aniso2027's fixed200k evaluation also has30/30 safe successes,0/30
violations/collisions, return11.4681 and199 training violations. Iso2027 had195
and Learned2027 had183 at200k. Thus the random control is now4 above Iso
(2.1%), reversing its earlier100k count advantage; the favorable early node
must not be selected as its final effect. Its complete300k budget is retained.

The waiting Iso-Low2028 queue was found self-blocked at the85% whole-card
utilization guard on GPU1, which already hosted the project's Random-Aniso
learner. Read-only inspection confirmed88GiB available RAM,60GiB disk free,
78GiB free GPU1 memory and94% GPU1 utilization, while GPU0 had about79GiB free
and only12% utilization. Only the still-unstarted queue PID564933 was stopped
after checking its command, waiting state and absent native output. Its JSON
record was preserved as `control-iso-low-2028-v2-gpu1-wait-preserved.json`.
The identical source645e19e queue was reissued on GPU0, retaining the85% guard,
12GiB minimum-free guard, seed, calibration and full300k budget. This is a
placement correction, not a scientific restart, shortened run or extra trial.
No active learner or foreign task was interrupted; no data was deleted.

New native Iso-Low2028 PID584259 is actually running with visible device0,
already at5k with finite post-warmup loss/energy. The19:08 capture is:

| Control | Seed | A100 GPU | Observed state/progress | Fixed target |
|---|---:|---:|---|---:|
| Iso-Low | 2027 | 1 | completed300k | 300k |
| Iso-Low | 2028 | 0 | running5k | 300k |
| Random-Aniso | 2027 | 1 | running206k | 300k |
| Random-Aniso | 2028 | 1 | queued, not started | 300k |

The latter still waits for its own2027 receipt. Concurrency remains at most
two native learners, now using both A100 cards. One of four requested controls
is complete; native evaluations total40 and formal rows remain0/9. The
half-hour watcher reads the current queue/native PIDs and remains live.

**The first original Oracle fixed300k result is now available.** Seed2027
obtained30/30 safe successes,0/30 violations/collisions, return11.4664 and210
training violations. Compared with Iso2027's223 at the same final budget, this
is13 fewer violations (5.8%) at matched success, not an evaluation-risk gain.
It is also9 below SAC2027's219, although a single development seed cannot
establish a robust ranking. The earlier24.1% reduction at100k and12.3% at200k
have narrowed; neither should be reported as the final effect size. Learned2027
still has223 violations and has not inherited this Oracle gain.

The native receipt records300000 newly collected transitions,298000 learner,
596000 reward and149000 actor updates, elapsed92509.26s. Its47045565 cloned
controller intervals are privileged approximate-model-query overhead, not
training environment transitions. The14:30 watcher observed the original2027
PID absent with its completion receipt, while original2028 PID297991 was live
at291k; the subsequent compact capture reached293k. Seven original development
runs and both bounded R1 runs are complete, with34 native evaluations and0/9
formal rows. The remaining original seed continues to its fixed300k endpoint;
no seed is replaced, budget extended, method frozen or additional matrix
launched from this one final Oracle result.

**R1 is now complete for both seeds at its predeclared100k endpoint.** At09:00
both native completion receipts were present and neither R1 PID remained.
Each final evaluation used the same30 common layouts and stochastic behavior
policy. The rho.50 revision did not replace either original rho.25 run:

| Method at100k | Seed2027 safe success / train violations | Seed2028 safe success / train violations | Mean return |
|---|---|---|---:|
| Flow-Iso | 30/30 /137 | 30/30 /107 | 11.3492 |
| Original Oracle rho.25 | 30/30 /104 | 0/30 /66 | 4.5827 |
| R1 Oracle rho.50 | 30/30 /117 | 30/30 /117 | 11.3961 |

All six evaluations have0/30 violations/collisions; original Oracle2028's zero
success is safe stalling, not a safety win. R1 removes that seed's early stall
but does not establish a robust Iso-relative safety advantage. Compared with
Iso, seed2027's training count is14.6% lower, while seed2028's is9.3% higher.
The aggregate234 versus244 is only4.1% lower and is not consistent across
seeds. No significance, population-risk, final300k or Learned-policy gain is
inferred from this short pair. Both R1 policies remain fixed final100k results;
there is no best-checkpoint selection or automatic extension to300k.

R1 seed2027 collected100000 new transitions in33887.09s, with98000 learner,
196000 reward and49000 actor updates. Its23523318 cloned controller intervals
are model-query overhead, not collected training transitions. The first-stage
conservative projected total remains2951808 transitions including the original
eight complete budgets, the two bounded R1 runs, prior engineering pilots and
the possible7000-step interrupted tail. Only the two original Oracle processes
remain live, observed at240k/218k at09:00. The R1 watcher exited normally after
confirming both completion receipts; the original-pair half-hour watcher
continues. Native evaluations now total33, formal rows remain0/9 and no method
is frozen. No further trial, mechanism control or new diagnostic was launched
while awaiting the original fixed300k endpoints.

**New fixed200k evidence now covers both original Oracle seeds.** Oracle2028
recovered from its retained0/30 success at100k to30/30 safe success,0/30
violations/collisions and return11.4563 at200k. Its training counter was183.
The matched200k comparison, without substituting final300k results, is:

| Method at200k | Seed2027 safe success / train violations | Seed2028 safe success / train violations | Mean return |
|---|---|---|---:|
| SAC-Lag | 30/30 /162 | 30/30 /119 | 11.4091 |
| Flow-Iso | 30/30 /195 | 30/30 /244 | 11.4392 |
| OracleDir, privileged controller approximation | 30/30 /171 | 30/30 /183 | 11.4499 |

All six evaluations have0/30 violations/collisions. Oracle's training counts
are12.3% and25.0% below Iso at matched success; their equal-seed aggregate is
354 versus439, a19.4% reduction. This is a useful two-development-seed interim
signal, not a final result, an evaluation-risk advantage, or proof of population
safety. SAC's aggregate281 is lower still. The Oracle uses privileged approximate
controller geometry; its result is not a Learned result or a perfect upper bound.
The original pair continues through300k unchanged, and no formal configuration
is frozen from this milestone.

**R1 seed2028 completed its predeclared100k budget**, with a native receipt and
the actual PID absent at08:00. It obtained30/30 safe successes,0/30
violations/collisions, return11.4108 and117 training violations. At the same
100k node original rho.25 Oracle2028 had0/30 success and66 violations, while
Iso2028 had30/30 success and107 violations. Thus rho.50 resolved this seed's
early safe stall but incurred9.3% more training violations than Iso at matched
success: it is not yet a task-preserving safety improvement. Do not compare
R1's final100k policy to an original300k policy or extend it automatically.
The run collected100000 new transitions in30127.99s, with98000 learner,
196000 reward and49000 actor updates. Its20239912 cloned controller intervals
are privileged model-query overhead, not extra environment training transitions.
The other R1 seed remains live at93k; its final result is still pending.

At08:00 the original Oracle PIDs270049/297991 were live at229k/205k, and R1
PID454684 was live at93k. The compact capture contains32 native evaluations;
formal rows remain0/9. No additional short trial, diagnostic, mechanism control,
formal seed or budget extension was launched. The original100k failures and
negative Learned/geometry interventions remain in the report and native files.

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
At that23:34 startup check no held-out R1 result existed. The matrix below is
updated to15:30, when both original Oracle and both R1 seeds have completed their fixed
final evaluations:

| Live Oracle group | Seed2027 progress | Seed2028 progress | Predeclared target per seed | A100 GPU |
|---|---:|---:|---:|---:|
| Original rho.25 | 300k, completed | 300k, completed | 300k | 0, finished |
| R1 rho.50 | 100k, completed | 100k, completed | 100k | 1, finished |

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
