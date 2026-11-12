# Paper CRAX / Payload / Piper v1 protocol

Status: active implementation, not experimental success. Independent from immutable
`fc5716e6dc38bc2eefd1b61cc3cb52090b8f07ca` on `codex/paper-crax-payload-v1`.
The complete user task is preserved in `MASTER_TASK.md`. Historical results and
protected user directories remain unchanged. No old failed training is resumed.

## Registered design and accounting

Authoritative settings/seeds/caps: `configs/paper_crax_payload/master_v1.json`.
Development train seeds41027/41028; formal train seeds51027/51028/51029;
validation61027–61031; independent test71027–71031. Never use test metrics to select
or silently replace seeds. Fix all methods' final common training budget before
formal launch: preferred5M PointGoal/0.6M Payload, optionally all methods10M/1M if
development learning trends justify it. Preserve every fixed endpoint/failed run.

PointGoal development includes native reference, engineering sampling, failed
attempts and all candidates under24M; at most six major configurations. Start
native SAC-Lag1M and500k Flow candidates, non-dominated two to2M, best shared
Iso/EA to3–5M, then second development seed. A useful candidate may proceed despite
a weaker second seed. Main comparison is SAC-Lag/Iso/EA ×three registered seeds;
CVPO is optional and cannot block them. PointGoal parameter/mechanism allowance6M.
Total PointGoal caps90M (preferred) /150M (common extension), not a demand to spend.

Payload starts three methods ×two development seeds ×300k=1.8M, targeted fixes
up to total3M development; total including confirmation at most12M. Fresh training
by default; any transfer separately charged and given fair initialization access.
Arm mechanism/parameter/repair training at most3M; Piper simulation at most3M.
Simulation evaluation/oracle queries recorded separately, never relabelled as
training. All genuine physical robot interaction is unauthorized and stays zero.

## Semantics before throughput

CRAX pinned63504b540705be98024abd675e646ff5e046c6c2. Native
`safe_goal_point`, difficulty1, state observations, no rendering; preserve native
time step/frame skip/1000-step horizon, goal refresh, all hazard and cost sources.
Inspect and serialize final effective configuration after difficulty application.
Native difficulty1 has two refreshable goals and12 noncollidable cylindrical
hazards plus collidable outer walls; do not substitute the base defaults.

Expose `cumulative` vs existing `episodic_binary` modes. CRAX cost-Q is an
unbounded discounted cumulative value, not a sigmoid probability. Raw episodic
budget25, raw cost scale1; reward scale initially1 shared with reference. Maintain
per-env complete-episode sums. A sampling block is not an episode. Save final
observations before auto-reset; distinguish truncation from termination and never
bootstrap from a reset observation. Action dimension, covariance trace and energy
target `d_a*0.25/variance` are runtime dimensions, not hardcoded3/18.75.

Only one short profile of Balanced(E256/B1024/G4), Fast(512/1024/4), and
Learn-heavy(128/512/8), all F1. Record update/transition G/(E*F), replay exposure
G*B/(E*F), compile, steady full-training SPS, optimizer updates/s, memory and
evaluation/save overhead. Select on speed plus actual task/safety learning, not
simulator-only FPS. Smaller proportional sizes permitted only for actual memory
limits. Shared JAX/Torch GPU allocation disables JAX preallocation before import;
do not upgrade the existing Arm environment. Margin ramp uses actual margin
optimizer updates1000→5000. Sampling and energy use identical current covariance.

## Fair selection and reporting

Each method gets at most five candidate checkpoints per seed, at registered
fractions. Common validation: risk requirement first, then task capability,
then capped completion time. No risk-qualified candidate => labelled Pareto
tradeoff, not success or a stationary low-cost winner. Candidate count and full
selection history retained. Test selected policies once independently,100episodes
per condition; freeze actor/margin/normalizer/dual throughout evaluation.

Report native return, raw episode cost, goals/episode, >=1goal proportion,
budget-compliant proportion, training cumulative cost/cost-per1000transitions,
and native return among budget-compliant episodes (with denominator). Negative
returns use absolute/dynamic-range gaps, never misleading ratios. Payload reports
safe success, violation and timeout plus capped/conditional time. Train seeds,
not evaluation episodes, are statistical units. Effort targets are not gates
requiring fabricated wins or deletion of weak seeds.

## Other tracks

Payload is genuine pre-held lightweight-object transport, fixed attitude,
XYZ end-effector deltas, no learned grasp. Control-interval safety includes
arm AND payload against obstacles; allowed grasp contacts alone are exempt.
Payload pose/size observations are shared across methods.

Frozen generalization: geometry, one execution-error family (action-scale error),
and alternative-route obstruction with another feasible path. Register fresh
conditions/layouts before testing; development30–50episodes then selected100.
Retain historical Reach evidence without rerunning old long training or fitting
a constant to the already seen8cm case. Any geometric-response/randomization
repair is independent and shared fairly with untouched unseen ranges.

Mechanisms concentrate on an Arm task: M1Iso-High/Iso-Low/Normal-Shrink/
Directional-Full with matching constraints stated; M2Local/Direct/Pullback/Random
on128states×64draws spanning safe/interior/boundary, frozen same actor/spectrum;
M3successful-route/coverage and native/common-sampler cross; M4linearflow variance,
misalignment, fixed-trace logdet and GaussianKL. Oracle evaluates only, never
selects a learned action. rho0.1/.25/.5/1 with at most two new trained parameter
points; K2/4/8 initially frozen-sampler sensitivity, not retraining claims.

Piper uses pinned official assets and actual model/gripper when identified;
otherwise explicit pending configuration while other tracks continue. Dryrun,
units/frames/period, timeouts, replay, versioned actor/margin/normalizer/scaling
and original/executed-action records are required. Panda weights cannot be
renamed Piper. No physical CAN connection or motion is enabled here. Real360
evaluation episodes and optional50episode fine-tuning branches are operator-
enabled future stages, not simulated or fabricated completed real results.

## Resources and delivery

A100 GPU0 nativeSAC/reference+interface, GPU1 Flow/geometry;4080 Payload then
mechanisms/Piper. Share GPUs without stopping foreign processes. No5080. Finite
simple slots fill as jobs finish; no nested watchdog architecture. Preserve large
replay/checkpoints on4080 4T and necessary A100 originals; publish compact evidence,
code/configuration/curves only. Required result files may initially be header-only
and explicitly pending; never put invented zero metrics into them. Only necessary
semantic, geometry, conversion and one genuine full-state recovery check, not a
repeat audit of historical18000episodes or unrelated refactoring.
