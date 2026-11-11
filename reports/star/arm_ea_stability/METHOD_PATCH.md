# EA-STAB v2 implementation status

Baseline: `1dc1f2945bb01ace08500147659a73623f30d977`.
Independent branch `codex/arm-ea-stability-v2`; historical runners, loading
and frozen-main validation remain unchanged.

## Implemented geometry, not yet trained

`endpoint_geometry.py` adds a separate `StableFlowActor`, retaining the
original velocity architecture, bounded midpoint scheme,K=4,A0=0,base
variance .04 and inverse-reference path-energy formula. The legacy
`ArmFlowActor` is not edited.

At each step, geometry is queried against the pre-noise proposal
`Zk = Ak + dt*vk`, not incorrectly against the midpoint. Pullback differentiates
the margin at the predicted deterministic endpoint through the injection
clamp and remaining midpoint/clamp operations with respect to independent
`u=Zk.detach().requires_grad_(True)`. The queried generator itself supplies
the tail, so a target-generator call cannot accidentally use the online tail.
No future random noise or physical dynamics predictor is introduced.

Parameters are temporarily frozen and restored; input gradients are enabled
locally even inside `torch.no_grad()`, consistent with the installed-version
[PyTorch2.7 enable_grad contract](https://docs.pytorch.org/docs/2.7/generated/torch.enable_grad.html).
All geometry results are detached. Real clamp derivatives are retained;
zero/invalid gradients fall back to isotropy and are counted, not replaced
with straight-through estimates or artificial normals. Proposal clipping
and final clipping have separate diagnostics.

Modes are `none`, `local`, `endpoint_direct` (diagnostic only) and
`endpoint_pullback`. One beta factor ramps geometry between10k and30k
real transitions. The SAME actual covariance is used by noise and energy;
normal/tangent eigenvalues retain total trace .12. The Actor objective is
not changed by this geometry implementation. There is no second executed
policy, candidate reranking,extra drift/barrier or8cm training augmentation.

Six focused numerical tests pass on local CPU Torch2.4 and the native server
Torch2.7.1+cu118: exact beta0 equivalence to the legacy Iso sampler/energy,
linear tail transpose-Jacobian/current-generator correctness,final-step
direct-gradient degeneration and true clamp fallback,no_grad input/parameter
gradient isolation,actual injection-site/Actor-backprop preservation,and
one-time ramp with shared covariance. No historical test suite was rerun.

## Shared learner and independent native runner implemented

`stability_models.py` now implements shared Actor LR1e-4,Huber beta1,
critic LR3e-4,unchanged update counts/target settings,1000 critic-only warmup
updates after2000 random transitions,dual eligibility `actor_updates>0`,
margin_online/margin_geometry EMA .005 and matching sampling/energy ramp.
It reuses the original critic/Actor network shapes and keeps both reward
updates,target sample count and target rates unchanged. The online margin
optimizer does not own the geometry copy. One EMA is applied only after the
entire learner round; no geometry change occurs between reward updates.
Warmup episodes remain in the runner's global cost/budget accounting but
cannot enter the dual EMA before any Actor learning.

The update path retains detached device-side diagnostics. Only an explicit
periodic `metrics()` call transfers values to the CPU, unlike the old
per-update synchronization. This is a reporting/performance change,not a
new reward or update-budget modification. Signed replay/policy Q and the
alpha used by the Actor are directly exposed rather than inferred from
post-update alpha and Actor loss.

Four additional focused tests cover the common package/dual preheat gate,
geometry frozen through a full round with exactly one EMA at the end,
exact CPU next-update restoration of EMA/optimizers/counters/Python/NumPy/
Torch RNG,and rejection of an undeclared settings change as a native resume.
All10 geometry/learner tests pass on the native A100 environment
Torch2.7.1+cu118 using CPU tensors and0 physical transitions. These synthetic
contracts are NOT evidence of useful trained policies.

Before any scientific training,the Iso control was also given the SAME online
margin capacity,EMA copy,label batch and margin optimizer updates as Local/EA.
Its geometry mode is `none`: it never uses these heads for sampling or energy.
This honors matched model capacity/update information rather than silently
comparing different safety-label training budgets. This is a pre-training
implementation clarification,not an outcome-selected candidate.

`train_stability.py` reuses the native Replay and simulator while keeping old
frozen-main validation/loading untouched. Every evaluation scopes Python,
NumPy,Torch CPU/CUDA RNG and uses separate physics clients. Development
100k/200k/300k/600k uses30 nominal and20 per stress; fixed1M uses100 per
condition on independent130027-series layouts rather than the development
120027-series. All success/safe-timeout/violation classes and conditional/capped
completion times are retained. Neither stress nor diagnostic data trains a head.

Full native checkpoints include online/EMA parameters,optimizers,replay,
transition/dual counters,all RNGs,partial episode returns/lengths,recent episode
records,and observed q/dq/layout/time/environment-RNG state. A native continuation
reconstructs that observed state without silently starting fresh episodes;
solver warm-start caches are NOT claimed bitwise portable. Protected full
300k/600k/1M snapshots accompany periodic full recovery checkpoints. Policy-only
weights cannot be paired with foreign replay or mislabelled as native resume.

On native Torch2.7.1+cu118,all12 focused contracts passed. One actual CUDA
engineering trial ran3200 steps and resumed to3210:replay count3210,
learner updates1210,Actor updates105,plus preserved partial episodes. A real
two-condition/two-episode evaluation consumed206 physical intervals and left
all CPU/CUDA/Python/NumPy training RNG states unchanged. These3210 training
transitions are included as engineering overhead,not hidden outside the6M cap.
The narrow physics reconstruction contract consumed3 additional diagnostic
intervals. No broad historical suite or failure-drill system was added.

`queue_stability.py` is a finite standard-library parent for exactly six
fresh300k trials; it cannot launch the9M confirmation. Each method spans
A100 and4080,with seeds distributed across hosts. A100 starts at one learner
per admitted card;4080 initially one and permits its second slot after5k
observed productive steps and resource headroom. Queues fill a released local
slot without waiting for a global wave; they never restart a failed run.
Only owned handles are polled frequently; GPU resource checks occur at admission,
and the read-only observation interval is1800s. No foreign process is stopped.

At22:48:20 local,the owned queues and three native handles were directly
verified alive:Iso2027 at19k on A100 GPU1,EA2027 at18k andLocal2027 at20k
on4080. A100 GPU0's initial100% utilization caused a resource wait; the next
admission check remains queued. The snapshot is not a claim that all six
trials have started or any efficacy target has been achieved.

At23:29 a resource-only controller handoff removed a misleading exclusive-idle
assumption:authorized shared-GPU admission now allows up to100% observed A100
utilization,while retaining6GiB free-memory,CPU-load and disk headroom guards.
Only this project's standard-library queue parent was stopped;the native
Iso2027 PID1587947 continued from76k to81k unchanged. New controller1627249
adopted that exact native handle and launched the originally pending EA2028
on GPU0,PID1627251. The original queue state was backed up before replacement.
See [RESOURCE_HANDOFF.json](RESOURCE_HANDOFF.json). This is not a new scientific
candidate or a training restart;all learners still execute immutable3c308e4
code/configuration. No foreign process,priority or clock was modified.

The23:48 read-only check verified both A100 native handles and both4080
handles alive,with two trials still waiting for their predeclared slots.
Whole-card GPU utilization includes other projects and is not attributed
entirely to these small learners. The4080 snapshot had about1.5GiB VRAM free,
so no further process is added there beyond its existing two learners.

Initial development is six fresh300k runs (1.8M). Only after these results
will up to four extra100k candidates be considered,with one main change each;
then paired Iso/selected-geometry development must cover600k-to1M. Total
new development training is capped at6M,with physical diagnostic queries
separately recorded. A new9M confirmation is conditional,not initially queued.
No obsolete job or5080 process is resumed.

## Explicit engineering completion,2026-11-11

Four initial300k trials completed. Two4080 trials ended with a CUDA launch
failure and no emergency full save;the host later rebooted. Driver94ffff3
adds only a declared native recovery path to the same300k endpoint,with a
terminal incident record and a matching existing full learner/replay snapshot.
Old raw output is never overwritten. Learner,geometry,evaluation,physical
task and the development profile are unchanged from3c308e4. These recoveries
are not new candidates or replacements for poor scientific results.

The finite e780fde resource controller completes290k-to300k EA2027 and
230k-to300k Iso2028 on A100 GPU0;Iso2027 independently continues its native
300k full snapshot to1M on A100 GPU1. The reserved recovery increment is80k.
Discarded logged tails (7k transitions,17 violations) remain charged,plus
an explicit4k budget reserve for unlogged tails. Evaluation tables distinguish
native lineage cost from the known-tail-adjusted cost lower bound. Native
reconstruction remains observed-state continuation,not bitwise solver-cache
identity across a reboot/GPU/host change. No automatic recovery loop is added.
