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

## Shared learner and execution still pending

The independent configuration declares shared Actor LR1e-4,Huber beta1,
critic LR3e-4,unchanged update counts/target settings,1000 critic-only warmup
updates after2000 random transitions,dual eligibility `actor_updates>0`,
margin_online/margin_geometry EMA .005 and matching sampling/energy ramp.
The shared learner integration,full-state resume/RNG test,isolated evaluation
runner and bounded dynamic queue are not implemented yet. Configuration
alone is NOT evidence those changes are active.

Initial development is six fresh300k runs (1.8M). Only after these results
will up to four extra100k candidates be considered,with one main change each;
then paired Iso/selected-geometry development must cover600k-to1M. Total
new development training is capped at6M,with physical diagnostic queries
separately recorded. A new9M confirmation is conditional,not initially queued.
No obsolete job or5080 process is resumed.
