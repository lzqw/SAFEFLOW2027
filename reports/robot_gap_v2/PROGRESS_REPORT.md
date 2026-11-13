# Robot-Gap v2 — execution progress

This file preserves the execution history. Current conclusions are in
[FINAL_DECISION_REPORT.md](FINAL_DECISION_REPORT.md); the whole goal remains
incomplete because4080 repair access and deployment-PC timing are outstanding.
All launched A100 training/evaluations are complete. All new work is
simulation/offline only. Initial training implementation:139c4e15; finite Reach
adaptation/final mechanism implementation:acb481dd; baseline:5a403e04.
Latest completed outcomes and the continuation decision are in the final
section, datedNovember14+08. Earlier timestamped execution snapshots are retained
as history, not descriptions of currently running processes.

## Current decision answers — not a completed-round claim

| Required question | Evidence-supported answer now | Remaining evidence |
|---|---|---|
| Why did Piper Payload fail to learn? | Original Iso is not immobile:0.424m path but only0.0684m gap reduction; detour and EA each succeed10/10 on the same diagnostic layouts. This identifies ineffective learned routing/progress, not a globally unreachable task. | No isolated causal proof that reward, values or exploration alone caused the failure. |
| Which repair helps? | Common Reach-Actor transfer plus profiled updates yields a useful validation-selected Iso60K candidate:100/100/100% independent success; EA120K96/75/98%. Actual branch training was300K each; poor fixed300K endpoints remain separately preserved. | No initialization-only causal attribution, cross-training-seed or physical zero-risk claim. |
| Has Reach geometry improved? | Independent-layout Iso400K success100/72/99%; EA400K100/70/100% for nominal/55mm/scale. Both received100K adaptation. | Geometry violations28/30% exceed the10% release rule.55mm was in training: no unseen-radius claim. |
| Does EA's gain survive equal adaptation? | PointGoal's frozen independent EA policy has35.43goals/21.32cost versus Iso28.29/26.85. Equal Reach adaptation does not show an EA advantage; selected Payload Iso has higher safe success/lower violations, EA faster successful completion. |4080 Panda repair. Three-seed fixed10M PointGoal remains separate from selected-policy tests. No universal EA dominance. |
| Is training genuinely faster? | Profile complete SPS10.63→57.66; actual EA100K transfer34.97SPS including evaluation/save, within20–40SPS engineering target. | Optimization/update counts changed and shared load is uncontrolled; no learning-equivalence or hardware-only speedup claim. |
| What can be taken to deployment? | Actual frozen Reach and selected Payload bundles, EMA margin/normalizer, offline replay and90-episode Reach on-site protocol are delivered. Payload remains offline only. | Not eligible for motion: Reach geometry violations are too high; useful Payload simulation does not calibrate hardware. Deployment-PC timing/frames/TCP/limits and现场确认 remain outstanding. |

Source tables and actual receipts: [Piper curves/efficiency](piper_results/README.md),
[independent Reach/bundles](reach_adapted_release/README.md),
[PointGoal paper material](paper/README.md). This table answers the requested
questions without replacing development results with independent tests or
declaring incomplete work finished. The new final-mechanism postprocessor
accepts only the completed64×32/four-sampler receipt and reports both native
cost magnitude and whether per-state event rates actually differ across
samplers. A mixed pool alone is not evidence of an event-safety effect.

## Actual old Piper finals

All six300K trials are complete. At30 fixed development episodes per condition:

| Task/method | Nominal safe success | New geometry | Execution scale0.85 |
|---|---:|---:|---:|
| Payload/Iso | 0% | 0% | 0% |
| Payload/Iso-Low | 0% | 0% | 0% |
| Payload/EA | 90% | 53.3% | 63.3% |
| Reach/Iso | 90% | 30% | 83.3% |
| Reach/Iso-Low | 0% | 0% | 0% |
| Reach/EA | 100% | 33.3% | 83.3% |

These are individual training-seed development results, not cross-training-seed
averages or independent-test estimates. The old snapshot describing EA still
at187K is superseded. No old trial is extended or repeated.

## Why Payload stalls

On the predeclared ten development layouts, detour and EA both succeed10/10;
zero,Iso,Iso-Low succeed0/10. Iso moves0.424m per episode on average but closes
only0.0684m of the0.240m initial goal gap. Its final goal distance averages0.172m,
compared with EA0.0278m. Iso-Low shows a similar0.170m residual.

This directly rules out literal immobility and demonstrates feasible native
payload transport on these layouts. It points to ineffective learned progress/
routing, not an established global simulator/IK impossibility. The precise
contribution of rewards, values, exploration and task initialization remains
unresolved; actor-only common-parent transfer is the paired intervention.
Raw per-step action,path,reward components,control tracking and interval risk
are retained on A100; compact episode evidence is committed alongside this report.

## Repair and speed experiment

Explicit task transfer copies the same preselected Reach/Iso300K Actor into Iso
and EA, including the compatible identity34D convention. Target Actor starts
from those same weights. Both rebuild reward/cost critics, online/EMA margin,
replay, optimizers,alpha,dual and counters; new-task random warm-up and critic
preheat count against their100K budgets. The existing strict resume checks stay.

Three exact10K speed profiles use(E,B,G)=(2,256,2),(8,512,4),(8,1024,2).
The latter two use independent-process environments, batched per-head VJP and
packed frozen target samples. All retain20 physics substeps and every distance
query. Their replay sample processing/transition is256, while optimizer-update
counts differ and must be reported. Full geometry is forced only in engineering
profiles; normal transfer uses the native transition ramp. Completed profile
timings below are throughput evidence, not long-budget learning equivalence.
The shared A100 workload is uncontrolled and must be disclosed in timing results.

12 focused tests pass on A100, including real official-Piper serial/process
parity, transfer isolation and preserved second reward-target Polyak ordering.
The original local Python test environment lacks panda_gym; no broad package
installation or repeated validation campaign was undertaken.

## Historical initial budget, resources and outstanding decisions

Receipt refresh confirms1.80M completed old Piper candidate transitions and
90,001,920 completed old formal PointGoal transitions. Neither is newly charged
to this round. The initial reservation was230K for profiles and paired transfer;
the diagnosed Reach followup adds200K, for430K reserved and1.77M unreserved.
Actual observed prefixes are in TASK_STATUS.json. The cap is2.20M, not a target.

The A100 controller starts the paired transfer after profiles, without waiting
for another turn. Healthy foreign jobs remain untouched.4080's confirmed address
172.21.221.238 still returns no route to host. No5080 is used. New artifacts go
to `/storage_0/SAFEFLOW2027_lzqw_uid1007/robot_gap_v2`, not/home.

Pending at that initial snapshot: complete transfer outcomes and paired decision; queued Reach
adaptation;4080 Payload repair; queued final-selected PointGoal64×32 mechanism;
actual deployment-machine latency. Original Reach selection and offline release
are complete; see the final section and reach_release/README.md.
Figures, full training efficiency and frozen risk diagnosis are complete below.
The executable offline policy replay cannot open physical transport; TCP,
frames,period,speed and workspace require现场确认. No real-world outcome or
calibrated safety is claimed.

## A100 storage completion (November13)

The authorized home-to-data migration completed at20:29:27+08:00. Five
directories (PAPER runs/artifacts, A100 runs/recovery_backups, ARM runs) were
checksum-compared before their original locations became compatibility
symlinks. Their logical sizes total73,597,662,059bytes (68.54GiB); hardlinks,
sparse allocation and filesystem rounding make this different from physical
space released. `/home` available space rose from approximately79GiB to145GiB;
`/storage_0` has approximately75GiB available after migration. No unique result
was discarded. Source/environment directories and other users' jobs were not
relocated. Mapping and checksum evidence remain in the operation directory.

New experiments and future caches/temp files use
`/storage_0/SAFEFLOW2027_lzqw_uid1007`; launch checks retain40GiB headroom.
An existing path that resolves through a migrated symlink already writes to
the data disk. Do not recreate those paths as physical home directories.

The next frozen CPU diagnostic fixes ten development layouts and state
times0,10,20,30 before measuring anything. It reports missing states after early
termination without replacement, original-margin value errors/false-safe counts,
physical finite-difference response along the learned outward direction, and
a paired frozen-Actor new-radius rollout with the diagnostic radius prior.
These labels are actual control-interval minimum distances, not endpoint
proxies or a safety certificate. It consumes no RL transitions and selects no
action using the physical counterfactual labels.

## Frozen geometry diagnosis and the bounded Reach decision

The fixed pool retained29 Reach and38 Payload physical states; early-terminal
missing time indices were reported without replacement. At55mm obstacle radius:

| Frozen EA margin | Original MAE | Radius-prior MAE | Original false-safe labels | Prior false-safe labels |
|---|---:|---:|---:|---:|
| Reach |10.321mm|0.178mm|15/116|0/116|
| Payload |10.447mm|0.209mm|10/152|1/152|

These denominators are fixed state/action probes, not independent episodes.
Outward physical finite-difference risk responses were positive in99.1% of
Reach and82.9% of Payload probes; most of the remainder were near-zero.
The original nominal margin MAE was only0.179/0.209mm. This supports a radius
value-extrapolation error on this pool rather than broadly reversed directions.
It is not a global model-calibration claim.

Actual paired frozen new-radius rollouts stayed4/10 safe successes and6/10
violations for BOTH tasks with or without the prior. Correct values alone did
not improve these frozen behaviors. There is no repeated constant tuning or
test-pool reselection. One Reach adaptation pair is therefore queued: original
Iso/EA300K parents, each100K extra transitions, shared45/50/55mm reset radius
family. Native reward/noise/safety/layout definitions remain fixed; evaluations
remain nominal/new_geometry/execution_scale_085. Replay,critics,EMA margin,
optimizers,dual and learning counters continue; environment episodes start
fresh under the explicit distribution fork. This adapts two already trained
systems, not common-Actor initialization or a from-scratch sample-efficiency
comparison. Strict normal resume is retained.

## Completed profiles and current execution

| Profile | E/B/G | Complete10K SPS | Learner updates | Actor updates |
|---|---|---:|---:|---:|
| Reference |2/256/2|10.626|8000|3500|
| Balanced |8/512/4|32.456|4000|1500|
| Large |8/1024/2|57.664|2000|500|

Time includes action generation, unchanged native environment/substeps, replay,
learner and checkpoint writes; the separate post-profile10-episode evaluation
is excluded. All profiles havezero short nominal successes. The selected large
profile is5.43× the reference throughput here, not a claim of equivalent
optimization/long-budget performance. Histories include the same2K random
warmup but different numbers of effective preheat/Actor updates. Learner/replay
dominates the reference timing (796s out of941s); with the selected profile,
that component falls to138s out of173s. Foreign GPU load is uncontrolled.
Both initial transfer runs are active on separate A100 GPUs; normal geometry
ramp is restored, and the requested100K budget is retained.

Reservation is430K (initial230K plus Reach200K), observed actual prefixes125K
at13:03:17UTC, and unreserved later allowance1.77M.30minute schedulers wait for
the corresponding own transfer slot and sufficient free memory; they do not
stop another project or spend optional curriculum/continuation budget.

## Paper and offline release progress

The `paper/` directory contains all-three-training-seed fixed10M results,
five-checkpoint learning curves, full training efficiency and the separately
labelled selected-policy independent-test figure/CSV. Fixed10M mean goals/cost:
SAC36.24/41.28, Iso11.87/22.26, EA27.47/21.97. Cross-training-seed variability is
retained; selected EA8M independent results do not replace these endpoints.
No new PointGoal RL training was run.

Full batch1 Flow inference on the A100 server CPU (one thread,20warmup+200
measurements): Iso p50/p951.849/1.890ms; EA9.617/9.764ms. EA executes20
velocity-network forwards,4 margin-ensemble forwards and4 batched VJP traversals
(8 per-head gradients), including tail/input gradients. These are server CPU
measurements, not deployment-machine measurements or SAC timings. The old4M
physical-state pool is only a fixed timing fixture; it is not relabelled as
the final-model mechanism study.

The final64×32 mechanism source is bound to the exact selected Actor/EMA/
normalizer bytes, four sampler interventions and16704 native one-step queries.
It is queued after Piper followups with≥8GiB shared-GPU free memory; no results
are claimed yet. Reach common validation completed all15 original checkpoint/
control candidates on50episodes×3conditions each (2250 total), followed by actual
bundle export and measured offline replay. The source includes a balanced
90-episode on-site Reach plan, but no physical command implementation.4080
remains unreachable at its reconfirmed address; existing Panda Payload repair
still awaits access.

## Completed original Reach common selection and offline release

These are validation results from one training seed, not independent tests.
All methods received the same five checkpoint opportunities and fixed layouts.
Selected-checkpoint safe successes are:

| Method | Selected steps | Nominal | New geometry | Execution scale0.85 | Selection status |
|---|---:|---:|---:|---:|---|
| Iso |180K|94%|28%|86%|Pareto tradeoff; not risk qualified|
| Iso-Low |120K|0%|0%|0%|Risk-qualified by violations, but stagnant|
| EA |300K|98%|24%|88%|Pareto tradeoff; not risk qualified|

The risk rule requires violation≤10% in every condition. Iso/EA new-geometry
violations are72%/76%; execution-scale violations are14%/12%. Thus neither
qualifies. Iso-Low's zero violation is not useful task success. These results
do not replace the fixed300K development endpoints or establish deployability.
They reinforce the already queued bounded geometry adaptation, without adding
new checkpoint sweeps or repeating validation.

`reach_release/` contains the actual Actor/EMA-margin/normalizer inference
bundles, manifests, all45 condition summary rows, frozen selection, simulated
observation fixture, recorded raw/proposed actions, latency and completion
receipt. All three replays accepted30 proposals withzero deadline misses and
zero physical commands. Acceptance is an offline limit/latency result, not a
calibrated safety result. EA server CPU batch1 p50/p95 is10.235/10.432ms;
Iso1.510/1.569ms and Iso-Low1.500/1.576ms. Real deployment-computer timing and
hardware calibration remain pending; all bundle motion flags remain false.
Large per-episode validation records remain on A100's data disk.

## November14: completed transfer/adaptation and the paired continuation

Both original100K transfer branches and both100K Reach geometry adaptations
finished normally. The full36 condition-row curve table, four completion
receipts, actual timing/update counts and PNG/PDF figure are in `piper_results/`.
No new evaluation was run to create this summary.

| Fixed endpoint | Nominal safe success | New geometry | Scale0.85 | Full segment SPS | New training violations |
|---|---:|---:|---:|---:|---:|
| Payload Transfer-Iso100K |50%|3.33%|56.67%|62.80|226|
| Payload Transfer-EA100K |66.67%|10%|70%|34.97|366|
| Reach Iso300K+100K |100%|80%|100%|66.53|421|
| Reach EA300K+100K |100%|73.33%|100%|38.73|407|

All rows use one training seed41027 and30 fixed development episodes/condition.
They are not independent tests or cross-training-seed estimates. Safety
violations here include crossing the safety distance, not necessarily collision;
zero collisions do not erase the reported violations.

Payload initialization now learns nontrivial task behavior, but not stably:
Iso60K achieved100% nominal,96.67% geometry and100% scale success, then fell to
50%/3.33%/56.67% at100K. EA20/40K hadzero nominal success,60K26.67%,80K6.67%,
100K66.67%. The full curves are preserved. EA has better100K task scores but
more training violations than Iso; this is not a universal safety improvement.
Actor transfer is bundled with the profiled execution/update configuration, so
the gain over old scratch Iso cannot be attributed exclusively to initialization.

Reach adaptation meets the development targets (nominal≥95%, geometry≥70%) for
both methods. Compared with their original300K fixed endpoints, new-geometry
success improves Iso30→80% and EA33.33→73.33%. Iso is slightly higher after
equal additional budgets; no EA superiority is claimed. Geometry violations
remain20%/26.67%, so neither is risk-qualified by the earlier≤10%-each-condition
deployment rule. That rule and common-validation data are distinct from these
30-episode development results.
The original `new_geometry` condition is55mm, included in the adaptation's
45/50/55mm training family. Improvement is adaptation to that geometry family,
not evidence of generalization to an unseen radius. Subsequent independent
tests use new layouts, not a newly held-out radius value.

Both transfer endpoints meet the plan's≥50% nominal learning signal. Continue
exactly those fixed100K full states by200K each to300K, not the best-looking
intermediate snapshot. Native configuration, critics, replay, RNG, episode
partial state, optimizers and dual are retained; no reward/noise/margin/radius
change and no curriculum. Controller PID1263416 and both workers1263426/1263428
were confirmed live at01:19+08. The implementation source matches the previously
tested training-core bytes. Source/config/evidence are checked before launch;
controllers never preempt foreign jobs, retry failures, or expand budgets.

New reservation is830K (profiles30K, Reach200K, Payload600K), of which430K is
fully completed; the01:18:55+08 collector observed431K including partial new
work. Unreserved allowance1.37M is not a spending target. Real Payload100K
segment times were26.54min Iso and47.66min EA, including evaluation/save.
The EA segment achieves the20–40SPS engineering target; learning regression
prevents a claim of optimization equivalence. Shared GPU load is uncontrolled.

Original Reach inference bundles and2250 common-validation episodes are already
published. Adapted400K models are not silently substituted into that frozen
release. Actual hardware latency/calibration remain pending. Final PointGoal
mechanism is waiting for≥8GiB free memory; its live controller has not run native
queries yet.4080 still has no route at its reconfirmed address; Panda repair
is not launched. No5080 or physical robot is used.

Native SAC batch1 CPU timing is now also measured with the exact frozen
selected native policy/normalizer, on the same genuine observation fixture as
Flow: p50/p950.301/0.337ms,20warmup+200measurements. JIT compile0.154s is excluded.
Normalization, stochastic Actor, RNG split and host return are included; no
simulator or training is instantiated. JAX CPU and Torch CPU runtime/threading
are not identical; do not call their ratio a hardware-normalized speedup or
deployment-computer control latency. Raw receipt is in `paper/`.

The CPU preflight exposed a missing native CRAX PYTHONPATH in the waiting
mechanism launcher. The own waiting controller was explicitly checked, stopped
and replaced before any query/output existed. Its previous status is preserved
on the data disk. The fixed launcher supplies the pinned native checkout and
waits for the Payload continuation pair too; no training core or foreign job
was changed and no mechanism experiment was restarted.

## Adapted Reach independent confirmation and the remaining closure

The sole400K endpoints received300 new validation episodes (50 per condition
for each matched method), then both identities were frozen before independent
layouts71027–71126. The original2250-episode selection was not repeated.

| Independent policy,100 episodes/condition | Nominal success/violation |55mm success/violation |Scale0.85 success/violation |
|---|---|---|---|
| Iso400K |100%/0%|72%/28%|99%/0%|
| EA400K |100%/0%|70%/30%|100%/0%|
| Original Iso-Low120K |0%/0%|0%/0%|0%/0%|

Iso-Low is an unchanged stagnant reference, not a matched100K adaptation arm.
All900 actual independent episode records bind the pre-test frozen selection.
Validation was Iso100/76/96%, EA100/72/100%. Single training seed, no claimed
statistical superiority from a2-point difference, and55mm is in the adaptation
training family. These confirm independent-layout task behavior, not unseen-
radius extrapolation or calibrated hardware safety. Geometry violations remain
well above the10% release rule; neither active method is deployment eligible.
Real native bundles, normalization/EMA-margin identities, raw/proposed-action
replays and latency are in reach_adapted_release/. The original release remains
read-only. Evaluation added56,697 native controls,1,133,940 physics substeps and
1,193,040 distance queries,zero RL or physical interaction.

PointGoal trajectory examples are also complete: one predeclared development
layout61027, same physical initial state and exact selected models. Iso gets
36goals/17.76cost, EA37/28.31; EA exceeds25 in this example. The unfavorable
case is retained, not replaced. All2000 real native control records, layout,
model hashes and readable PNG/PDF are in paper/selected_examples/. The old
500-episode independent aggregate remains unchanged and is not replaced by
these two examples. CPU capture took72.54s andzero RL updates.

Payload closure is queued without further training: exactlyfive canonical
checkpoints/method at60/120/180/240/300K, with60K taken from the completed100K
transfer and the rest from its fixed100K→300K continuation.20/40/80/100K
development snapshots do not become extra candidates. The plan was registered
after reviewing the initial100K development learning curves for the continuation
decision, but before new common validation or independent labels. Both methods
share the same candidate opportunities, validation/test layouts and budgets.
The CPU controller1282516 waits in30minute intervals for both full-budget
receipts, then validates each candidate once, freezes both selected identities
and evaluates independent layouts. No optional training/curriculum is enabled.
Original scratch trials remain separate references with different pretraining
and update histories, not a common-initialization sample-efficiency comparison.

## November14,02:19+08 — first completed300K Payload continuation

Iso's exact100K endpoint continuation finished normally at300K. Fixed endpoint
nominal/geometry/scale safe success is100%/10%/66.67%, with0%/90%/33.33%
violations on30 development episodes/condition. Geometry success at120/180/240K
was66.67/43.33/56.67%; all points and the poor300K endpoint are preserved.
This is nominal learning without stable robustness, not a reason to extend
training or replace the endpoint by an intermediate maximum. Its200K segment
took3019.12s at66.24SPS, with456 additional training violations (682 cumulative
Payload violations). The five completed-branch efficiency table now has48
condition/checkpoint rows and common x-axis scales.

EA was confirmed live at239K/300K,38.20SPS prefix. At unchanged prefix speed the
remaining61K takes about27minutes, plus remaining evaluation/save. This is an
estimate for this training branch, not an ETA for independent closure or the
whole project. Current observed new RL is769K, including630K complete and139K
EA continuation prefix; reservation remains830K. The CPU closure still waits
for both complete receipts. GPU0 has5.28GiB free despite0% instantaneous compute
utilization: other projects retain the remaining memory. The final mechanism's
8GiB launch condition is not met, and no foreign process is stopped. Data free
space is approximately74GiB; home remains145GiB.4080 still has no route.

## November14,02:29+08 — final PointGoal mechanism completed on native CPU

Rather than reduce GPU headroom or interrupt another project, the exact native
analysis gained an explicit CPU backend. Six focused pool/identity/output
contract tests passed. The waiting GPU controller1267618 was verified before
any query job log/output existed, stopped, and its status archived. CPU
controller1319063/worker1319072 then completed once in50.73s. Execution source
1322745e, analysis SHA81d9edd7; same64-state selection rule,32 draws, four
samplers, two exact selected Actors and16,704 genuine native controls. Torch
CPU/GPU random generators are not bitwise equivalent; CPU was declared before
sampling, all arms share the same CPU increments, and no label-driven selection
or retry occurred. No training core, physics semantics or GPU workload changed.

The eight-row CSV, figure, original pool/plan/per-state labels/receipt are in
paper/final_mechanism/. Reference→pullback VN/VT is Iso0.3482→0.3274,
EA0.4514→0.4308. Local/direct have still lower ratios; no pullback superiority
is established. Native mean cost changes Iso0.009484→0.009440 and
EA0.009718→0.009733, so there is no consistent native-cost reduction. All
samplers have identical within-state event rates:60 states always zero and4
always positive. A mixed global pool (6.25% positive) does not make this a
discriminative event-safety experiment. The negative/inconclusive event result
is retained without new states/samples. Proxy normals remain one-step endpoint
clearance probes, not arm interval-minimum or cost-Q certificates.

The02:29 collector now observes790K newly trained transitions,630K complete
plus160K of EA's200K continuation (260K cumulative Payload). Payload closure,
4080 repair access and actual deployment-PC latency remain outstanding; this
is not a completed-round claim.

## November14,02:49+08 — all authorized Piper training complete

EA's fixed100K→300K continuation completed normally: nominal/geometry/scale
safe success83.33%/23.33%/63.33%, violations16.67%/76.67%/36.67% (30 fixed
development episodes/condition). Its120K success96.67%/80%/96.67% regressed
to56.67%/3.33%/53.33% at240K; the full curve is published without substituting
an intermediate peak for the fixed endpoint. EA's200K segment took5253.53s
(38.07SPS),481 additional training violations and847 cumulative Payload
violations. Iso has682 cumulative violations and better300K nominal success;
EA has higher but still poor new-geometry success. No consistent safety/task
dominance is established. Original scratch EA300K development90/53.33/63.33%
is retained as a differently pretrained/configured reference, not a matched
common-initialization control; transfer does not improve its geometry endpoint.

All830K newly reserved transitions are now complete. No curriculum, extension,
extra seed or optional matrix is launched. The1.37M unreserved ceiling is not
a spending target. The six-branch60-row development curve and exact complete
segment timing/updates are in piper_results/. CPU closure1282516 was confirmed
sleeping/live at02:49 and will start its canonical-five common validation at
the next registered30minute tick (about03:09+08), then freeze both identities
before independent testing. No validation/test result is claimed yet.4080 is
still unreachable; A100 retains145GiB home and74GiB data free space, and both
shared GPUs show100% instantaneous utilization by the remaining other projects.
