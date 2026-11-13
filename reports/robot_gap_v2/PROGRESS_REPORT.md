# Robot-Gap v2 — execution progress

This is an interim report, not the final2.2M decision report. All new work is
simulation/offline only. Initial training implementation:139c4e15; finite Reach
adaptation/final mechanism implementation:acb481dd; baseline:5a403e04.

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

## Budget, resources and outstanding decisions

Receipt refresh confirms1.80M completed old Piper candidate transitions and
90,001,920 completed old formal PointGoal transitions. Neither is newly charged
to this round. The initial reservation was230K for profiles and paired transfer;
the diagnosed Reach followup adds200K, for430K reserved and1.77M unreserved.
Actual observed prefixes are in TASK_STATUS.json. The cap is2.20M, not a target.

The A100 controller starts the paired transfer after profiles, without waiting
for another turn. Healthy foreign jobs remain untouched.4080's confirmed address
172.21.221.238 still returns no route to host. No5080 is used. New artifacts go
to `/storage_0/SAFEFLOW2027_lzqw_uid1007/robot_gap_v2`, not/home.

Still pending: complete transfer outcomes and paired decision; queued Reach
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
