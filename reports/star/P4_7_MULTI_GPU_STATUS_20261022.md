# P4.7 Multi-GPU Training Status

## Latest continuation update: 2026-10-27 12:25 CST — Goal2030 remains the only active cell

The half-hour observation recorded **19/20 terminal training cells**. The only
remaining SAFEFLOW worker is A100 `point-goal-2030`, now at **8,925,000 /
10,000,000 steps** and still running with MPS verified. The 4080 and 5080 have
no SAFEFLOW workers; their foreign jobs were left untouched. Storage remains
healthy: A100 has about **65 GiB** on the project filesystem and **103 GiB** on
the archive filesystem; 4080 has **214 GiB** plus **2.04 TiB** on its archive
disk; 5080 has **136 GiB** free. No duplicate launch, early stop, or budget
reduction was introduced. The collector remains at
`training_receipts_collected_pending_final_strict_audit`; the final strict
matrix audit must wait until Goal2030 reaches its full budget and its native
receipt is collected.

## Latest continuation update: 2026-10-27 09:24 CST — 19 terminal cells; one A100 cell remains

The current collector reports **19/20 training cells collected as terminal**.
All four 4080 cells and all but one A100 cell are finished. The only active
worker is:

| Host | Cell | Latest logged steps | Fixed full budget | State |
|---|---|---:|---:|---|
|A100|Goal2030|8,875,000|10,000,000|running|

The remaining cell has about1.125M logged steps left at the current observed
rate. This is only a rough runtime indication; no early stop or budget
reduction is being applied. The result collector has no reported training
receipt backlog and remains at `training_receipts_collected_pending_final_strict_audit`.

Current free space is A100 primary17.57GiB and separate archive data103.94GiB;
4080 primary223.58GiB and4T archive2.04TiB. 5080 has zero SAFEFLOW workers.
No new seed, duplicate job, scientific evaluation, or frozen-source change
was introduced. After Goal2030 reaches its full budget, its native receipt
will be collected and the one final strict matrix audit can be considered;
the known provenance/path limitations remain unchanged and are not waived.

## Latest continuation update: 2026-10-26 20:25 CST — 17 terminal cells; three A100 cells continue

The current scheduled monitor shows **17/20 training cells collected as
terminal**. The four 4080 workers have finished; only three A100 workers
remain active. This is a training-receipt count, not a claim that the final
strict scientific audit has passed.

| Host | Cell | Latest logged steps | Fixed full budget | State |
|---|---|---:|---:|---|
|A100|Goal2031|9,985,000|10,000,000|running|
|A100|Button2031|9,910,000|10,000,000|running|
|A100|Goal2030|8,595,000|10,000,000|running|
|4080|all four remaining cells|—|—|training complete|

The collector state at20:23 reports17 completed training runs and no pending
result-collection backlog. A100 primary free space is15.64GiB and its
separate archive disk has116.15GiB free. 4080 has246.25GiB free on the
primary filesystem and about2.17TiB on its4T archive disk. The three
remaining workers retain their full budgets; no scientific early stop,
duplicate launch, or new evaluation was introduced.

5080 has zero SAFEFLOW training workers. Its earlier eight-hour hold has
expired, but no SAFEFLOW task was started there: the seven original Point
cells already have owners elsewhere and are not duplicated. Other work on
5080 remains untouched. The next action is simply to let the three A100
workers finish, then collect their native receipts before the single final
strict audit.

## Latest continuation update: 2026-10-24 21:24 CST — ten continuing cells advance; 5080 hold maintained

The ordinary half-hour monitor reports all ten remaining workers advancing,
with unchanged worker identities: six on A100 and four on4080. The
[unaltered scheduled observation](shapo_p47_four_env_artifacts/operations/20261019/capacity_progress_observation_20261024_2124.json)
is included. No additional remote sampling wave, scientific evaluation,
model load or training restart was performed for this update.

| Host | Cell | Latest logged steps | Fixed full budget | Change since20:54 |
|---|---|---:|---:|---:|
|A100|Goal2027|9,555,000|10,000,000|+15,000|
|A100|Ant2029|4,245,000|5,000,000|+15,000|
|A100|Button2027|9,250,000|10,000,000|+10,000|
|A100|Goal2031|8,690,000|10,000,000|+15,000|
|A100|Button2031|8,630,000|10,000,000|+15,000|
|A100|Goal2030|7,300,000|10,000,000|+15,000|
|4080|Button2029|8,595,000|10,000,000|+45,000|
|4080|Goal2029|6,175,000|10,000,000|+45,000|
|4080|Goal2028|6,725,000|10,000,000|+45,000|
|4080|Button2028|8,205,000|10,000,000|+45,000|

The21:14 ordinary collector still reports ten native training-complete
containers. Those completed budgets plus the ten active CSV counters total
132.37M/150M accepted-lineage steps (88.25% of the step budget, **not** of
wall-clock time). CSV counters are not newly verified durable-checkpoint
counters. Walker's complete five-seed native final results remain below;
there is no new completed environment or final strict scientific audit.
The accepted-lineage total advanced520,000 logged steps since the preserved
[20:24 observation](shapo_p47_four_env_artifacts/operations/20261019/capacity_progress_observation_20261024_2024.json).
The ordinary scientific-evidence archive also reports no pending backlog.

Free space is A100 primary16.34GiB/data132.22GiB and4080
primary13.49GiB/4T2456.91GiB. Primary filesystems remain below the20GiB
warning level, while both separate archive/data filesystems exceed their
100GiB alert threshold. No active canonical directory or other project's
files were moved in this update.

**5080 has zero SAFEFLOW training workers and remains reserved until at
least2026-10-24 23:50 CST.** Other GPU work on this host is left untouched;
it is not evidence of an unpaused SAFEFLOW trainer. No obsolete
seven-worker resume script is replayed and no existing cell is duplicated.
The proposed later Goal2030 migration still awaits permission. Full budgets,
seeds, frozen scientific source and fixed evaluation protocol are unchanged;
known original-parent/path-resolution limitations remain unresolved and no
final scientific gate is waived. The next ordinary capacity frame is due
around21:54.

## Earlier continuation update: 2026-10-24 18:55 CST — Walker five-seed native final results complete

**Walker2028 completed its full5M-step budget and original final evaluation
at18:55:37.** The18:54 monitor still saw its live worker and last4.995M CSV
row; a targeted follow-up found that PID had just exited. Native receipt,
training summary and5M evaluation then confirmed normal completion, clean
frozen source62ca739 and no engineering hard-stop. It was not restarted.
The [three native JSON files](shapo_p47_four_env_artifacts/partial_milestones/native_terminal_receipts/walker2d-2028/manifest.json)
were copied from the A100 primary source to the local host without another
rollout or model load. The final-epoch CSV pause was therefore not treated
as proof of a training failure.

**Walker2d is the first environment with all five native final-budget
results available.** Each training seed2027–2031 ran5M training steps and
used the fixed five evaluation seeds times20 episodes. The [complete
five-seed summary](shapo_p47_four_env_artifacts/partial_milestones/walker2d_native_final_all5/summary.json),
[per-training-seed CSV](shapo_p47_four_env_artifacts/partial_milestones/walker2d_native_final_all5/per_training_seed_metrics.csv)
and byte-preserved evaluations accompany this report. Only local existing
JSONs were summarized; no seed or checkpoint was selected by performance.

| Walker native final metric | Mean ± sample SD across five training seeds |
|---|---:|
|Return|3986.00 ±2108.40|
|Failure rate|29.8% ±32.8 percentage points|
|Mean evaluation cost|0.298 ±0.328|
|Per-seed CVaR-20|0.680 ±0.351|
|Accepted-lineage training cumulative cost, per seed|4905.8 ±631.9|
|High-risk20% normal/tangent variance ratio|0.7839 ±0.0803|
|High-risk20% dangerous outward flow velocity|0.2579 ±0.0705|

The five accepted-lineage training-cost counters sum to24,529; this is not
the total physical recovery cost. The statistical unit above is the training
seed, **not500 independent training replicates**. CVaR values are averaged
across the five per-seed CVaRs, not recomputed from a pooled episode tail.
Locomotion failure rate is not Point-task success rate.

| Walker training seed | Final return | Failure rate | Cost | CVaR-20 | High-risk normal/tangent ratio |
|---|---:|---:|---:|---:|---:|
|2027|5672.55|3%|0.03|0.15|0.7881|
|2028|6236.60|12%|0.12|0.60|0.6947|
|2029|4209.48|37%|0.37|1.00|0.9136|
|2030|1141.98|13%|0.13|0.65|0.7613|
|2031|2669.40|84%|0.84|1.00|0.7617|

**Both the stronger and weaker seeds are retained.** Cross-seed task/failure
variation remains substantial. All five native high-risk variance ratios
are below one (range0.695–0.914), showing normal-direction suppression in
these diagnostics. This differs descriptively from the already terminal
PointButton2030 ratio near0.996, but the tasks are not a matched control
comparison. Neither geometric suppression nor these results alone establish
causal safety improvement, statistical superiority or a final Pareto claim.
No new scientific configuration or weight search is started.

The full matrix now has **ten training-complete cells and ten remaining
cells** (A100 six,4080 four). All ten other workers advanced in the18:54
scheduled frame. Its global counters, updated only for Walker's confirmed
native completion, account for131.075M/150M accepted-lineage steps (87.38%
of the step budget, not a wall-clock estimate). The ordinary collector then
reported **ten terminal containers collected at19:14:49**, including the
new Walker2028 container; the earlier18:44 nine-container snapshot preceded
its native completion. No second training or evaluation was started.

A metadata-only inventory found Walker2028's evaluation JSON, counterfactual
JSON, trajectory CSV and policy checkpoint at each of0.5M/2M/5M, in both the
primary and collected containers. The initial verified migration manifests
also list all four artifact types at all three milestones for the other four
Walker seeds. No historical restoration or filename alias was needed. This
is file-presence evidence, not a new full-model load, provenance audit or
final scientific-gate pass.

At18:54, free space was A100 primary16.69GiB/data133.07GiB and4080 primary
12.99GiB/4T2462.82GiB. No active canonical directory or other project was
moved.5080 still has zero SAFEFLOW training workers and remains reserved
until at least23:50; the proposed later Goal2030 migration still awaits
permission. Full training budgets, fixed evaluation protocol and late-resource
accounting are unchanged. Original-parent/multi-hop path limitations and all
final scientific-certification requirements remain unresolved/unwaived.
This is a complete **one-environment native result**, not completion of the
full four-environment objective or of the strict final matrix audit.

## Earlier continuation update: 2026-10-24 17:54 CST — all eleven continuing cells advance

The ordinary17:54 monitoring frame confirms **nine training-complete cells
and eleven continuing cells with positive logged deltas**, with the same
worker/logger identities. A100 still runs seven and4080 four;5080 has zero
SAFEFLOW training workers and remains reserved until at least23:50. There
was no extra remote sampling wave, policy evaluation or model-verification
load. No seed, training budget, frozen source or scientific setting changed.

| Host | Cell | Latest logged steps | Fixed full budget |
|---|---|---:|---:|
|A100|Walker2028|4,980,000|5,000,000|
|A100|Ant2029|4,155,000|5,000,000|
|A100|Goal2027|9,455,000|10,000,000|
|A100|Button2027|9,155,000|10,000,000|
|A100|Goal2031|8,590,000|10,000,000|
|A100|Button2031|8,530,000|10,000,000|
|A100|Goal2030|7,200,000|10,000,000|
|4080|Button2029|8,290,000|10,000,000|
|4080|Goal2029|5,860,000|10,000,000|
|4080|Goal2028|6,415,000|10,000,000|
|4080|Button2028|7,895,000|10,000,000|

These are CSV counters, not a claim that every listed step is already a
durable full checkpoint. The accepted-lineage counters plus nine completed
budgets now total130.525M/150M steps (**87.02% of the step budget, not time**).
Walker2028 has20,000 training steps remaining; the recent half-hour rate
suggests roughly40 more minutes of training, followed by its original native
evaluation. This is an estimate, not a completion receipt or a new stop rule.

The collector reported nine terminal containers at17:44:43. Existing local
backup metadata also acknowledged replication of the latest observed full
checkpoint exports for all eleven active cells. Goal2028's17:39 backup
observation specifically points to its new16:35 target logger's full
checkpoint, written at17:09, rather than only its older resume input. That
is an export/replication acknowledgment, not a fresh serialized-counter or
cryptographic full-model audit; no CSV counter is relabelled as durable.

Current free space is A100 primary **16.28GiB**, A100 data **133.07GiB**,
4080 primary **13.25GiB**, and4080-4T **2463.58GiB**. Shared total GPU use
was A10035%/46% and408066%; GPU-memory headroom is not an authorization to
add new scientific cells. Primary-space monitoring and the existing archive
and backup processes remain active; no active canonical path or other
project was moved. The [scheduled capacity snapshot](shapo_p47_four_env_artifacts/operations/20261019/gpu_capacity_state.json)
accompanies this status, and the exact [frozen training source](https://github.com/lzqw/SAFEFLOW2027/tree/62ca7399357c10645a955dcbead7ed2153a82c69)
is reachable in the already uploaded branch history.

Permission has been requested before moving long-tail Goal2030 to5080
after the reservation; no approval is inferred from automatic goal
continuations and no migration is scheduled or performed. The complete
training budgets, missing-original-parent/path disclosures, late-resource
accounting and all final scientific gates remain unchanged. The full
objective is not yet complete.

## Earlier continuation update: 2026-10-24 16:54 CST — all remaining cells advancing; nine terminal results collected

**Goal2028's continuation is now epoch-confirmed.** The same PID1902362
advanced from its6.3M full input to **6,325,000 logged steps**, with the
observed epoch at approximately24.72 FPS. The scheduled16:54 frame verified
CUDA/MPS membership, and the controller confirmed the same target identity
and no source duplicate. This is the first scheduled observation of a new
epoch, not necessarily its first logged epoch or a6.325M durable checkpoint.
The [updated continuation receipt](p4_7_training_receipts_20261022/goal2028_from4t_to4080_20261024.json)
retains the exact full input and copied scientific-file scope.

The original matrix now has **nine training-complete cells and eleven
confirmed advancing cells**, with no stopped/unassigned cell. A100 runs seven
and4080 four. All ten older workers also had positive logged deltas during
this half-hour. Their accepted-lineage counters plus the nine terminal budgets
sum to129.975M/150M steps (86.65%); this is a step-budget fraction, **not** a
wall-clock completion estimate or a final-audit pass.

| Host | Cell | Latest logged steps | Fixed full budget |
|---|---|---:|---:|
|A100|Walker2028|4,950,000|5,000,000|
|A100|Ant2029|4,130,000|5,000,000|
|A100|Goal2027|9,425,000|10,000,000|
|A100|Button2027|9,125,000|10,000,000|
|A100|Goal2031|8,565,000|10,000,000|
|A100|Button2031|8,500,000|10,000,000|
|A100|Goal2030|7,170,000|10,000,000|
|4080|Button2029|8,205,000|10,000,000|
|4080|Goal2029|5,770,000|10,000,000|
|4080|Goal2028|6,325,000|10,000,000|
|4080|Button2028|7,810,000|10,000,000|

At16:44:39 the ordinary collector reported **nine terminal containers
collected on A100's separate data disk**, including Button2030. This is a
successful collection report, not a claim that all final scientific gates
have passed. Its native terminal files were already preserved on the host
and GitHub; no extra scientific evaluation was needed.

The **late-resource accounting subset** now passes with20 records,
**1,420,000 known extra physical steps /4,905 cost units**, including
Goal2028's25,000-step /493-cost discarded source tail exactly once. These
numbers refer to this late-resource subset: earlier interruption/recovery
records are additionally accounted by the final publisher, and unknown
unlogged work remains unknown, not zero. No unconsumed incident cell remains.
Accepted-lineage costs are kept separate from physical recovery work.

The existing half-hour read-only monitor was extended to A100's separate
archive disk. Its [deployment and natural-cycle proof](p4_7_training_receipts_20261022/a100_data_disk_monitor_deployment_20261024.json)
records that only the idle owned monitor service was restarted, the prior
state/next due time were preserved and no extra SSH sampling wave or trainer
signal was introduced. The first regular16:54 frame confirms the separate
data volume and its100GiB alert threshold; the primary alert remains20GiB.

Current free space is A100 primary **16.54GiB**, A100 data **132.85GiB**,
4080 primary **13.19GiB**, and4080-4T **2466.07GiB**. The disclosed12GiB
post-copy reserve applied only to the last4080 admission; no further bulk
admission is performed. Primary space remains under observation. No active
canonical training directory, other project, budget, method or evaluation
setting is changed. Original-parent/path limitations and all final scientific
certification gates remain unresolved/unwaived.

**5080 still has zero SAFEFLOW training workers and remains reserved until
at least23:50.** All seven old Point cells have confirmed owners elsewhere;
the obsolete seven-worker resume must not be replayed or used to duplicate
them. No new seed, extra rollout, scientific early stop or model-verification
load was introduced by this continuation.

## Earlier continuation update: 2026-10-24 16:35 CST — last stopped cell launched on4080;5080 stays free

**Goal2028, the last stopped fixed-budget cell, has launched on4080** in the
slot released by completed Button2030. It resumes the same seed2028 from its
**6,300,000-step full input**, continuing to the original10M budget. All109
recorded scientific files were restored from the pinned4T raw capsule into a
previously absent canonical target, with their hashes checked before launch.
This is the recorded suffix scope, not a complete filesystem clone. The5080
original and immutable4T inputs remain intact; no5080 GPU job or source signal
was used. The [continuation receipt](p4_7_training_receipts_20261022/goal2028_from4t_to4080_20261024.json)
records its exact input, restored scope and resource checks.

PID1902362 was dispatched at16:35:07 beside the three current4080 clients.
**Its first new epoch is pending** the next ordinary16:54 monitoring frame.
The controller waits on local scheduled JSON and never auto-relaunches if an
observation is delayed. Process launch is not substituted for progress proof.
The original20 cells are now accounted for as **nine training-complete and
eleven dispatched continuing cells, with none left unassigned**. The new
Goal2028 continuation is explicitly not yet epoch-confirmed; A100 has seven
workers and4080 now has four dispatched workers.

For this replacement slot, the **engineering post-copy primary-disk admission
reserve was explicitly adjusted from16GiB to12GiB**. This is not a training
budget, scientific parameter, evaluation threshold or final-validator change.
The recorded capsule is1.932GiB; primary free space was about15.07GiB before
copying and **13.141GiB after copying**. The retained buffer exceeds the whole
capsule plus four simultaneous243MB full-checkpoint temporary files. Current
raw-evidence staging and large archives already use4T, no active canonical
path is moved or symlinked, the old20GiB storage-alert threshold remains, and
half-hour monitoring continues. Shared-filesystem growth is still uncertain;
this engineering adjustment is disclosed rather than represented as proof
that another project's disk consumption is bounded.

Admission verified the exact three existing MPS clients,57% total GPU use,
13,772MiB free GPU memory and a separate8GiB GPU-memory reserve plus2GiB client
allowance. No other project was paused, reprioritized or modified. The frozen
source62ca739, full10M budget, batch256, replay400k and5000-step epochs remain
unchanged. Source and target Torch versions are recorded without a bitwise
equivalence claim.

The recovery ledger correctly waits for Goal2028's scheduled new epoch before
accepting its25,000-step /493-cost source tail. The last successful complete
metadata read had19 records /1,395,000 known extra physical steps /4,412 cost;
this in-flight tail is not yet reported as a confirmed twentieth record.
No extra scientific evaluation or model-verification load was introduced.
All original-parent/path limitations and final scientific gates remain intact.

**5080 remains free of SAFEFLOW training until at least23:50.** Since all seven
original Point cells now have owners on A100/4080, no automatic source resume
is needed after that reservation. The earlier question about automatically
resuming Goal2028 there is now superseded by this placement; even a late
approval must not duplicate an already-owned cell. The dated hold/ownership
note was updated both locally and in the5080 project operations directory.

## Earlier continuation update: 2026-10-24 16:24 CST — Goal2030 epoch confirmed; Button2030 terminal

**Button2030 has completed its full10M-step training budget and original
100-episode final evaluation.** Its native receipt and completed training
summary were written at16:20:44; the16:24 monitor then observed its worker
absent. Targeted metadata retrieval confirmed a passed native receipt, clean
frozen source62ca739 and no engineering hard-stop. It was not restarted.
The [native terminal files](shapo_p47_four_env_artifacts/partial_milestones/native_terminal_receipts/point-button-2030/manifest.json)
have been copied to the local host with their original byte hashes, without
another policy rollout or model load.

| Prespecified native final metric | Button2030 at10M |
|---|---:|
|Mean return|-17.7141|
|Success rate|4%|
|Mean evaluation cost|5.38|
|CVaR-20|23.95|
|Accepted-lineage training cumulative cost|150,109|
|Highest-risk20% normal/tangent variance ratio|0.995862|
|Highest-risk20% dangerous outward velocity|0.028221|

This is **one of five PointButton training seeds**, not the environment's
final aggregate or a matched-control comparison. Task performance is weak
for this seed. The recorded high-risk variance ratio remains close to one;
the same native diagnostic gives mean safety weight0.902672 but reliability
0.003154. These are diagnostic observations, not a scientific stopping gate
or a demonstrated explanation of the outcome. The remaining seeds, frozen
configuration, complete budgets and all final scientific gates are retained.

**Goal2030's A100 GPU1 continuation is also confirmed.** The same PID2508133
logged7,160,000 steps at approximately8.12 FPS, beyond its7.15M full input.
The scheduled16:24 frame verified CUDA/MPS membership and the controller
confirmed the same worker identity and no source duplicate. This is a logged
counter, not a7.16M durable-checkpoint claim. Its [updated continuation
receipt](p4_7_training_receipts_20261022/goal2030_from4t_toa100_gpu1_20261024.json)
contains the complete scheduled new-epoch proof. The recovery ledger now
accepts **19 records,1,395,000 known extra physical steps /4,412 cost units**,
including its25,000-step /248-cost discarded source tail exactly once.
Goal2028's25,000-step /493-cost tail remains conditional, not executed.

Placement now accounts for all20 cells: **nine training-complete, ten running,
one stopped** (Goal2028). A100 runs seven;4080 three;5080 none. The collector's
16:13 cached count was still eight, before Button2030's native terminal
receipt. Its next ordinary collection can import that completed container;
the native completion above is not being misreported as an already cached
ninth cell or as a strict final-audit pass.

All nine older continuing workers advanced relative to15:54. The16:24 snapshot
reports A100 primary16.69GiB free and total GPU utilization44%/45%;4080 primary
15.08GiB, separate4T2467.11GiB and GPU56%. The fast4080 slot is free but its
primary filesystem is below the16GiB admission reserve even before another
capsule, so no additional bulk input is copied there now. No foreign project
or active canonical checkpoint is moved. The5080 reservation remains in
force until23:50; permission for an automatic post-reservation Goal2028 resume
has been requested and is not assumed. Known original-parent/multi-hop path
limitations and final scientific-certification requirements remain unresolved.

## Earlier continuation update: 2026-10-24 16:15 CST — Goal2030 continuing; eight native terminal results retained

The stopped **Goal2030** cell has now launched on shared A100 GPU1 from its
**7,150,000-step full checkpoint**, retaining the original10M-step budget.
The98 recorded scientific files were streamed directly from the pinned4080-4T
raw archive to a previously absent A100 target and every file hash was checked
before launch. This is the recorded scientific-suffix scope, not a complete
filesystem clone. The5080 original remains intact; no5080 GPU task or source
signal was used. Its [continuation receipt](p4_7_training_receipts_20261022/goal2030_from4t_toa100_gpu1_20261024.json)
records the exact input, copied scope, launch and resource checks.

**The first new epoch remains pending.** PID2508133 is live with the correct
seed2030/full budget and has reached native STAR training initialization.
The controller waits for the next ordinary half-hour progress observation;
neither liveness nor successful file restoration substitutes for the new-epoch
proof. It will not relaunch if that observation is delayed or times out.

Placement is now **eight training-complete, eleven launched continuing workers,
one still-stopped cell** (Goal2028,6.3M full input). A100 has seven workers;
4080 four. The newly launched cell's first epoch is explicitly unconfirmed.
5080 remains reserved until23:50 with no SAFEFLOW training. No new experiment,
scientific configuration, evaluation or reduced training budget is introduced.

Before admission, [six inactive A100 export sets](p4_7_training_receipts_20261022/inactive_a100_export_offload_20261024_receipt.json)
were copied and rehashed on the private separate data disk. Their canonical
containers were absent on A100. Only redundant old export copies were released;
the real export directories keep dated location notes and may receive future
fresh exports. Original training and all other projects were untouched. These
six export sets include older host-local copies of cells now on4080, not new
resume inputs. Primary free space recovered to17.915GiB before admission.

The smaller Goal2030 capsule fits that headroom with the **16GiB post-copy
primary reserve retained** (16.285GiB observed after restoration). Unlike the
earlier4080 restoration helper, this controller does not require an additional
half-GiB copy slack above that reserve. GPU1 had36% total utilization, about
78,674MiB free GPU memory and the exact three expected owned clients at admission;
it now shares that pool without pausing or reprioritizing another project.
This smaller-capsule placement, ahead of the larger Goal2028 input, is based on
available storage/resources, not reward, cost or scientific result selection.

The metadata ledger deliberately refuses final recovery annotation until this
same target produces the required scheduled new epoch. Its known consumed
source tail is25,000 steps /248 cost; after confirmation it will be charged
exactly once, separately from accepted-lineage learning costs. Goal2028's
remaining conditional source tail is25,000 steps /493 cost. Unknown unlogged
tails remain unknown, not zero. The source/target CUDA builds differ and are
recorded without any bitwise-equivalence claim. All final scientific gates,
original-parent gaps and pending multi-hop path limitations remain unchanged.

The eight already cached terminal cells' **24 existing JSON files** (native
receipt, training summary and5M evaluation for each cell) have now been copied
back to the local host with their byte hashes. The [copy manifest and native
files](shapo_p47_four_env_artifacts/partial_milestones/terminal_native_receipts_available8_20261024/manifest.json)
retain the original results; no policy was evaluated again and no remote file
was changed. These eight were selected solely as all cells already terminal
at the cache migration, not by performance. They are not the full five-seed
result for either environment and are not a final scientific certification.

| Existing cell | Final budget | Native mean return | Failure rate | Native mean cost | CVaR-20 |
|---|---:|---:|---:|---:|---:|
|Ant2027|5M|-1027.77|27%|0.27|1.00|
|Ant2028|5M|2994.90|25%|0.25|1.00|
|Ant2030|5M|6526.60|4%|0.04|0.20|
|Ant2031|5M|3673.26|57%|0.57|1.00|
|Walker2027|5M|5672.55|3%|0.03|0.15|
|Walker2029|5M|4209.48|37%|0.37|1.00|
|Walker2030|5M|1141.98|13%|0.13|0.65|
|Walker2031|5M|2669.40|84%|0.84|1.00|

Each native final evaluation already contains the fixed five evaluation seeds
times20 episodes (100 episodes per training seed). Locomotion failure is not
Point-task success. **Seed-to-seed outcomes are heterogeneous**, including the
negative Ant2027 return and high Walker2031 failure rate. These observations
are retained without replacing a seed, switching to a best checkpoint or
changing the frozen method. There is no matched C0 comparison or causal
superiority claim here; the remaining cells still run their complete budgets.

## Earlier continuation update: 2026-10-24 15:47 CST — terminal cache on data disk;5080 reserved

The20-cell matrix remains **eight training-complete, ten running, two stopped**.
A fresh read-only process/native-CSV check confirmed six A100 workers and four
4080 workers; every continuing cell advanced relative to the14:24 frame.
These are logged training counters, not claims that each latest row is a durable
full checkpoint or that final scientific certification has passed.

| Host | Existing training cell | Latest logged steps | Fixed budget |
|---|---|---:|---:|
| A100 | Walker2028 |4,920,000|5,000,000|
| A100 | Ant2029 |4,100,000|5,000,000|
| A100 | Goal2027 |9,395,000|10,000,000|
| A100 | Button2027 |9,095,000|10,000,000|
| A100 | Goal2031 |8,535,000|10,000,000|
| A100 | Button2031 |8,470,000|10,000,000|
|4080|Button2029|8,105,000|10,000,000|
|4080|Goal2029|5,670,000|10,000,000|
|4080|Button2028|7,710,000|10,000,000|
|4080|Button2030|9,965,000|10,000,000|

Button2030's remaining training is approximately25 minutes at its observed
23.95-step/s epoch rate, followed by its original native evaluation. This is
an estimate, not a terminal receipt. Goal2028/2030 remain stopped with their
previously pinned6.3M/7.15M full inputs; no extra seed or reduced budget is used.

**5080 has no SAFEFLOW training worker.** A new user reservation keeps it free
of this project's training until **2026-10-24 23:50 CST**, at least eight hours
from the check. Its other workloads are untouched. The dated hold is recorded
locally and on5080; no obsolete seven-worker resume script is scheduled, since
five of those cells now run elsewhere. After the hold, cell ownership and
current user workload must be checked before any placement. The existing
read-only half-hour monitor is live and does not launch jobs automatically.

Two storage-only operations are now complete:

- [Eight inactive A100 directories](p4_7_training_receipts_20261022/cold_a100_directory_offload_20261024_receipt.json)
  were copied and rehashed in the project's private separate-disk namespace.
  Seven are cold recovery archives and one is a legacy Ant run, not a current
  formal-matrix source. Exact old read paths remain same-content links; no
  active canonical training directory was moved.
- [Eight terminal assembly copies](p4_7_training_receipts_20261022/terminal_result_cache_migration_20261024_receipt.json)
  now reside on A100's separate data disk. Only four redundant primary import
  replicas were released after verification; their4080 originals and all
  native A100 source containers remain intact. The replacement collector is
  running and reports eight readable cached terminal cells. Future completed
  imports target this data cache, not the primary training filesystem.

The collector still calls the unchanged frozen final validator with a run-root
argument. No validator fix, path alias, original-parent reconstruction, extra
rollout, trainer signal or other-project file operation was introduced. Storage
relocation is not final scientific certification and does not waive the known
multi-hop/original-parent gaps. Metadata accounting remains **18 training records,
1,370,000 known extra physical steps /4,164 cost**; these storage moves add zero.

At15:47, primary free space was A100 **16.60GiB** and4080 **15.39GiB**;4080's
separate4T archive had **2468.23GiB** free. Shared GPU utilization was A10040%/34%
and408066%. Primary headroom still constrains new bulk admissions, even though
GPU capacity exists. Only SAFEFLOW files have been handled; permission to move
another project's old archives has not been assumed. The next placement will
reuse a verified free slot with adequate disk headroom rather than duplicate a
running cell. Full budgets and all final evidence gates remain unchanged.

## Earlier continuation update: 2026-10-24 14:24 CST — Goal2029 epoch confirmed

**Goal-2029's new4080 epoch is now confirmed.** The same PID1831158 advanced
from its5,500,000-step input to **5,545,000 logged steps**, with the observed
epoch at approximately **25.16 FPS**. The scheduled14:24 frame verified its
CUDA/MPS membership; the controller then confirmed the same worker identity
and no source duplicate. This is a logged observation, not a claim of a new
5.545M durable full checkpoint. The [updated continuation receipt](p4_7_training_receipts_20261022/goal2029_from4t_to4080_20261024.json)
contains the exact full input, restored scientific-file scope, preserved old
target and scheduled-epoch proof.

Counts remain **eight training-complete cells, ten continuing cells, and two
stopped cells** (Goal2028/2030). All nine older workers also had positive
logged deltas in the14:24 snapshot. A100 runs six workers,4080 four, and5080
none of this project's training. No seed or training budget was cut and no
new scientific cell or evaluation was added.

The raw-recovery record now adds its known **25,000 repeated physical steps
/44 cost** exactly once. The complete metadata ledger passes with **18 records,
at least1,370,000 extra physical steps /4,164 cost units**; the two remaining
unconsumed incident tails are conditional (**50,000 steps /741 cost**).
Unknown unlogged work remains unknown, not zero, and physical-work corrections
are separate from accepted-lineage learning-curve costs. A metadata-only
variable-name collision in the new storage-record reader was corrected before
this successful ledger read; neither training nor any scientific gate changed.

The14:24 capacity frame reports4080 primary **17.74GiB free**, its separate4T
archive2471.54GiB, and A100 primary10.68GiB. Shared total GPU utilization was
A10042%/35% and408067%; capacity is available, but primary storage is still
constraining further bulk admissions. Only this project's verified cold data
has been relocated. Approval to touch another project's idle archives has not
been received and no such operation has been performed. Training, backups and
half-hour observation continue; original-parent/path limitations and all final
scientific-certification requirements remain unresolved and unwaived.

## Earlier continuation update: 2026-10-24 — space recovered; Goal2029 epoch pending

**Eight cells have now completed training**, including the newly collected
Ant-2028 and Walker-2029 terminal receipts. These are training receipts, not
final strict scientific certification. The six A100 workers continued, while
the4080 pool fell from five to three workers as those two jobs finished.

The13:24 snapshot found only **6.43GiB free on4080's project filesystem**.
Two storage-only operations have now preserved this project's cold data on
the separate4T disk before releasing redundant primary copies:

- [Eight named cold archive directories](p4_7_training_receipts_20261022/cold_4080_directory_offload_20261024_receipt.json)
  were copied and rehashed; their original read paths remain same-content
  directory links. The three archive parent directories remain real primary
  directories, preserving future same-filesystem retirement renames.
- [Three obsolete4080 export sets](p4_7_training_receipts_20261022/inactive_4080_export_offload_20261024_receipt.json)
  were separately copied and verified. Their canonical containers were absent
  on4080. Original export directories remain real primary directories with
  historical-location notes for later fresh exports; no active export was
  removed. Some same-named cells run elsewhere, so these are explicitly old
  host-local copies, not the newer active inputs.

Observed primary free space recovered to about **18.7GiB**. Complete archived
contents and manifests remain recoverable on4T; only verified redundant primary
replicas were removed. No canonical training container, active input, GPU task
or other project was moved, stopped or reprioritized. Free counters include
concurrent shared-filesystem activity; the cause of the broader space decline
is still unproven. Approval has been requested before any transfer of another
project's idle archives; no such transfer has been performed.

**Goal-2029 has now launched on4080 from its5,500,000-step full checkpoint**,
using the previously pinned raw4T archive. All92 recorded scientific-evidence
files were restored, reusing identical historical checkpoint/CSV prefixes;
the entire pre-restore target was independently backed up on4T first. Recorded
scientific suffix coverage is not represented as a complete filesystem clone.
The canonical target remains a real directory and its input SHA matches the
original incident inventory. The original5080 container remains intact and no
source GPU job was launched or signalled. The frozen source62ca739, seed2029,
10M budget, batch256, replay400k and5000-step epochs are unchanged.

The13:54 ordinary snapshot verifies new PID1831158's CUDA/MPS connection beside
the three continuing4080 PointButton workers. **Its first new epoch remains
pending**: process/GPU liveness is not substituted for that proof. The [in-flight
recovery receipt](p4_7_training_receipts_20261022/goal2029_from4t_to4080_20261024.json)
records input identity, restored files, preserved old target and launch state.
The controller waits on the existing scheduled JSON, not repeated SSH checks;
it will not auto-relaunch the worker if an observation window expires.

Placement now accounts for all20 cells: **eight trained, ten live processes,
two still stopped** (Goal2028/2030). A100 has six workers;4080 now has four;
5080 has none of this project's training. Any future5080 recovery must exclude
all five Point cells already running elsewhere, including this in-flight Goal.
After restoring the missing files,4080 primary free space was about17.5GiB;
further bulk admissions remain deferred at this headroom. A100 had10.80GiB in
the13:24 snapshot and is also not admitting another capsule now.

The final ledger explicitly waits for Goal2029's new-epoch proof. Its known
25,000-step /44-cost source tail will then be counted exactly once. Storage
copies add no training interactions or scientific evaluations. Full budgets,
fixed evaluation protocol, and original final gates remain intact; historical
original-parent and multi-hop path limitations are not repaired or waived by
these storage and continuation operations.

## Earlier storage update: 2026-10-23 09:13 CST — native archive cycles confirmed

Future **temporary raw-evidence snapshots** now use the separate disks:

| Origin | New temporary snapshot parent |
|---|---|
| A100 | `/storage_0/SAFEFLOW2027_lzqw_uid1007/evidence_staging` |
| 4080 | `/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_source_staging` |

Naturally scheduled manifests from active **Walker-2028 on A100** and
**Walker-2029 on4080** confirm that these locations are on filesystems separate
from training. Their small-manifest hashes and storage metadata accompany the
[deployment/native-cycle receipt](p4_7_training_receipts_20261022/evidence_data_staging_deployment_20261023.json).
No extra snapshot was invoked, and the same archive coordinator PID67233 was
not restarted. Its existing background priority/affinity and half-hour cadence
remain unchanged. The remote helpers were atomically replaced only after their
old hashes matched; original versions remain beside them for inspection.

This changes **only future temporary cache placement**. Canonical training
containers, checkpoint inputs, existing archives, scientific source/configs,
and path-resolution code were neither moved nor aliased. Both old home and new
temporary locations remain accepted for safe cleanup of in-flight snapshots.
New data-disk snapshot admission preserves100GiB for other users; insufficient
cache space defers archiving, not scientific training. No trainer or foreign
project was signalled, paused or reprioritized, and no model/evaluation was run
to verify this change.

The08:53 ordinary snapshot still showed **all eleven continuing cells with
positive step deltas**, six terminal training receipts, and three stopped
Goal cells with pinned4T recovery locations. A100 GPU totals were96%/95% and
4080's90%, including shared foreign work.5080 had no training workers. Available
space was A100 home19.11GiB,4080 project255.29GiB, and separate4T2526.11GiB.

A one-time scoped size check measured this project's primary allocations at
approximately31GiB on4080 and21GiB on A100. **The cause of the filesystem-wide
free-space decline was not established**; no other project was inspected or
modified. Moving future cache copies reduces transient primary-disk pressure,
but is not claimed to cure that unexplained trend. Further cold admissions
remain deferred at the current A100 headroom. Primary-space monitoring continues.

This storage operation adds no scientific interaction: the physical-work ledger
remains17 records /1,345,000 extra steps /4,120 cost. Full budgets and all final
scientific gates remain unchanged, including unresolved original-parent and
multi-hop path limitations. It is not final experiment certification.

## Earlier readiness update: 2026-10-23 07:47 CST — no new training launch

The existing4080-4T raw-evidence archive already contains the recorded recovery
inputs for all three still-stopped cells. Their immutable manifest identities
are now pinned in a [recovery-location receipt](p4_7_training_receipts_20261022/remaining_goal_archive_inputs_20261023.json),
so no redundant multi-GB transfer or new source snapshot was needed.

| Planned cell | Recorded full input | Remaining fixed10M budget | Archived raw files |
|---|---:|---:|---:|
| Goal-2029 | 5.500M | 4.500M | 92 |
| Goal-2028 | 6.300M | 3.700M | 109 |
| Goal-2030 | 7.150M | 2.850M | 98 |

This metadata-only preparation checked each small manifest's own SHA, its
checkpoint identity against the existing incident inventory, and the presence
and sizes of its recorded blobs. Raw bytes were previously SHA-verified at
archive ingestion; **no fresh blob-content hash or tensor load was performed
by this preparation**. Raw-manifest suffix coverage is not represented as a
newly inventoried complete filesystem capsule or a final scientific audit.

Future admission must recheck live placement, capacity and exact input bytes;
these locations alone do not authorize or execute a launch. Priority is largest
remaining fixed budget first, not selection by reward or safety outcome.
No GPU job, data transfer, evaluation or scientific cell was added, and5080
remains without training. Counts are unchanged: six trained, eleven continuing,
three stopped. The physical-work ledger is still17 records /1,345,000 extra
steps /4,120 cost; the three unconsumed incident tails are still conditional.
No original2027 parent gap, path-locator issue or final gate was repaired,
aliased or waived by pinning these later recovery inputs.

## Earlier continuation update: 2026-10-23 07:35 CST — Button2031 confirmed

**PointButton-2031 has completed its first new A100 epoch**: PID1679029
advanced from7,650,000 to **7,655,000 steps**, approximately **6.04 FPS**.
CUDA/MPS membership on GPU0 was verified alongside Goal2027/2031. The same
worker was followed from launch, without a duplicate restart. Its [updated
receipt](p4_7_training_receipts_20261022/point_button_2031_cold_to_a100_20261023.json)
contains the complete source/archive/input identity and first-epoch proof.
This CSV observation is not a claim that a new7.655M full checkpoint exists.

There are now **six terminal training receipts, eleven continuing cells and
three stopped cells**. A100 runs six (three per GPU),4080 runs five, and5080
runs zero. The remaining stopped cells are Goal2028/2029/2030. All four relocated
Point cells (Goal2027/2031 and Button2027/2031) have verified new A100 epochs;
exclude them from any future5080 restart. The07:23 snapshot showed positive
progress for the older workers and Goal2031 beyond its first epoch; the
new Button epoch is separately confirmed at07:35.

The completed-proof resource ledger now contains **17 records, at least
1,345,000 extra physical steps /4,120 cost units**, including this recovery's
20,000-step /290-cost logged rollback exactly once. Only the three remaining
stopped incident tails are conditional: **75,000 steps /785 cost**, not added
as executed recovery. Unknown unlogged work is not zero; physical-work totals
remain separate from accepted-lineage learning curves.

The last ordinary capacity snapshot remains07:23: A100 home19.61GiB free,
4080 project264.75GiB, separate4T archive2527.18GiB. With A100 below its
unchanged20GiB warning, no further cold-copy admissions are planned at this
headroom. Existing training, half-hour monitoring, backups and receipt
collection continue. No foreign job was paused, no5080 GPU job started, no
additional scientific rollout or cell added, and no budget/configuration/final
gate changed. Scientific certification is still pending, including the
previously disclosed original-parent and path-resolution limitations.

## Earlier recovery update: 2026-10-23 07:23 CST — Button2031 epoch pending

The complete **3,015,931,202-byte PointButton-2031 container** has been copied
to A100, its **7,650,000-step full input** matched to the incident inventory's
SHA and pinned on4080's separate4T disk, and the original preserved in its
5080 cold archive. PID1679029 is live on A100 GPU0. The ordinary07:23 snapshot
verified its CUDA/MPS membership alongside Goal2027/2031. **Its first new
epoch is still pending**, so a live process/GPU connection is not treated as
completed recovery proof. The same controller/session is being followed.
See the [in-flight Button2031 receipt](p4_7_training_receipts_20261022/point_button_2031_cold_to_a100_20261023.json).

All20 planned cells are accounted for: **six terminal training receipts,
eleven live training processes, three stopped cells**. A100 has six processes
(three on each GPU),4080 has five, and5080 has zero. All nine older workers
had positive logged deltas; Goal2031 reached7.710M, beyond its confirmed first
7.705M epoch. The remaining stopped cells are Goal2028/2029/2030. Any future
5080 restart must exclude all four relocated Point cells, including this
in-flight Button worker; do not recreate the original seven blindly.

After the full3GB copy, A100 home free space is **19.61GiB**, below the
unchanged20GiB monitoring warning but above the16GiB operational reserve.
No further cold-copy admissions are planned with this headroom. The4080
project and separate4T archive have264.75GiB and2527.18GiB free. Other projects
remain untouched and GPU sharing continues. No5080 GPU training, extra
evaluation, new scientific cell, configuration change or budget reduction
was introduced.

The final ledger requires Button2031's new-epoch proof and will refuse to
finalize while it is pending. Its known20,000-step /290-cost incident tail
will be counted once that proof is available. The last completed-proof ledger
still contains16 records (1,325,000 extra physical steps /3,830 cost), and no
unknown unlogged work is assumed zero. Training completion remains distinct
from final scientific certification; historical parent/path limitations and
all original gates remain unchanged.

## Earlier continuation update: 2026-10-23 07:01 CST — Goal2031 confirmed

**Goal-2031's first new A100 epoch is now confirmed**: PID1581960 continued
from7,700,000 to **7,705,000 steps**, approximately **6.08 FPS**, with verified
CUDA/MPS membership on GPU0. The [updated recovery receipt](p4_7_training_receipts_20261022/point_goal_2031_cold_to_a100_20261023.json)
contains the first epoch, worker identity, exact checkpoint SHA and preserved
source/input locations. The same worker was followed throughout; no automatic
relaunch or duplicate experimental cell was introduced.

Current placement remains **six terminal training receipts, ten running cells,
and four stopped cells**. The06:53 ordinary snapshot verified positive step
deltas for all nine older workers, and live MPS membership for Goal2031 before
its first epoch. A100 hosts Goal2027/2031 on GPU0 and Button2027, Ant2029 and
Walker2028 on GPU1;4080 hosts Button2028/2029/2030, Ant2028 and Walker2029.
5080 has zero training workers; the remaining stopped cells are Goal2028/2029/2030
and Button2031. Do not recreate any of the three relocated Point cells there.

The06:53 snapshot measured A100 home **22.42GiB free**, above the unchanged
20GiB warning after the backup relocation and this complete-container copy.
4080 project and separate4T archive free space were267.13GiB and2527.81GiB.
The backup service's new physical destination has already accepted a subsequent
snapshot. No training container, foreign project or scientific validator was
changed by storage maintenance, and no5080 GPU training was launched.

Goal2031's consumed logged tail is **25,000 repeated steps /308 cost**.
The metadata-only physical-work ledger now has **16 records, at least1,325,000
extra steps /3,830 cost units**. The other four incident tails remain
conditional (**95,000 steps /1,075 cost**) until their own executed recoveries.
Unknown unlogged work is not assumed zero; these adjustments are separate from
accepted-lineage curves. Full budgets, fixed seeds/evaluations and scientific
gates remain intact. Historical2027 parent gaps and the pending multi-hop
locator change are not repaired or waived by these continuations.

## Earlier recovery update: 2026-10-23 06:48 CST — first epoch pending

**PointGoal-2031 is now a live A100 GPU0 process**, continuing its existing
10M cell from the incident inventory's SHA-matched **7,700,000-step full
checkpoint**. The full1,524,676,884-byte cold container was copied through
the local host, the exact input pinned on the separate4080-4T disk, and the
source moved intact to its cold archive. No5080 GPU training was launched.

A direct06:48 process check found PID1581960 running, with the expected
Goal-2031 configuration, seed2031, budget10M and GPU0 UUID. **The first new
epoch and final CUDA/MPS recovery confirmation are still pending**; a live
process is not substituted for that proof. The same controller/session is
being observed, with no automatic duplicate restart. Its [in-flight receipt](p4_7_training_receipts_20261022/point_goal_2031_cold_to_a100_20261023.json)
will be updated after confirmation or an engineering exception.

This brings process placement to **A100 five /4080 five /5080 zero**, alongside
six terminal training receipts and four still-stopped cells: Goal2028/2029/2030
and Button2031. These counts distinguish a launched worker from a verified new
epoch. Before any future5080 restart, exclude all three Point cells already
moved to A100 (Goal2027/2031 and Button2027), including this in-flight worker.
No other project's process, scientific setting, total budget or acceptance gate
was changed. The final resource ledger explicitly refuses to finalize while
this recovery lacks its required first-epoch proof; its incident-tail accounting
will be added once that proof exists, not silently omitted or counted twice.

## Earlier storage update: 2026-10-23 06:38 CST

The A100 secondary replicas of4080 checkpoints now live on **A100's separate
data disk**, under the new private, UID1007-owned project namespace
`/storage_0/SAFEFLOW2027_lzqw_uid1007/recovery_backups/from_4080`.
The existing home path remains a symlink to those same backups. Only our
backup writer was briefly held/stopped while idle; **no trainer or foreign
project was signalled or paused**, and no scientific code/path resolver changed.

All regular-file bytes, relative paths, modes, modification times and internal
hard-link groups matched before switching paths. Only then was the redundant
home copy removed; its entire content remains recoverable through the original
logical path and on the separate data disk. Logical payload8,291,468,873 bytes
includes hard-linked recovery names; home free space increased from
19,892,809,728 to25,600,233,472 bytes (about **5.32GiB reclaimed**).
The data disk retained about150GiB free, including the100GiB migration reserve.
These are operation-time counters, not a later whole-cohort capacity snapshot.

The same backup service restarted with PID501665 and successfully installed
another4080 snapshot through the new explicitly allowlisted canonical path.
Training budgets, canonical training containers, histories and full scientific
gates were not changed. The [storage relocation receipt](p4_7_training_receipts_20261022/a100_backup_replica_storage_relocation_20261023.json)
contains the complete content manifest and service identity.5080 training
remains off. With the recovered home headroom, one additional existing stopped
cell, Goal-2031, is being copied for A100 GPU0; this is **not yet a confirmed
new training epoch** and is not counted in the executed-recovery ledger yet.

## Earlier continuation update: 2026-10-23 06:31 CST

**PointButton-2027 has also resumed on A100 GPU1**, from its SHA-verified
8,250,000-step full checkpoint. At06:11, PID1442660 had logged its first new
epoch at **8,255,000 steps /6.71 FPS**, with verified CUDA/MPS membership beside
the existing Ant-2029 and Walker2d-2028 clients. The full cold container was
copied through the local host; its exact resume input was pinned on the4080-4T
filesystem, and the source container was preserved in its cold archive.
The [PointButton recovery receipt](p4_7_training_receipts_20261022/point_button_2027_cold_to_a100_20261023.json)
records the checkpoint SHA, source archive, target worker and first epoch.
This is continuation of an existing cell, not a new experiment or a claim of
cross-device bitwise equivalence.

The ordinary **06:23 snapshot** now accounts for all20 planned cells:

| State /host | Cells | Evidence |
|---|---:|---|
| Terminal training receipts | 6 | Final strict scientific audit still pending |
| A100 running | 4 | Goal-2027 on GPU0; Button-2027, Ant-2029 and Walker-2028 on GPU1 |
| 4080 running | 5 | Button2028/2029/2030, Ant-2028 and Walker-2029 |
| 5080 stopped | 5 | Goal2028/2029/2030/2031 and Button2031; zero training workers |

All eight previously running workers had positive logged step deltas; the ninth
is the newly confirmed Button worker. Goal-2027 reached8.545M and Button-2027
8.255M. These are CSV observations, not assertions of newly durable full
checkpoints. The original8-hour5080 reservation was honored on22October;
following the separate23:34 exits, no5080 GPU training has been relaunched.
Pending clarification about those exits, do not blindly recreate the original
seven workers: both2027 Point cells are already continuing on A100.

A100 home has **18.75GiB free**, below the unchanged20GiB monitoring warning.
This single cold recovery retained the16GiB operational storage reserve; no
further cold-copy admissions are planned with the current headroom. The4080
project disk has269.56GiB free and its separate4T archive has2528.10GiB free.
No archive or unrelated file was deleted, no foreign job paused, and no5080
GPU training started. Half-hour observation, backups and receipt collection
remain active. Budgets, seed/configuration, optimizer/replay/RNG restoration
and the frozen scientific source62ca739 remain unchanged; no additional
scientific evaluation or final-gate relaxation was introduced.

Button's known logged rollback is **20,000 steps /250 cost**, included once.
The metadata-only late-resource ledger now contains **15 records, at least
1,300,000 extra physical steps /3,522 cost units**. The remaining five stopped
incident tails are still conditional: **120,000 steps /1,383 cost**, not yet
counted as executed recovery. Unknown unlogged work remains unknown, not zero;
physical-work adjustments are not added to accepted-lineage learning curves.
The earlier2027 parent-identity gaps and pending multi-hop path-locator change
remain unresolved; training completion is not final scientific certification.

## Earlier continuation update: 2026-10-23 04:05 CST

One of the seven stopped 5080 cells, **PointGoal-2027**, is now continuing on
**A100 GPU0**, without restarting training on5080. Its complete cold container
was copied, the **8,500,000-step full checkpoint** matched the incident
inventory's SHA, and that exact input was pinned on the separate4080-4T disk.
The new worker (PID1073197) logged **8,505,000 steps** at approximately
**4.83 FPS**; CUDA/MPS client membership on A100 GPU0 was verified at04:05.
This restores progress to a stopped cell; it does not claim greater throughput
than the previous5080 placement or a proven whole-cohort speedup.

| Host | Running unfinished cells | Placement |
|---|---:|---|
| A100 | 3 | PointGoal-2027 on GPU0; Ant-2029 and Walker2d-2028 on GPU1 |
| 4080 | 5 | PointButton2028/2029/2030, Ant-2028 and Walker2d-2029 |
| 5080 | 0 | Six cells remain stopped; no GPU training launched here |

Together with **six terminal training receipts**, this accounts for all20
planned cells: six trained, eight running, six stopped. The last whole-cohort
snapshot at03:53 recorded positive deltas for the other seven running cells
and the new Point process live, before its first epoch. A100 home free space
was20.45GiB in that snapshot. Admission of this single1.26GB capsule preserves
a16GiB operational reserve; the existing20GiB monitoring warning is unchanged.
No old archive, unrelated file or other project's process was removed.

The original5080 container was moved, not deleted, into its cold recovery
archive after the target copy/hash and4T input were verified. Its location,
the new A100 log, checkpoint identity, runtime versions and worker proof are
in the [cold-recovery receipt](p4_7_training_receipts_20261022/point_goal_2027_cold_to_a100_20261023.json).
The scientific source remains62ca739, with the same seed, optimizer/replay/RNG
state, frozen configuration and **10M budget**. PyTorch changed from2.7.1+cu128
to2.7.1+cu118; bitwise cross-device equivalence is not claimed. No trainer was
signalled, no extra scientific evaluation performed and no5080 GPU job started.

The known logged rollback for this recovered cell is **25,000 steps /199 cost**,
from its8.525M last CSV back to the durable8.5M input. It is now included once
in the late physical-work ledger: **fourteen records, at least1,280,000 extra
physical steps and3,272 cost units**. The other six incident tails remain
conditional (**140,000 steps /1,633 cost**) until their own recovery is actually
executed. Unknown unlogged work remains unknown, not zero; none of these
physical-work adjustments are added to accepted-lineage learning-curve costs.

The pending question concerns use of the5080 GPU. Continuing an already planned
cell on the separately authorized A100 does not restart5080. Before any future
5080 recovery, check current placement and exclude PointGoal-2027, which is
already running on A100; do not recreate all seven old workers blindly.
Historical2027 original-parent gaps and the unapproved multi-hop locator change
remain unresolved. No final scientific gate is waived by this recovery.

## Earlier incident update: 2026-10-23, after the 23:53 snapshot

The 23:53 ordinary snapshot found **zero 5080 training workers**. A subsequent
direct process check confirmed all seven recorded PIDs absent. MPS logs record
all seven clients disconnecting at **2026-10-22 23:34:35 CST**. Application logs
end after ordinary epochs, without a Python traceback; the checked kernel
journal contains no OOM/GPU error for the incident window. The cause is **not
established**, and the absence of an error does not prove either a crash or an
intentional user stop.

| Host | Current evidence |
|---|---|
| A100 | Two continuing cells; Walker-2028 reached 3.910M and Ant-2029 3.180M in the 23:53 snapshot |
| 4080 | Five continuing cells, all with positive logged step deltas |
| 5080 | Seven unfinished cells stopped; no automatic relaunch performed |

There are still **six terminal training receipts**: seven further cells are
running and seven are stopped. The Walker handoff below remains valid; its
new A100 worker continued after the first verified epoch. No other project's
job has been paused or terminated.

The [read-only checkpoint inventory](p4_7_training_receipts_20261022/incident_5080_2334_checkpoint_inventory.json)
successfully loaded all seven full recovery checkpoints on CPU, verified that
replay/RNG/optimizers are present, and recorded their counters and SHA hashes.
Saved PointGoal counters are 8.50M/6.30M/5.50M/7.15M/7.70M for seeds 2027–2031;
PointButton 2027/2031 counters are 8.25M/7.65M. All remain below their original
10M budgets. No recovery launch, signal to a trainer, or new evaluation was
performed by this inventory.

The user has been asked whether these simultaneous exits were intentional and
whether shared 5080 training may resume, to avoid overriding a manual resource
reservation. A100/4080 training, half-hour observation and backups continue.
If all seven restart from these recorded checkpoints, their known logged tails
would add **165,000 repeated physical steps and 1,832 training-cost units**;
these are conditional recovery accounting, not yet added as an executed
handoff, and unknown unlogged work is not assumed zero. Historical scientific
provenance limitations remain unchanged. No full-budget or final-audit gate
has been reduced or waived.

## Earlier operational update: 2026-10-22 23:37 CST

Walker2d-2028 continued from its ordinary **3,900,000-step full checkpoint** on
A100 GPU1. At 23:36, the new worker (PID 333807) had logged **3,905,000 steps** at
approximately **10.08 FPS**, with verified CUDA/MPS membership. Only after this
new epoch was observed did the held 4080 worker exit through its normal
CUDA-synchronizing signal handler. Its complete source container remains in the
4080 recovery archive; no history was deleted. The exact resume input was also
SHA-verified on the separate 4080-4T filesystem.

| Host | Unfinished cells placed after this handoff | Placement |
|---|---:|---|
| A100 | 2 | Ant-2029 and Walker2d-2028 on GPU1, sharing our existing MPS runtime |
| 4080 | 5 | PointButton 2028/2029/2030, Ant-2028 and Walker2d-2029 |
| 5080 | 7 | Five PointGoal seeds and PointButton 2027/2031; not moved by this handoff |

There are still **six terminal training receipts and fourteen unfinished cells**.
This is an operational handoff, not another completed scientific run. The last
ordinary whole-cohort snapshot at 23:23 recorded positive step deltas for all
fourteen cells. Periodic snapshots can catch a temporarily stopped worker; a
subsequent 5080 read-only observation returned running states without any resume
signal from this handoff. A transient observation is not used to trigger a
duplicate restart.

The purpose of moving the shorter Walker continuation is to release 4080
capacity for longer remaining cells. Its first A100 epoch does **not** by itself
prove an overall makespan improvement; subsequent ordinary half-hour snapshots
will show the cohort effect. No training seed, learner setting, evaluation
schedule, 10M/5M budget, or other project's process was changed. Both sides used
PyTorch 2.7.1+cu118; bitwise cross-device equivalence is not claimed.

The proposed PointGoal-2027 move was **not performed**: another project returned
to the previously empty A100 GPU0 before admission, and by 22:53 its utilization
was 84%. Preflight stopped before any source signal or target launch. That
5080 task stayed in place. Other projects were neither paused nor terminated.

The [Walker handoff receipt](p4_7_training_receipts_20261022/walker2d_2028_4080_to_a100.json)
contains checkpoint counters/hash, immutable input/archive paths, worker identity,
first-epoch evidence and source-retirement location. The
[declined Point preflight](p4_7_training_receipts_20261022/idle_a100_short_cell_preflight.json)
records that no Point migration occurred. The checkpoint and last logged Walker
counter both equal 3.9M, so this handoff adds no **known logged** rollback; any
unlogged in-flight tail remains unknown, not assumed zero. The late-resource
ledger now contains thirteen records, with its known lower bounds unchanged at
**1,255,000 additional physical steps and 3,073 training-cost units**. These are
not added to accepted-lineage learning-curve metrics.

The full scientific final audit remains pending. Existing historical 2027
checkpoint-parent gaps and the unapproved multi-hop path-locator change remain
explicitly unresolved; this resource handoff does not repair or waive them.

## Earlier running status: 2026-10-22 19:23 CST

The eight-hour 5080 reservation has ended. Dispatch occurred at **19:11:00 CST**,
after the not-before deadline. The ordinary 19:23 snapshot observed exactly
seven expected new workers, each with its recorded PID, verified CUDA/MPS
execution, and **10,000 steps beyond its saved checkpoint**. Last epoch rates
were approximately **19.4–20.0 FPS**. This is observed training progress, not
merely a successful timer exit.

| Host | Distinct active cells | Status |
|---|---:|---|
| 5080 | 7 | Five PointGoal seeds and PointButton 2027/2031 resumed |
| 4080 | 6 | Three PointButton, two Walker2d and one Ant cell continuing |
| A100 | 1 | Ant-2029 continuing |

Together with six terminal training receipts, this accounts for all twenty
planned cells: **six trained, fourteen actively training, none held**. The full
scientific final audit remains pending. All Point budgets remain 10M steps and
Ant/Walker budgets 5M; no other projects were stopped and no additional
evaluation was launched to confirm this restart.

The [dispatch record](p4_7_training_receipts_20261022/hold_5080_8h_resumed.json)
and [new-epoch confirmation](p4_7_training_receipts_20261022/resume_confirmation_5080.json)
retain the actual timestamps, saved counters, worker identities and log paths.
Recovery starts from durable checkpoints rather than the later pre-stop CSV
counters; the already-disclosed discarded tails are not counted twice. The
publisher now preserves this confirmation alongside the unchanged physical-work
ledger. Historical checkpoint-ancestry gaps and the unapproved multi-hop locator
change are not silently treated as repaired.

## Fixed 40%-budget results: all twenty cells available

This is **descriptive mid-training context, not the final scientific report**.
All four environments now have their five prescribed training seeds at the
fixed 40% milestone: 4M steps for Point and 2M for Ant/Walker. Each native result
uses the scheduled five evaluation seeds × twenty episodes. Eighteen local raw
files were reused and two newly archived Point-2029 JSON files retrieved; no
policy inference, additional evaluation, training restart, configuration search,
or early stop was performed.

| Environment | Return, mean ± sample SD | Task metric, mean | Evaluation cost, mean | CVaR-20, mean |
|---|---:|---:|---:|---:|
| PointGoal, 4M | -0.379 ± 0.267 | Success 4.6% | 9.442 | 46.710 |
| PointButton, 4M | -3.485 ± 1.530 | Success 3.8% | 14.978 | 49.350 |
| Ant, 2M | 5635.3 ± 1203.4 | Failure 15.4% | 0.154 | 0.530 |
| Walker2d, 2M | 4741.2 ± 1360.5 | Failure 37.6% | 0.376 | 0.810 |

Statistics use five **training seeds** within each environment, not 500
independent training runs. CVaR entries average the five per-training-seed
CVaR-20 values; evaluation episodes are not pooled across seeds. Locomotion
failure rate must not be confused with Point success rate. Different tasks'
reward/cost scales are not directly comparable.

| Environment | Retained-lineage training cost at 40%, mean | High-risk V_N/V_T, mean | High-risk dangerous outward velocity, mean |
|---|---:|---:|---:|
| PointGoal | 69,984.8 | 0.9484 | 0.1572 |
| PointButton | 94,537.0 | 0.9781 | 0.0508 |
| Ant | 573.4 | 0.8163 | 0.2898 |
| Walker2d | 2,721.8 | 0.7430 | 0.2359 |

The Point task success rates are low at this stage. Point normal/tangent
variance ratios remain close to one. Mean PointButton cost exceeds its budget
of ten, and neither locomotion environment meets its zero-cost constraint.
PointGoal's average cost is below ten, but individual seeds range from 6.72 to
12.28 and tail costs remain substantial. These observations do not establish
improvement over C0 or causal benefit from a particular mechanism: there is no
matched-control comparison here, and no significance test was performed.

The [descriptive JSON summary](shapo_p47_four_env_artifacts/partial_milestones/four_env_40pct_all20_20261022/summary.json)
contains all twenty per-seed records, sample SDs, diagnostic ranges, and raw-file
hashes; [the per-seed CSV](shapo_p47_four_env_artifacts/partial_milestones/four_env_40pct_all20_20261022/per_training_seed_metrics.csv)
provides the same tabular metrics. Original native JSON files accompany them.
Training-cost snapshots reflect accepted lineages, not all physical discarded
work, and must not be retroactively adjusted using later recovery tails.

Known historical 2027 original-full-parent checkpoint gaps remain unresolved;
matching metrics JSON hashes does not repair checkpoint ancestry. This
mid-training assembly does not certify full provenance or replace the strict
twenty-cell final audit. All 10M/5M training budgets remain unchanged, with no
new configuration selection or scientific early stopping.

## Earlier operational summary: 2026-10-22 13:52 CST

- Six of twenty cells have terminal training receipts; full scientific finalization is pending.
- Seven distinct cells are active: six on 4080 and Ant-2029 on A100.
- Seven 5080 cells remain reserved until 19:11; the persistent resume timer is active.
- Conditional remaining training estimate: four to six days, assuming tonight's resume and similar shared-resource availability.
- Earlier PointButton 2028/2030 milestone files have been restored from retained same-checkpoint histories, without overwriting active files or rerunning evaluations.

## Baseline operational update: 2026-10-22 11:11 CST

This update supersedes the historical observations below. Counts refer to actual
Python training workers, not shell wrappers. Training completion is not a claim
that the safety constraint or the full twenty-cell experiment has passed.

| Host | Actual workers | Latest observation |
|---|---:|---|
| A100 | 6 | PointButton 2028/2030: 4.425M/6.675M of 10M; Ant 2028/2029: 2.850M/2.900M of 5M; Walker2d 2028/2029: 2.950M/2.955M of 5M |
| 4080 | 4 | PointButton 2029: 4.830M of 10M; Walker2d 2028/2029: 3.055M/3.045M of 5M; Ant 2028: 2.945M of 5M |
| 5080 | 0 | User reservation: no SAFEFLOW training until 2026-10-22 19:11 CST |

Three A100 receipts now show `status=passed`, `training_summary.status=completed`,
and 5,000,000 environment steps: [Ant 2030](p4_7_training_receipts_20261022/ant-2030/receipt.json),
[Walker2d 2030](p4_7_training_receipts_20261022/walker2d-2030/receipt.json), and
[Walker2d 2031](p4_7_training_receipts_20261022/walker2d-2031/receipt.json).
These are engineering completion receipts; their provisional checkpoint
selections do **not** satisfy the configured zero-cost constraint.

At inspection, all seven former 5080 workers were already absent. MPS logs show
their disconnections at approximately 09:17; no cause is established here, and
this inspection did not terminate them. Their latest full replay checkpoints
were successfully loaded on CPU and retained unchanged:

| Lane | Durable checkpoint steps | Last logged steps |
|---|---:|---:|
| PointGoal 2027 | 8,250,000 | 8,270,000 |
| PointGoal 2028 | 6,050,000 | 6,055,000 |
| PointGoal 2029 | 5,250,000 | 5,285,000 |
| PointGoal 2030 | 6,900,000 | 6,910,000 |
| PointGoal 2031 | 7,450,000 | 7,470,000 |
| PointButton 2027 | 8,000,000 | 8,030,000 |
| PointButton 2031 | 7,400,000 | 7,415,000 |

The enabled host-side timer `safeflow-5080-resume-20261022-persistent.timer` is
active for 19:11 CST. It is stored in the host's user-service configuration,
not just a transient session; a missed trigger is caught up when the host returns.
The remote reservation begins at 11:10:21 CST and refuses launches before eight
hours elapse. The timer invokes a seven-lane-only, duplicate-aware resume script;
SSH or memory-headroom failures retry every 30 minutes. It never terminates
other projects. Recovery resumes from durable checkpoints, not the later CSV
counters; the unsaved tails will be replayed. This schedule requires the host
and network to be available. A100 and 4080 training were not changed.

Ant 2028 and Walker2d 2028/2029 currently have continuation branches on both A100
and 4080. They are duplicate branches of the same matrix cells, **not** extra
independent seeds. No results should count both branches as separate runs; no
branch was stopped or selected in this update.

A100 `/home` has approximately 25 GiB free; 4080's archive filesystem has about
2.6 TiB free. Existing half-hour monitoring and checkpoint backups remain active.
No new scientific evaluations, training-budget reductions, or source changes
were introduced for this reservation.

## Duplicate-branch retirement: 2026-10-22 11:28 CST

The three A100 duplicate branches described above have now been retired. The
4080 continuations are the selected branches, on an operational basis alone:
their code commit, resolved-config hashes, seeds, and initial resume-checkpoint
hashes match the A100 duplicates, while their durable checkpoints are ahead
and observed throughput is higher. Return and safety scores were not used to
select a branch. The [selection record](p4_7_training_receipts_20261022/duplicate_branch_selection.json)
preserves both branches' checkpoint counters, including cumulative costs.

| Matrix cell | Retired A100 checkpoint | Selected 4080 checkpoint |
|---|---:|---:|
| Ant 2028 | 2.850M | 2.950M |
| Walker2d 2028 | 2.950M | 3.050M |
| Walker2d 2029 | 2.950M | 3.050M |

Only the three identity-checked A100 training PIDs received SIGTERM through the
existing CUDA-synchronizing launcher. They exited without force-killing. Their
complete containers were renamed, without deleting files, into
`/home/lzqw/SAFEFLOW2027_A100/recovery_backups/retired_duplicate_branches_20261022/`.
These are excluded alternatives, not three completed matrix cells. No training
budget or scientific configuration was changed, and no other projects were
stopped. The existing result collector can import the selected 4080 terminal
containers into the now-vacant A100 matrix slots without overwriting a branch.

A100 now retains three distinct active tasks (PointButton 2028/2030 and Ant
2029), while 4080 retains four. The seven 5080 Point lanes remain reserved until
19:11. Together with the collector's six terminal receipts this accounts for
the twenty requested cells: six trained, seven actively training, seven held.
The final strict scientific audit is still pending. A new stable observation
interval is needed before claiming a speedup from removing redundant work.

### Retired-branch backup: 2026-10-22 11:40 CST

All three complete retired containers (approximately 704 MiB) were copied to
4080's separate 4T filesystem, at
`/home/carla/4T/SAFEFLOW2027_4080/recovery_backups/retired_duplicate_branches_20261022_a100/`.
The A100 originals remain intact; nothing was deleted. The transfer completed
successfully. A single post-transfer integrity comparison of the sorted relative
regular-file paths and their SHA-256 values matched on both hosts. The aggregate
manifest SHA-256 is
`12b2884ccf0d09e7853d8c2cd9ac4838cb1215754dbc51fe9ebdde38802f6702`.
This was an archive integrity check, not another scientific evaluation.

## PointButton-2028 acceleration handoff: 2026-10-22 12:14 CST

The longest-remaining A100 PointButton cell was migrated to 4080 using its full
4.400M-step checkpoint, with unchanged code, configuration, seed, optimizer/RNG/
replay state, and 10M-step budget. Its first new epoch reached 4.405M steps at
21.71 FPS, versus approximately 5.55 logged steps/second over the preceding A100
half-hour. This is an encouraging startup observation, not yet a long-term
throughput guarantee or an end-to-end completion estimate.

The full source container was held during transfer and its checkpoint SHA-256
matched the target. Only after the new target epoch and MPS connection were
observed did the A100 source exit through its CUDA-synchronizing launcher.
The [completed handoff record](p4_7_training_receipts_20261022/point_button_2028_a100_to_4080.json)
preserves the exact locations and counters. The source container and the
previous, stale 4080 container were retained under their respective
`recovery_backups/retired_gpu_workers/` directories; no files were deleted.

The source had logged 4.445M steps and cumulative cost 92,862, while the full
checkpoint held 4.400M steps and cost 92,178. This means at least 45,000 repeated
physical steps and 684 extra training-cost units, plus an unknown unlogged tail.
These must be included in final physical-work accounting, not hidden by the
accepted checkpoint lineage. Bitwise equivalence across GPU/runtime versions is
not claimed. A100 now has two unique active tasks and 4080 has five; the seven
5080 tasks remain reserved until 19:11.

The earlier three excluded duplicate branches also incurred measured work.
Their [separate accounting record](p4_7_training_receipts_20261022/duplicate_branch_accounting.json)
shows at least 1,075,000 additional physical steps and 823 training-cost units
after the shared resume checkpoints. Combined with this handoff, these newly
recorded operations add at least 1,120,000 physical steps and 1,507 cost units;
unknown tails remain unknown, not zero. Integration into the final consolidated
recovery ledger is still required before a complete-publication claim.

## Second PointButton handoff and accounting integration: 2026-10-22 12:38 CST

The scheduled 12:22 snapshot showed all five 4080 workers at approximately
21–23 FPS, with GPU utilization 72% and about 12 GiB free VRAM. On that basis,
PointButton-2030 was also moved from A100, using its 6.700M full checkpoint.
The target's first new epoch reached 6.705M at 20.82 FPS. Its A100 source was
retired only after this new epoch and verified MPS execution. The
[completed handoff record](p4_7_training_receipts_20261022/point_button_2030_a100_to_4080.json)
retains the source/target identities, original data locations, and unchanged
10M-step budget. The last logged source counter equals the checkpoint counter,
so the *known logged* repeated tail is zero; this does not establish a zero
unlogged tail.

A100 now runs Ant-2029; 4080 runs six distinct cells. Seven 5080 cells remain
held until 19:11. Together with six terminal receipts this still accounts for
exactly twenty cells, with no duplicate active continuation branches.

The final publisher now incorporates the three excluded duplicate branches and
both new A100-to-4080 handoffs into its existing consolidated recovery ledger,
and includes their original records in the report's checksums. It rejects an
unfinished handoff rather than silently omitting it. One engineering-only
accounting read confirmed five records totaling the same known lower bound:
1,120,000 additional physical steps and 1,507 training-cost units. No raw metric,
scientific audit gate, configuration, or evaluation budget was modified. The
full scientific finalization still waits for all twenty cells; the later 5080
stop/resume tails also require their own explicit accounting.

## Six-worker observation and pre-existing stop tails: 2026-10-22 12:52 CST

The regular half-hour snapshot observed six distinct 4080 workers, with last
logged epoch rates around 21–23 FPS and GPU utilization 73% (about 12 GiB free
VRAM). A100 Ant-2029 remained live at 2.940M / 5M, with about 5.55 logged steps
per second over the previous interval. No SAFEFLOW training worker was present
on 5080, and its reservation timer remains active for 19:11.

| 4080 cell | Logged steps | Previous-window logged steps/s |
|---|---:|---:|
| PointButton 2029 | 4.955M / 10M | 19.44 |
| Walker2d 2028 | 3.190M / 5M | 22.21 |
| Ant 2028 | 3.075M / 5M | 22.21 |
| Walker2d 2029 | 3.175M / 5M | 22.21 |
| PointButton 2028 | 4.455M / 10M | 22.21 |
| PointButton 2030 | 6.725M / 10M | New worker; no full matching interval yet |

At unchanged rates, the active 4080 work is roughly a three-to-four-day tail,
while the A100 Ant run alone is approximately 4.3 days. A provisional overall
training estimate is **four to six days**, conditional on the seven 5080 jobs
resuming tonight and shared-resource contention remaining near prior levels.
This is not a promise or a final-scientific-audit completion estimate; rates and
remaining time will be updated using the existing half-hour monitor.

Reading only the already-stopped 5080 logs recovered another known lower bound
of 135,000 discarded physical steps and 1,566 cost units across seven lanes.
These stops predated the user's reservation; idle reservation time is not
training, and the stop cause remains unestablished. The
[historical reservation/accounting snapshot](p4_7_training_receipts_20261022/hold_5080_8h.json)
retains both full-checkpoint and last-logged counters. This metadata collection
did not launch a learner or change the 19:11 resume time.

The final publisher now includes all twelve newly documented accounting rows,
totaling at least 1,255,000 additional physical steps and 3,073 cost units.
Unknown unlogged tails remain unknown, not zero. These are recovery/placement
records, not additional experiment cells; the training matrix still has six
terminal cells, seven active cells, and seven reserved cells. The full scientific
audit is still deferred until all twenty cells finish.

## Earlier-segment preservation repair: 2026-10-22 13:51 CST

Inspection of the minimal A100-to-4080 handoff packages found that earlier
PointButton 2028/2030 evaluation files at 1M and 4M were not in their active
containers. The original files were still present in the retained 4080
containers. For each cell, the exact shared resume checkpoint in the active and
retained histories had identical SHA-256 bytes, establishing the retained
history as the same checkpoint ancestor rather than another seed or candidate.

Only missing training-history files were restored with ignore-existing copy
semantics: 80 files for PointButton 2028 and 75 for PointButton 2030, totaling
3,725,870,066 bytes. Their SHA-256 values matched the retained originals. Active
files and their shared checkpoints were not overwritten, and the retained
originals remain intact. The
[restoration manifest](p4_7_training_receipts_20261022/historical_segment_restoration.json)
records every added relative path, size, checksum, and ancestor identity.

The restored milestone resume-path metadata refers to same-cell 4080 training
parents (or no resume), compatible with the existing audited relocation
resolver when terminal containers are collected back to A100. No validator
rule or scientific gate was weakened, no training process was restarted, and
no scientific evaluation was rerun. The final publisher now includes this
preservation record in the consolidated ledger and report checksums. Physical
work accounting is unchanged: the twelve newly documented recovery rows still
total at least 1,255,000 additional steps and 3,073 cost units.

The regular 13:52 snapshot still observed A100 Ant-2029 advancing to 2.960M,
and six 4080 cells advancing normally: PointButton 2029/2028/2030 at
5.030M/4.530M/6.800M, Walker2d 2028/2029 at 3.275M/3.260M, and Ant 2028 at
3.155M. No 5080 training worker was present. Full final validation is still
deferred until all twenty cells finish.

## Ant-2029 historical package repair

The active A100 Ant-2029 container similarly lacked its 500k/2M milestone
history and source-version marker. The retained 4080 container held the original
files. Its immutable 2.5M shared resume checkpoint matched A100 byte-for-byte
(SHA-256 `ad81d3c7eccdca5bd4b0566405d85f886e05957a36af4f76e1cc613a4a492cab`),
and the resolved formal configuration hashes also matched.

Fifty-eight missing files, totaling 344,539,866 bytes, were restored without
overwriting an existing training file. Restored bytes matched their originals;
the source and complete transferred input package remain retained. Historical
5080 artifact paths are explicitly recorded as transferred through 4080 to A100,
so the existing audited relocation resolver can locate the early milestone
parents without editing original checkpoint metadata or weakening its rules.
The [preservation receipt](p4_7_training_receipts_20261022/ant2029_historical_restoration.json)
records every restored path and checksum.

No signal was sent to the live Ant trainer, no process was restarted, and no
scientific evaluation was rerun. The final publisher includes this receipt in
its recovery ledger and checksums; physical-work totals are unchanged. Six
4080 workers and one A100 worker continue training, while the 5080 reservation
still ends at 19:11. The twenty-cell final scientific audit remains pending.

## 5080 local backup resume timer

A second, 5080-local user timer is now enabled for the same 19:11 CST deadline,
alongside the existing host-side SSH timer. Both call the same reservation
dispatcher and share its exclusive file lock, per-lane duplicate checks, and
eight-hour not-before guard. This reduces dependence on the host's network
availability; it does not advance the resume time or terminate another project.

The [local timer](p4_7_training_receipts_20261022/runtime_schedule/safeflow-5080-local-resume-20261022.timer)
was observed active with the correct next deadline; its
[dispatcher service](p4_7_training_receipts_20261022/runtime_schedule/safeflow-5080-local-resume-20261022.service)
was still inactive, and the reservation state remained `held`. No resume command
was invoked for this installation. The service deliberately retains its cgroup
after dispatch so successful detached trainers are not killed when the short
dispatcher exits. Memory/SSH failures retain the existing thirty-minute retry
policy and do not interrupt other workloads.

The local timer requires the 5080 user manager to be available at trigger time;
no user-linger or unrelated account setting was changed. The host timer remains
an independent fallback. Training code, budgets, and scientific gates are
unchanged, and the pending request to change only multi-hop artifact lookup is
not treated as approved.

## Fixed preservation of the seven 19:11 resume inputs

The ordinary recovery backup intentionally rotates a limited number of recent
versions. To prevent that rotation from eventually dropping the exact input
files named by the 19:11 restart, all seven were pinned in a separate fixed
archive on the 4080 4T filesystem:
`/home/carla/4T/SAFEFLOW2027_4080/runtime_resume_inputs_20261022/5080/`.

The [input preservation manifest](p4_7_training_receipts_20261022/resume_inputs_5080_manifest.json)
records each original input path, its saved counters, size, SHA-256 and fixed
archive path. Total retained payload is **1,805,765,534 bytes**. Every archived
file matched its named 5080 original. Existing same-filesystem backup inodes
were hard-linked before rolling replacement, so no large checkpoint payload
was staged on the host and no extra network payload transfer was needed.
Rolling backups use atomic replacement, leaving these retained input inodes
intact. Originals were neither moved nor deleted.

SHA values were captured after startup; this is preservation of the named
current inputs, not a retroactive content-at-start attestation. No earlier
missing 2027 parent was aliased to a later checkpoint or declared recovered.
No model/tensor was loaded, learner signalled, evaluation run, or training
configuration changed. The manifest is now included in final publication
metadata and checksums; physical-work accounting is unchanged.

## Historical observations

This is an operational snapshot of the ongoing P4.7 training lanes. It is not a completion claim; all listed jobs remain within their configured training budgets.

Snapshot time: **2026-10-22 02:02 CST**

| Host | Active lane | Latest verified progress | GPU utilization |
|---|---|---:|---:|
| A100 | PointButton seed 2028 | 4.230M / 10M steps (FPS about 5.71) | 50% |
| A100 | PointButton seed 2030 | 6.480M / 10M steps (FPS about 5.76) | 51% |
| 4080 | PointButton seed 2029 | 4.065M / 10M steps (FPS about 23.52) | 31% |
| 5080 | 7 active Point lanes | Running; latest aggregate GPU observation | 76% |

All observed training processes were alive at the snapshot. The A100 recovery history is maintained in [P4.7 A100 recovery status](P4_7_A100_RECOVERY_STATUS_20261021.md).

At the **2026-10-22 02:07 CST** follow-up, all three hosts remained reachable with the same active-lane counts. A100 remained at 4.230M/6.480M for the two tracked PointButton jobs (GPU utilization about 40%/43%); 4080 advanced to **4.070M / 10M steps** (FPS about 24.56, GPU utilization about 27%); 5080 remained at seven active lanes (GPU utilization about 82%).

At the **2026-10-22 02:08 CST** follow-up, A100 PointButton 2028 advanced to **4.235M / 10M steps** (FPS about 6.31), while PointButton 2030 remained at 6.480M; A100 GPU utilization was about 50%/52%. The 4080 lane remained at 4.070M and the 5080 lane count remained seven, with GPU utilization about 29% and 82%, respectively.

At the **2026-10-22 02:28 CST** follow-up, A100 PointButton 2028/2030 advanced to **4.240M/6.490M / 10M steps** (FPS about 6.32/6.27), with GPU utilization about 44%/51%. The 4080 PointButton 2029 lane advanced to **4.100M / 10M steps** (FPS about 24.56, GPU utilization about 28%). The 5080 retained seven active lanes with GPU utilization about 79%; all observed training processes remained alive.

At the **2026-10-22 02:30 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.105M / 10M steps** (FPS about 24.43, GPU utilization about 31%). A100 remained at 4.240M/6.490M for the two tracked lanes (GPU utilization about 35%/41%), while 5080 retained seven active lanes (GPU utilization about 85%); all observed training processes remained alive.

At the **2026-10-22 02:32 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.110M / 10M steps** (FPS about 24.80, GPU utilization about 31%). A100 remained at 4.240M/6.490M (GPU utilization about 36%/41%), and 5080 retained seven active lanes (GPU utilization about 80%); all observed training processes remained alive.

At the **2026-10-22 02:36 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.115M / 10M steps** (FPS about 25.12, GPU utilization about 29%). A100 remained at 4.240M/6.490M (GPU utilization about 36%/42%), and 5080 retained seven active lanes (GPU utilization about 81%); all observed training processes remained alive.
At the **2026-10-22 02:56 CST** follow-up, A100 PointButton 2028/2030 advanced to **4.250M/6.500M / 10M steps** (FPS about 5.73/5.79, GPU utilization about 45%/52%). The 4080 PointButton 2029 lane advanced to **4.140M / 10M steps** (FPS about 24.87, GPU utilization about 29%). The 5080 retained seven active lanes and its GPU utilization was about 86%; all observed training processes remained alive.

At the **2026-10-22 02:57 CST** follow-up, A100 remained at **4.250M/6.500M / 10M steps** (GPU utilization about 46%/50%). The 4080 PointButton 2029 lane advanced to **4.145M / 10M steps** (FPS about 24.32, GPU utilization about 33%). The 5080 retained seven active lanes with GPU utilization about 83%; all observed training processes remained alive.

At the **2026-10-22 03:00 CST** follow-up, A100 remained at **4.250M/6.500M / 10M steps** (GPU utilization about 36%/42%). The 4080 PointButton 2029 lane advanced to **4.150M / 10M steps** (FPS about 24.71, GPU utilization about 31%). The 5080 retained seven active lanes with GPU utilization about 80%; all observed training processes remained alive.

At the **2026-10-22 03:01 CST** follow-up, A100 remained at **4.250M/6.500M / 10M steps** (GPU utilization about 35%/42%) and 4080 remained at **4.150M / 10M steps** (GPU utilization about 33%). The 5080 retained seven active lanes and advanced its latest observed lanes to approximately **7.020M, 7.590M, 7.820M, 6.970M, 5.610M, 6.460M, and 4.840M / 10M steps**, with GPU utilization about 86%; all observed training processes remained alive.

At the **2026-10-22 03:02 CST** follow-up, the 5080 retained seven active lanes and advanced three observed lanes to approximately **5.615M, 6.465M, and 4.845M / 10M steps**; the other four remained at approximately 7.020M, 7.590M, 7.820M, and 6.970M. GPU utilization was about 79%, and all observed training processes remained alive.

At the **2026-10-22 03:03 CST** follow-up, the 5080 PointButton 2031 lane advanced to approximately **6.975M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 80%, and all seven training processes remained alive.

At the **2026-10-22 03:03 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.155M / 10M steps** (FPS about 24.16, GPU utilization about 32%). A100 remained at **4.250M/6.500M / 10M steps** (GPU utilization about 36%/42%), while 5080 retained seven active lanes with GPU utilization about 82%; all observed training processes remained alive.

At the **2026-10-22 03:04 CST** follow-up, the 5080 PointGoal lanes advanced to approximately **7.595M and 7.825M / 10M steps**; the other five observed lanes remained at approximately 6.975M, 5.615M, 6.465M, 4.845M, and 7.020M. GPU utilization was about 83%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 03:05 CST** follow-up, a further 5080 lane advanced to approximately **7.025M / 10M steps**; the other six observed lanes remained unchanged. The 5080 retained seven active processes with GPU utilization about 78%; A100 and 4080 remained active as well.

At the **2026-10-22 03:07 CST** follow-up, A100 PointButton 2028 advanced to **4.255M / 10M steps** while PointButton 2030 remained at **6.500M / 10M steps** (GPU utilization about 35%/44%). A 5080 lane also advanced to approximately **4.850M / 10M steps**; its other six observed lanes remained at approximately 7.025M, 7.595M, 7.825M, 6.975M, 5.615M, and 6.465M, with GPU utilization about 81%. The 4080 remained at 4.155M; all observed training processes remained alive.

At the **2026-10-22 03:07:35 CST** follow-up, A100 PointButton 2030 advanced to **6.505M / 10M steps** (PointButton 2028 remained at 4.255M; GPU utilization about 33%/43%). The 4080 PointButton 2029 lane advanced to **4.160M / 10M steps** (FPS about 24.81, GPU utilization about 25%). On 5080, three lanes advanced to approximately **6.980M, 5.620M, and 6.470M / 10M steps**; the other four were approximately 4.850M, 7.025M, 7.595M, and 7.825M. GPU utilization was about 83%; all observed training processes remained alive.

At the **2026-10-22 03:08:46 CST** follow-up, a 5080 lane advanced to approximately **7.830M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:09 CST** follow-up, two 5080 lanes advanced to approximately **7.030M and 7.600M / 10M steps**; the remaining observed lanes were approximately 7.830M, 6.980M, 5.620M, 6.470M, and 4.850M. GPU utilization was about 80%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 03:10 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.165M / 10M steps** (FPS about 24.07, GPU utilization about 30%). A 5080 lane advanced to approximately **4.855M / 10M steps**; its other six observed lanes remained at approximately 7.030M, 7.600M, 7.830M, 6.980M, 5.620M, and 6.470M. GPU utilization was about 83%, and all observed training processes remained alive.

At the **2026-10-22 03:10:42 CST** follow-up, a 5080 lane advanced to approximately **6.475M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:11 CST** follow-up, two 5080 lanes advanced to approximately **6.985M and 5.625M / 10M steps**; the other five observed lanes remained at approximately 6.475M, 4.855M, 7.030M, 7.600M, and 7.830M. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 03:12:54 CST** follow-up, a 5080 lane advanced to approximately **7.835M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 80%, and all seven training processes remained alive.

At the **2026-10-22 03:13 CST** follow-up, two 5080 lanes advanced to approximately **7.035M and 7.605M / 10M steps**; the other five observed lanes remained at approximately 7.835M, 6.985M, 5.625M, 6.475M, and 4.855M. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:14 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.170M / 10M steps** (FPS about 24.40, GPU utilization about 31%). A100 remained active at 4.255M/6.505M, and 5080 retained seven active lanes with GPU utilization about 81%; all observed training processes remained alive.

At the **2026-10-22 03:14:34 CST** follow-up, two 5080 lanes advanced to approximately **6.480M and 4.860M / 10M steps**; the other five observed lanes remained at approximately 7.035M, 7.605M, 7.835M, 6.985M, and 5.625M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:15 CST** follow-up, two 5080 lanes advanced to approximately **5.630M and 6.480M / 10M steps**; the other five observed lanes remained at approximately 7.035M, 7.605M, 7.835M, 6.985M, and 4.860M. GPU utilization was about 79%, and all seven training processes remained alive.

At the **2026-10-22 03:15:46 CST** follow-up, two 5080 lanes advanced to approximately **6.990M and 5.630M / 10M steps**; the other five observed lanes remained at approximately 6.480M, 4.860M, 7.035M, 7.605M, and 7.835M. GPU utilization was about 80%, and all seven training processes remained alive.

At the **2026-10-22 03:16:55 CST** follow-up, a 5080 lane advanced to approximately **7.840M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 86%, and all seven training processes remained alive.

At the **2026-10-22 03:17 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.175M / 10M steps** (FPS about 24.28, GPU utilization about 27%). Two 5080 lanes advanced to approximately **7.040M and 7.610M / 10M steps**; its other five observed lanes remained at approximately 7.840M, 6.990M, 5.630M, 6.480M, and 4.860M. GPU utilization was about 84%, and all observed training processes remained alive.

At the **2026-10-22 03:18:43 CST** follow-up, two 5080 lanes advanced to approximately **6.485M and 4.865M / 10M steps**; the other five observed lanes remained at approximately 7.040M, 7.610M, 7.840M, 6.990M, and 5.630M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:19 CST** follow-up, a 5080 lane advanced to approximately **5.635M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:19:52 CST** follow-up, A100 PointButton 2028 advanced to **4.260M / 10M steps** while PointButton 2030 remained at **6.505M / 10M steps** (GPU utilization about 41%/49%). A 5080 lane advanced to approximately **6.995M / 10M steps**; its other six observed lanes remained at approximately 5.635M, 6.485M, 4.865M, 7.040M, 7.610M, and 7.840M. GPU utilization was about 84%; all observed training processes remained alive.

At the **2026-10-22 03:20 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.180M / 10M steps** (FPS about 24.75, GPU utilization about 34%). A100 remained active at 4.260M/6.505M, and 5080 retained seven active lanes with GPU utilization about 87%; all observed training processes remained alive.

At the **2026-10-22 03:21:06 CST** follow-up, A100 PointButton 2030 advanced to **6.510M / 10M steps** while PointButton 2028 remained at **4.260M / 10M steps** (GPU utilization about 36%/42%). A 5080 lane advanced to approximately **7.845M / 10M steps**; its other six observed lanes remained at approximately 6.995M, 5.635M, 6.485M, 4.865M, 7.040M, and 7.610M. GPU utilization was about 83%; all observed training processes remained alive.

At the **2026-10-22 03:21:42 CST** follow-up, two 5080 lanes advanced to approximately **7.045M and 7.615M / 10M steps**; the other five observed lanes remained at approximately 7.845M, 6.995M, 5.635M, 6.485M, and 4.865M. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:22:42 CST** follow-up, two 5080 lanes advanced to approximately **4.870M and 6.490M / 10M steps**; the other five observed lanes remained at approximately 7.045M, 7.615M, 7.845M, 6.995M, and 5.635M. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:23:40 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.185M / 10M steps** (FPS about 24.80, GPU utilization about 29%). Three 5080 lanes advanced to approximately **7.000M, 5.640M, and 4.870M / 10M steps**; the other four remained at approximately 6.490M, 7.045M, 7.615M, and 7.845M. GPU utilization was about 85%, and all observed training processes remained alive.

At the **2026-10-22 03:25:19 CST** follow-up, five 5080 lanes advanced to approximately **7.850M, 7.000M, 5.640M, 4.870M, and 6.490M / 10M steps**; the other two remained at approximately 7.045M and 7.615M. GPU utilization was about 79%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 03:25:57 CST** follow-up, two 5080 lanes advanced to approximately **7.050M and 7.620M / 10M steps**; the other five observed lanes remained at approximately 7.850M, 7.000M, 5.640M, 4.870M, and 6.490M. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:26:54 CST** follow-up, two 5080 lanes advanced to approximately **4.875M and 6.495M / 10M steps**; the other five observed lanes remained at approximately 7.050M, 7.620M, 7.850M, 7.000M, and 5.640M. GPU utilization was about 80%, and all seven training processes remained alive.

At the **2026-10-22 03:27:30 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.190M / 10M steps** (FPS about 24.82, GPU utilization about 31%). A100 remained active at 4.260M/6.510M, and 5080 retained seven active lanes with GPU utilization about 79%; all observed training processes remained alive.

At the **2026-10-22 03:28:11 CST** follow-up, three 5080 lanes advanced to approximately **7.005M, 5.645M, and 4.875M / 10M steps**; the other four observed lanes remained at approximately 6.495M, 7.050M, 7.620M, and 7.850M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:30:11 CST** follow-up, a 5080 lane advanced to approximately **7.855M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 86%, and all seven training processes remained alive.

At the **2026-10-22 03:30:46 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.195M / 10M steps** (FPS about 24.93, GPU utilization about 34%). Two 5080 lanes advanced to approximately **7.055M and 7.625M / 10M steps**; the other five remained at approximately 7.855M, 7.005M, 5.645M, 4.875M, and 6.495M. GPU utilization was about 82%, and all observed training processes remained alive.

At the **2026-10-22 03:31:57 CST** follow-up, two 5080 lanes advanced to approximately **5.650M and 4.880M / 10M steps**; the other five observed lanes remained at approximately 7.055M, 7.625M, 7.855M, 7.005M, and 6.500M. GPU utilization was about 79%, and all seven training processes remained alive.

At the **2026-10-22 03:32:32 CST** follow-up, three 5080 lanes advanced to approximately **7.010M, 5.650M, and 4.880M / 10M steps**; the other four observed lanes remained at approximately 6.500M, 7.625M, 7.855M, and 7.055M. GPU utilization was about 79%, and all seven training processes remained alive.

At the **2026-10-22 03:33:36 CST** follow-up, a 5080 lane advanced to approximately **7.860M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:34:12 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.200M / 10M steps** (FPS about 24.97, GPU utilization about 35%). A 5080 lane advanced to approximately **7.060M / 10M steps**; its other six observed lanes remained at approximately 7.860M, 7.005M, 5.650M, 4.880M, 6.500M, and 7.625M. GPU utilization was about 86%, and all observed training processes remained alive.

At the **2026-10-22 03:34:48 CST** follow-up, A100 PointButton 2028 advanced to **4.265M / 10M steps** while PointButton 2030 remained at **6.510M / 10M steps** (GPU utilization about 37%/44%). A 5080 lane advanced to approximately **7.630M / 10M steps**; its other six observed lanes remained at approximately 7.060M, 7.860M, 7.005M, 5.650M, 4.880M, and 6.500M. GPU utilization was about 84%; all observed training processes remained alive.

At the **2026-10-22 03:35:28 CST** follow-up, A100 PointButton 2030 advanced to **6.515M / 10M steps** while PointButton 2028 remained at **4.265M / 10M steps** (GPU utilization about 35%/41%). Three 5080 lanes advanced to approximately **4.885M, 6.505M, and 7.630M / 10M steps**; the other four remained at approximately 7.060M, 7.860M, 7.005M, and 5.650M. GPU utilization was about 87%; all observed training processes remained alive.

At the **2026-10-22 03:36:07 CST** follow-up, three 5080 lanes advanced to approximately **5.655M, 4.885M, and 6.505M / 10M steps**; the other four observed lanes remained at approximately 7.060M, 7.860M, 7.005M, and 7.630M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:36:41 CST** follow-up, a 5080 lane advanced to approximately **7.015M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:37:18 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.205M / 10M steps** (FPS about 24.87, GPU utilization about 29%). A100 remained active at 4.265M/6.515M, and 5080 retained seven active lanes with GPU utilization about 83%; all observed training processes remained alive.

At the **2026-10-22 03:37:49 CST** follow-up, a 5080 lane advanced to approximately **7.865M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 03:38:21 CST** follow-up, a 5080 lane advanced to approximately **7.065M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 86%, and all seven training processes remained alive.

At the **2026-10-22 03:38:51 CST** follow-up, a 5080 lane advanced to approximately **7.635M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:39:26 CST** follow-up, two 5080 lanes advanced to approximately **6.510M and 7.635M / 10M steps**; the other five observed lanes remained at approximately 7.065M, 7.865M, 7.015M, 5.655M, and 4.885M. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:40:02 CST** follow-up, a 5080 lane advanced to approximately **4.890M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 79%, and all seven training processes remained alive.

At the **2026-10-22 03:40:37 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.210M / 10M steps** (FPS about 24.44, GPU utilization about 29%). Three 5080 lanes advanced to approximately **7.020M, 5.660M, and 6.510M / 10M steps**; the other four remained at approximately 7.635M, 7.065M, 7.865M, and 4.890M. GPU utilization was about 86%, and all observed training processes remained alive.

At the **2026-10-22 03:41:11 CST** follow-up, two 5080 lanes advanced to approximately **7.020M and 5.660M / 10M steps**; the other five observed lanes remained at approximately 6.510M, 7.635M, 7.065M, 7.865M, and 4.890M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:42:09 CST** follow-up, a 5080 lane advanced to approximately **7.870M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:42:51 CST** follow-up, three 5080 lanes advanced to approximately **7.070M, 7.870M, and 7.640M / 10M steps**; the other four observed lanes remained at approximately 6.510M, 5.660M, 4.890M, and 7.065M. GPU utilization was about 86%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 03:43:25 CST** follow-up, two 5080 lanes advanced to approximately **7.070M and 7.640M / 10M steps**; the other five observed lanes remained at approximately 6.510M, 7.870M, 5.660M, 4.890M, and 7.065M. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 03:43:59 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.215M / 10M steps** (FPS about 24.24, GPU utilization about 30%). Three 5080 lanes advanced to approximately **4.895M, 6.515M, and 7.640M / 10M steps**; the other four remained at approximately 7.070M, 7.870M, 5.660M, and 7.065M. GPU utilization was about 83%, and all observed training processes remained alive.

At the **2026-10-22 03:44:33 CST** follow-up, a 5080 lane advanced to approximately **5.665M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:45:06 CST** follow-up, three 5080 lanes advanced to approximately **7.025M, 6.515M, and 7.640M / 10M steps**; the other four observed lanes remained at approximately 5.665M, 4.895M, 7.070M, and 7.870M. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 03:45:38 CST** follow-up, two 5080 lanes advanced to approximately **7.025M and 6.515M / 10M steps**; the other five observed lanes remained at approximately 7.640M, 5.665M, 4.895M, 7.070M, and 7.870M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:46:12 CST** follow-up, a 5080 lane advanced to approximately **7.875M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:46:44 CST** follow-up, two 5080 lanes advanced to approximately **7.075M and 7.875M / 10M steps**; the other five observed lanes remained at approximately 6.515M, 5.665M, 4.895M, 7.640M, and 7.870M. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:47:18 CST** follow-up, two 5080 lanes advanced to approximately **7.075M and 7.645M / 10M steps**; the other five observed lanes remained at approximately 6.515M, 7.875M, 5.665M, 4.895M, and 7.070M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:31:21 CST** follow-up, two 5080 lanes advanced to approximately **4.880M and 6.500M / 10M steps**; the other five observed lanes remained at approximately 7.055M, 7.625M, 7.855M, 7.005M, and 5.645M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:47:55 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.220M / 10M steps** (FPS about 24.32, GPU utilization about 35%). Three 5080 lanes advanced to approximately **6.520M, 7.645M, and 7.075M / 10M steps**; the other four remained at approximately 7.870M, 7.025M, 5.665M, and 4.895M. GPU utilization was about 85%, and all observed training processes remained alive.

At the **2026-10-22 03:48:52 CST** follow-up, A100 PointButton 2028 advanced to **4.270M / 10M steps** while PointButton 2030 remained at **6.515M / 10M steps** (GPU utilization about 35%/41%). Four 5080 lanes advanced to approximately **5.670M, 4.900M, 6.520M, and 7.875M / 10M steps**; the other three remained at approximately 7.075M, 7.645M, and 7.025M. GPU utilization was about 84%; all observed training processes remained alive.

At the **2026-10-22 03:49:31 CST** follow-up, A100 PointButton 2028 advanced to **4.275M / 10M steps** while PointButton 2030 remained at **6.515M / 10M steps** (GPU utilization about 34%/41%). Three 5080 lanes advanced to approximately **7.030M, 7.645M, and 7.875M / 10M steps**; the other four remained at approximately 6.520M, 5.670M, 4.900M, and 7.075M. GPU utilization was about 86%; all observed training processes remained alive.

At the **2026-10-22 03:50:10 CST** follow-up, A100 PointButton 2030 advanced to **6.520M / 10M steps** while PointButton 2028 remained at **4.275M / 10M steps** (GPU utilization about 37%/44%). Five 5080 lanes advanced to approximately **7.880M, 7.030M, 5.670M, 4.900M, and 6.520M / 10M steps**; the other two remained at approximately 7.645M and 7.075M. GPU utilization was about 83%; all observed training processes remained alive.

At the **2026-10-22 03:51:12 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.225M / 10M steps** (FPS about 24.60, GPU utilization about 30%). Three 5080 lanes advanced to approximately **7.080M, 7.880M, and 7.030M / 10M steps**; the other four remained at approximately 5.670M, 4.900M, 6.520M, and 7.645M. GPU utilization was about 83%, and all observed training processes remained alive.

At the **2026-10-22 03:51:54 CST** follow-up, two 5080 lanes advanced to approximately **7.080M and 7.650M / 10M steps**; the other five observed lanes remained at approximately 7.880M, 7.030M, 5.670M, 4.900M, and 6.520M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:52:32 CST** follow-up, three 5080 lanes advanced to approximately **4.905M, 6.525M, and 7.650M / 10M steps**; the other four observed lanes remained at approximately 7.880M, 7.030M, 5.670M, and 7.080M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:53:09 CST** follow-up, three 5080 lanes advanced to approximately **5.675M, 4.905M, and 6.525M / 10M steps**; the other four observed lanes remained at approximately 7.650M, 7.080M, 7.880M, and 7.030M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:53:42 CST** follow-up, a 5080 lane advanced to approximately **7.035M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:54:14 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.230M / 10M steps** (FPS about 24.84, GPU utilization about 31%). Two 5080 lanes advanced to approximately **7.035M and 6.525M / 10M steps**; the other five remained at approximately 7.650M, 7.080M, 7.880M, 5.675M, and 4.905M. GPU utilization was about 84%, and all observed training processes remained alive.

At the **2026-10-22 03:54:53 CST** follow-up, two 5080 lanes advanced to approximately **7.885M and 7.035M / 10M steps**; the other five observed lanes remained at approximately 6.525M, 7.650M, 7.080M, 5.675M, and 4.905M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:55:36 CST** follow-up, three 5080 lanes advanced to approximately **7.085M, 7.885M, and 7.035M / 10M steps**; the other four observed lanes remained at approximately 6.525M, 7.650M, 5.675M, and 4.905M. GPU utilization was about 76%, and all seven training processes remained alive.

At the **2026-10-22 03:56:08 CST** follow-up, four 5080 lanes advanced to approximately **6.530M, 7.655M, 7.085M, and 7.035M / 10M steps**; the other three observed lanes remained at approximately 7.885M, 5.675M, and 4.905M. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:56:43 CST** follow-up, five 5080 lanes advanced to approximately **4.910M, 7.655M, 7.085M, 7.035M, and 6.530M / 10M steps**; the other two remained at approximately 7.885M and 5.675M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:57:22 CST** follow-up, four 5080 lanes advanced to approximately **5.680M, 4.910M, 6.530M, and 7.655M / 10M steps**; the other three remained at approximately 7.085M, 7.885M, and 7.035M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:57:57 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.235M / 10M steps** (FPS about 24.63, GPU utilization about 34%). Two 5080 lanes advanced to approximately **7.040M and 7.655M / 10M steps**; the other five remained at approximately 5.680M, 6.530M, 7.085M, 7.885M, and 4.910M. GPU utilization was about 83%, and all observed training processes remained alive.

At the **2026-10-22 03:58:38 CST** follow-up, two 5080 lanes advanced to approximately **7.890M and 7.040M / 10M steps**; the other five remained at approximately 6.530M, 7.655M, 7.085M, 5.680M, and 4.910M. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 03:59:14 CST** follow-up, three 5080 lanes advanced to approximately **7.040M, 7.890M, and 7.040M / 10M steps**; the other four observed lanes remained at approximately 6.530M, 7.655M, 7.085M, and 5.680M. GPU utilization was about 86%, and all seven training processes remained alive.

At the **2026-10-22 03:59:56 CST** follow-up, four 5080 lanes advanced to approximately **7.660M, 7.090M, 7.890M, and 7.040M / 10M steps**; the other three remained at approximately 6.530M, 5.680M, and 4.910M. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 04:00:34 CST** follow-up, three 5080 lanes advanced to approximately **6.535M, 7.660M, and 7.090M / 10M steps**; the other four remained at approximately 7.890M, 7.040M, 5.680M, and 4.910M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 04:01:13 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.240M / 10M steps** (FPS about 24.84, GPU utilization about 35%). Three 5080 lanes advanced to approximately **4.915M, 6.535M, and 7.660M / 10M steps**; the other four remained at approximately 7.090M, 7.890M, 7.040M, and 5.680M. GPU utilization was about 82%, and all observed training processes remained alive.

At the **2026-10-22 04:01:57 CST** follow-up, four 5080 lanes advanced to approximately **7.045M, 5.685M, 4.915M, and 6.535M / 10M steps**; the other three remained at approximately 7.660M, 7.090M, and 7.890M. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 04:02:37 CST** follow-up, five 5080 lanes advanced to approximately **7.895M, 7.045M, 5.685M, 4.915M, and 6.535M / 10M steps**; the other two remained at approximately 7.660M and 7.090M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 04:03:16 CST** follow-up, three 5080 lanes advanced to approximately **7.090M, 7.895M, and 7.045M / 10M steps**; the other four remained at approximately 5.685M, 4.915M, 6.535M, and 7.660M. GPU utilization was about 83%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:03:57 CST** follow-up, three 5080 lanes advanced to approximately **7.095M, 7.895M, and 7.045M / 10M steps**; the other four remained at approximately 5.685M, 4.915M, 6.535M, and 7.660M. GPU utilization was about 84%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:04:38 CST** follow-up, A100 PointButton 2030 advanced to **6.525M / 10M steps** while PointButton 2028 remained at **4.275M / 10M steps** (GPU utilization about 35%/44%). The 4080 PointButton 2029 lane advanced to **4.245M / 10M steps** (FPS about 25.04, GPU utilization about 32%). On 5080, lanes advanced to approximately **6.540M, 7.665M, 7.095M, 7.895M, 7.045M, 5.685M, and 4.915M / 10M steps**, with GPU utilization about 85%; all observed training processes remained alive.

At the **2026-10-22 04:05:26 CST** follow-up, five 5080 lanes advanced to approximately **4.920M, 6.540M, 7.665M, 7.095M, and 7.045M / 10M steps**; the other two remained at approximately 7.895M and 5.685M. GPU utilization was about 84%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:06:09 CST** follow-up, five 5080 lanes advanced to approximately **7.050M, 5.690M, 4.920M, 6.540M, and 7.665M / 10M steps**; the other two remained at approximately 7.895M and 7.095M. GPU utilization was about 84%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:06:49 CST** follow-up, four 5080 lanes advanced to approximately **7.900M, 7.050M, 5.690M, and 4.920M / 10M steps**; the other three remained at approximately 6.540M, 7.665M, and 7.095M. GPU utilization was about 80%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:07:31 CST** follow-up, four 5080 lanes advanced to approximately **7.900M, 7.050M, 7.665M, and 7.100M / 10M steps**; the other three remained at approximately 6.540M, 5.690M, and 4.920M. GPU utilization was about 85%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:14:48 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.260M / 10M steps** (FPS about 24.50, GPU utilization about 42%). All seven 5080 lanes advanced, reaching approximately **7.905M, 7.675M, 7.105M, 7.060M, 6.550M, 5.700M, and 4.930M / 10M steps**; GPU utilization was about 83%. All observed training processes remained alive; A100 remained active at 4.275M/6.525M.

At the **2026-10-22 04:15:32 CST** follow-up, the 5080 lanes advanced to approximately **7.910M, 7.675M, 7.105M, 7.060M, 6.550M, 5.700M, and 4.930M / 10M steps**; GPU utilization was about 82%. All seven 5080 processes remained alive, and A100/4080 remained active.

At the **2026-10-22 04:16:13 CST** follow-up, the 5080 lanes advanced to approximately **7.910M, 7.675M, 7.110M, 7.060M, 6.550M, 5.700M, and 4.930M / 10M steps**; GPU utilization was about 85%. All seven 5080 processes remained alive; A100 remained at 4.275M/6.525M and 4080 at 4.260M.

At the **2026-10-22 04:17:24 CST** follow-up, the 5080 lanes advanced to approximately **7.910M, 7.680M, 7.110M, 7.060M, 6.555M, 5.700M, and 4.930M / 10M steps**; GPU utilization was about 85%. All seven 5080 processes remained alive; A100 and 4080 remained active.

At the **2026-10-22 04:18:06 CST** follow-up, A100 PointButton 2028 advanced to **4.280M / 10M steps** while PointButton 2030 remained at **6.525M / 10M steps** (GPU utilization about 35%/44%). The 4080 PointButton 2029 lane advanced to **4.265M / 10M steps** (FPS about 24.80, GPU utilization about 34%). 5080 lanes advanced to approximately **7.680M, 6.555M, 7.110M, 7.910M, 7.060M, 5.700M, and 4.935M / 10M steps**, with GPU utilization about 86%; all observed training processes remained alive.

At the **2026-10-22 04:18:57 CST** follow-up, A100 PointButton 2030 advanced to **6.530M / 10M steps** while PointButton 2028 remained at **4.280M / 10M steps** (GPU utilization about 35%/44%). The 4080 remained at **4.265M / 10M steps**, and 5080 lanes advanced to approximately **7.680M, 7.110M, 7.065M, 7.910M, 6.555M, 5.705M, and 4.935M / 10M steps** with GPU utilization about 82%; all observed training processes remained alive.

At the **2026-10-22 04:22:24 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.270M / 10M steps** (GPU utilization about 32%). The 5080 lanes advanced to approximately **7.915M, 7.685M, 7.115M, 7.065M, 6.560M, 5.705M, and 4.940M / 10M steps** with GPU utilization about 82%; all observed training processes remained alive. A100 remained active at approximately 4.280M and 6.530M steps.

At the **2026-10-22 04:24:10 CST** follow-up, three 5080 lanes advanced to approximately **7.920M, 7.120M, and 7.070M / 10M steps**; the remaining lanes were approximately 7.685M, 6.560M, 5.710M, and 4.940M. GPU utilization was about 83%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:24:41 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.275M / 10M steps** (GPU utilization about 34%). The 5080 lanes remained approximately **7.920M, 7.120M, 7.070M, 5.710M, 4.940M, 7.685M, and 6.560M / 10M steps** with GPU utilization about 85%; all observed training processes remained alive. A100 remained active at approximately 4.280M and 6.530M steps.

At the **2026-10-22 04:26:23 CST** follow-up, three 5080 lanes advanced to approximately **7.690M, 6.565M, and 4.945M / 10M steps**; the other lanes remained approximately 7.920M, 7.120M, 7.070M, and 5.710M. GPU utilization was about 85%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:26:51 CST** follow-up, one 5080 lane advanced to approximately **5.715M / 10M steps**; the other lanes were approximately 7.920M, 7.690M, 7.120M, 7.070M, 6.565M, and 4.945M. GPU utilization was about 85%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:27:23 CST** follow-up, one 5080 lane advanced to approximately **7.075M / 10M steps**; the other lanes remained approximately 7.920M, 7.690M, 7.120M, 6.565M, 5.715M, and 4.945M. GPU utilization was about 84%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:27:50 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.280M / 10M steps** (GPU utilization about 34%), and one 5080 lane advanced to approximately **7.925M / 10M steps** (GPU utilization about 82%). The remaining 5080 lanes were approximately 7.075M, 7.120M, 7.690M, 6.565M, 5.715M, and 4.945M; all observed training processes remained alive. A100 remained active at approximately 4.280M and 6.530M steps.

At the **2026-10-22 04:28:17 CST** follow-up, one 5080 lane advanced to approximately **7.125M / 10M steps**; the other lanes were approximately 7.925M, 7.075M, 7.690M, 6.565M, 5.715M, and 4.945M. GPU utilization was about 83%, and all seven 5080 training processes remained alive. A100 and 4080 remained active at approximately 4.280M and 6.530M/4.280M steps respectively.

At the **2026-10-22 04:30:17 CST** follow-up, two 5080 lanes advanced to approximately **7.695M and 6.570M / 10M steps**; the other lanes were approximately 7.925M, 7.125M, 7.075M, 5.715M, and 4.945M. GPU utilization was about 84%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:30:42 CST** follow-up, one 5080 lane advanced to approximately **4.950M / 10M steps**; the other lanes were approximately 7.925M, 7.695M, 7.125M, 7.075M, 6.570M, and 5.715M. GPU utilization was about 85%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:31:13 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.285M / 10M steps** (GPU utilization about 28%), and one 5080 lane advanced to approximately **5.720M / 10M steps** (GPU utilization about 76%). The remaining 5080 lanes were approximately 7.925M, 7.695M, 7.125M, 7.075M, 6.570M, and 4.950M; all observed training processes remained alive. A100 remained active at approximately 4.280M and 6.530M steps.

At the **2026-10-22 04:31:39 CST** follow-up, A100 PointButton 2028 advanced to **4.285M / 10M steps** while PointButton 2030 remained at **6.530M / 10M steps** (GPU utilization about 42%/50%). One 5080 lane advanced to approximately **7.080M / 10M steps** (GPU utilization about 85%); the remaining lanes were approximately 7.925M, 7.695M, 7.125M, 6.570M, 5.720M, and 4.950M. The 4080 remained active at approximately 4.285M; all observed training processes remained alive.

At the **2026-10-22 04:32:15 CST** follow-up, the leading 5080 lane advanced to approximately **7.930M / 10M steps** (GPU utilization about 81%); the remaining lanes were approximately 7.080M, 7.695M, 7.125M, 6.570M, 5.720M, and 4.950M. All seven 5080 training processes remained alive; A100 and 4080 remained active.

At the **2026-10-22 04:32:42 CST** follow-up, one 5080 lane advanced to approximately **7.130M / 10M steps** (GPU utilization about 83%), while the remaining lanes were approximately 7.930M, 7.695M, 7.080M, 6.570M, 5.720M, and 4.950M. A100 PointButton 2030 advanced to approximately **6.535M / 10M steps** (A100 GPU utilization about 35%/43%); 4080 remained active at approximately 4.285M. All observed training processes remained alive.

At the **2026-10-22 04:34:30 CST** follow-up, two 5080 lanes advanced to approximately **7.700M and 6.575M / 10M steps**; the other lanes were approximately 7.930M, 7.130M, 7.080M, 5.720M, and 4.950M. GPU utilization was about 85%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:35:02 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.290M / 10M steps** (GPU utilization about 30%), and one 5080 lane advanced to approximately **4.955M / 10M steps** (GPU utilization about 82%). The remaining 5080 lanes were approximately 7.700M, 7.930M, 7.130M, 7.080M, 6.575M, and 5.720M; all observed training processes remained alive. A100 remained active at approximately 4.285M and 6.535M steps.

At the **2026-10-22 04:35:31 CST** follow-up, one 5080 lane advanced to approximately **5.725M / 10M steps** (GPU utilization about 83%); the remaining lanes were approximately 7.930M, 7.700M, 7.130M, 7.080M, 6.575M, and 4.955M. All seven 5080 training processes remained alive; A100 and 4080 remained active.

At the **2026-10-22 04:35:58 CST** follow-up, two 5080 lanes advanced to approximately **7.935M and 7.085M / 10M steps**; the other lanes were approximately 7.700M, 7.130M, 6.575M, 5.725M, and 4.955M. GPU utilization was about 85%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:36:44 CST** follow-up, one 5080 lane advanced to approximately **7.135M / 10M steps** (GPU utilization about 85%); the remaining lanes were approximately 7.935M, 7.700M, 7.085M, 6.575M, 5.725M, and 4.955M. All seven 5080 training processes remained alive; A100 and 4080 remained active.

At the **2026-10-22 04:38:36 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.295M / 10M steps** (GPU utilization about 28%), and two 5080 lanes advanced to approximately **7.705M and 6.580M / 10M steps** (GPU utilization about 85%). The remaining 5080 lanes were approximately 7.935M, 7.135M, 7.085M, 5.725M, and 4.955M; all observed training processes remained alive. A100 remained active at approximately 4.285M and 6.535M steps.

At the **2026-10-22 04:39:05 CST** follow-up, the 4080 PointButton 2029 lane remained at approximately **4.295M / 10M steps** (GPU utilization about 30%). Three 5080 lanes advanced to approximately **7.705M, 6.580M, and 4.960M / 10M steps** (GPU utilization about 83%); the other lanes were approximately 7.935M, 7.135M, 7.085M, and 5.725M. All observed training processes remained alive; A100 remained active at approximately 4.285M and 6.535M steps.

At the **2026-10-22 04:39:32 CST** follow-up, one 5080 lane advanced to approximately **5.730M / 10M steps** (GPU utilization about 83%); the remaining lanes were approximately 7.935M, 7.705M, 7.135M, 7.085M, 6.580M, and 4.960M. All seven 5080 training processes remained alive; A100 and 4080 remained active.

At the **2026-10-22 04:41:15 CST** follow-up, three 5080 lanes advanced to approximately **7.940M, 7.140M, and 7.090M / 10M steps**; the other lanes were approximately 7.705M, 6.580M, 5.730M, and 4.960M. GPU utilization was about 87%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:41:43 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.300M / 10M steps** (GPU utilization about 29%). The 5080 lanes remained approximately 7.940M, 7.140M, 7.090M, 7.705M, 6.580M, 5.730M, and 4.960M with GPU utilization about 84%; all observed training processes remained alive. A100 remained active at approximately 4.285M and 6.535M steps.

At the **2026-10-22 04:43:25 CST** follow-up, three 5080 lanes advanced to approximately **7.710M, 6.585M, and 4.965M / 10M steps**; the other lanes were approximately 7.940M, 7.140M, 7.090M, and 5.730M. GPU utilization was about 77%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:43:52 CST** follow-up, one 5080 lane advanced to approximately **5.735M / 10M steps**; the other lanes were approximately 7.940M, 7.710M, 7.140M, 7.090M, 6.585M, and 4.965M. GPU utilization was about 84%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:44:26 CST** follow-up, two 5080 lanes advanced to approximately **7.945M and 7.095M / 10M steps**; the other lanes were approximately 7.710M, 7.140M, 6.585M, 5.735M, and 4.965M. GPU utilization was about 87%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:44:54 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.305M / 10M steps** (GPU utilization about 32%). Three 5080 lanes advanced to approximately **7.945M, 7.145M, and 7.095M / 10M steps** (GPU utilization about 86%); the other lanes were approximately 7.710M, 6.585M, 5.735M, and 4.965M. All observed training processes remained alive; A100 remained active at approximately 4.285M and 6.535M steps.

At the **2026-10-22 04:46:40 CST** follow-up, A100 PointButton 2028 advanced to **4.290M / 10M steps** while PointButton 2030 remained at **6.535M / 10M steps** (GPU utilization about 34%/41%). One 5080 lane advanced to approximately **6.590M / 10M steps** (GPU utilization about 84%); the remaining lanes were approximately 7.945M, 7.145M, 7.095M, 7.710M, 5.735M, and 4.965M. The 4080 remained active at approximately 4.305M; all observed training processes remained alive.

At the **2026-10-22 04:47:15 CST** follow-up, A100 PointButton 2030 advanced to **6.540M / 10M steps** while PointButton 2028 remained at **4.290M / 10M steps** (GPU utilization about 43%/51%). One 5080 lane advanced to approximately **7.715M / 10M steps** (GPU utilization about 76%); the remaining lanes were approximately 7.945M, 7.145M, 7.095M, 6.590M, 5.735M, and 4.965M. The 4080 remained active at approximately 4.305M; all observed training processes remained alive.

At the **2026-10-22 04:47:51 CST** follow-up, one 5080 lane advanced to approximately **4.970M / 10M steps** (GPU utilization about 80%); the remaining lanes were approximately 7.945M, 7.715M, 7.145M, 7.095M, 6.590M, and 5.735M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 04:48:30 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.310M / 10M steps** (GPU utilization about 26%). Two 5080 lanes advanced to approximately **7.950M and 5.740M / 10M steps** (GPU utilization about 81%); the other lanes were approximately 7.715M, 7.145M, 7.095M, 6.590M, and 4.970M. A100 remained active at approximately 4.290M and 6.540M; all observed training processes remained alive.

At the **2026-10-22 04:49:00 CST** follow-up, one 5080 lane advanced to approximately **7.100M / 10M steps** (GPU utilization about 83%); the remaining lanes were approximately 7.950M, 7.715M, 7.145M, 6.590M, 5.740M, and 4.970M. A100 and 4080 remained active at approximately 4.290M/6.540M and 4.310M respectively; all observed training processes remained alive.

At the **2026-10-22 04:49:28 CST** follow-up, one 5080 lane advanced to approximately **7.150M / 10M steps** (GPU utilization about 88%); the remaining lanes were approximately 7.950M, 7.715M, 7.100M, 6.590M, 5.740M, and 4.970M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 04:51:21 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.315M / 10M steps** (GPU utilization about 27%), and two 5080 lanes advanced to approximately **7.720M and 6.595M / 10M steps** (GPU utilization about 83%). The other lanes were approximately 7.950M, 7.150M, 7.100M, 5.740M, and 4.970M; all observed training processes remained alive. A100 remained active at approximately 4.290M and 6.540M steps.

At the **2026-10-22 04:51:53 CST** follow-up, one 5080 lane advanced to approximately **4.975M / 10M steps** (GPU utilization about 84%); the other lanes were approximately 7.950M, 7.720M, 7.150M, 7.100M, 6.595M, and 5.740M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 04:52:26 CST** follow-up, one 5080 lane advanced to approximately **5.745M / 10M steps** (GPU utilization about 84%); the remaining lanes were approximately 7.950M, 7.720M, 7.150M, 7.100M, 6.595M, and 4.975M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 04:52:55 CST** follow-up, two 5080 lanes advanced to approximately **7.955M and 5.745M / 10M steps** (GPU utilization about 85%); the remaining lanes were approximately 7.720M, 7.150M, 7.100M, 6.595M, and 4.975M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 04:53:23 CST** follow-up, three 5080 lanes advanced to approximately **7.955M, 7.155M, and 7.105M / 10M steps**; the other lanes were approximately 7.720M, 6.595M, 5.745M, and 4.975M. GPU utilization was about 81%, and all seven 5080 training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 04:55:13 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.320M / 10M steps** (GPU utilization about 28%), and one 5080 lane advanced to approximately **6.600M / 10M steps** (GPU utilization about 81%). The other 5080 lanes were approximately 7.955M, 7.155M, 7.105M, 7.720M, 5.745M, and 4.975M; A100 remained active at approximately 4.290M and 6.540M. All observed training processes remained alive.

At the **2026-10-22 04:55:44 CST** follow-up, one 5080 lane advanced to approximately **7.725M / 10M steps** (GPU utilization about 86%); the remaining lanes were approximately 7.955M, 7.155M, 7.105M, 6.600M, 5.745M, and 4.975M. The 4080 remained active at approximately 4.320M; A100 remained active at approximately 4.290M and 6.540M. All observed training processes remained alive.

At the **2026-10-22 04:56:17 CST** follow-up, one 5080 lane advanced to approximately **4.980M / 10M steps** (GPU utilization about 83%); the remaining lanes were approximately 7.955M, 7.725M, 7.155M, 7.105M, 6.600M, and 5.745M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 04:56:48 CST** follow-up, two 5080 lanes advanced to approximately **7.960M and 5.750M / 10M steps** (GPU utilization about 86%); the remaining lanes were approximately 7.725M, 7.155M, 7.105M, 6.600M, and 4.980M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 04:57:24 CST** follow-up, one 5080 lane advanced to approximately **7.110M / 10M steps** (GPU utilization about 83%); the remaining lanes were approximately 7.960M, 7.725M, 7.155M, 6.600M, 5.750M, and 4.980M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 04:57:52 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.325M / 10M steps** (GPU utilization about 22%). Four 5080 lanes advanced to approximately **7.960M, 7.160M, 7.110M, and 5.750M / 10M steps** (GPU utilization about 85%); the remaining lanes were approximately 7.725M, 6.600M, and 4.980M. A100 remained active at approximately 4.290M and 6.540M; all observed training processes remained alive.

At the **2026-10-22 04:59:45 CST** follow-up, one 5080 lane advanced to approximately **6.605M / 10M steps** (GPU utilization about 82%); the remaining lanes were approximately 7.960M, 7.725M, 7.160M, 7.110M, 5.750M, and 4.980M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:00:15 CST** follow-up, three 5080 lanes advanced to approximately **7.730M, 6.605M, and 4.985M / 10M steps** (GPU utilization about 81%); the other lanes were approximately 7.960M, 7.160M, 7.110M, and 5.750M. A100 remained active at approximately 4.290M and 6.540M, and 4080 at approximately 4.325M; all observed training processes remained alive.

At the **2026-10-22 05:00:46 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.330M / 10M steps** (GPU utilization about 27%), and one 5080 lane advanced to approximately **5.755M / 10M steps** (GPU utilization about 79%). The other 5080 lanes were approximately 7.960M, 7.730M, 7.160M, 7.110M, 6.605M, and 4.985M; A100 remained active at approximately 4.290M and 6.540M. All observed training processes remained alive.

At the **2026-10-22 05:01:15 CST** follow-up, A100 PointButton 2028 advanced to **4.295M / 10M steps** while PointButton 2030 remained at **6.540M / 10M steps** (GPU utilization about 42%/51%). Two 5080 lanes advanced to approximately **7.965M and 5.755M / 10M steps** (GPU utilization about 84%); the other lanes were approximately 7.730M, 7.160M, 7.110M, 6.605M, and 4.985M. The 4080 remained active at approximately 4.330M; all observed training processes remained alive.

At the **2026-10-22 05:01:50 CST** follow-up, two 5080 lanes advanced to approximately **7.165M and 7.115M / 10M steps** (GPU utilization about 84%); the other lanes were approximately 7.965M, 7.730M, 6.605M, 5.755M, and 4.985M. A100 and 4080 remained active at approximately 4.295M/6.540M and 4.330M respectively; all observed training processes remained alive.

At the **2026-10-22 05:02:22 CST** follow-up, A100 PointButton 2030 advanced to **6.545M / 10M steps** while PointButton 2028 remained at **4.295M / 10M steps** (GPU utilization about 34%/42%). The 5080 lanes remained approximately 7.965M, 7.165M, 7.115M, 7.730M, 6.605M, 5.755M, and 4.985M with GPU utilization about 80%; the 4080 remained active at approximately 4.330M. All observed training processes remained alive.

At the **2026-10-22 05:04:18 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.335M / 10M steps** (GPU utilization about 27%), and two 5080 lanes advanced to approximately **7.735M and 6.610M / 10M steps** (GPU utilization about 79%). The other 5080 lanes were approximately 7.965M, 7.165M, 7.115M, 5.755M, and 4.985M; A100 remained active at approximately 4.295M and 6.545M. All observed training processes remained alive.

At the **2026-10-22 05:04:47 CST** follow-up, two 5080 lanes advanced to approximately **7.735M and 4.990M / 10M steps** (GPU utilization about 81%); the other lanes were approximately 7.165M, 7.115M, 7.965M, 6.610M, and 5.755M. The 4080 remained active at approximately 4.335M, and A100 at approximately 4.295M and 6.545M; all observed training processes remained alive.

At the **2026-10-22 05:05:17 CST** follow-up, three 5080 lanes advanced to approximately **7.970M, 5.760M, and 4.990M / 10M steps** (GPU utilization about 81%); the other lanes were approximately 7.735M, 7.165M, 7.115M, and 6.610M. A100 and 4080 remained active at approximately 4.295M/6.545M and 4.335M respectively; all observed training processes remained alive.

At the **2026-10-22 05:06:58 CST** follow-up, two 5080 lanes advanced to approximately **7.170M and 7.120M / 10M steps** (GPU utilization about 87%); the other lanes were approximately 7.970M, 7.735M, 6.610M, 5.760M, and 4.990M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:07:27 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.340M / 10M steps** (GPU utilization about 27%). The 5080 lanes remained approximately 7.970M, 7.170M, 7.120M, 7.735M, 6.610M, 5.760M, and 4.990M with GPU utilization about 84%; A100 remained active at approximately 4.295M and 6.545M. All observed training processes remained alive.

At the **2026-10-22 05:07:58 CST** follow-up, one 5080 lane advanced to approximately **6.615M / 10M steps** (GPU utilization about 83%); the remaining lanes were approximately 7.970M, 7.735M, 7.170M, 7.120M, 5.760M, and 4.990M. A100 and 4080 remained active at approximately 4.295M/6.545M and 4.340M respectively; all observed training processes remained alive.

At the **2026-10-22 05:08:26 CST** follow-up, one 5080 lane advanced to approximately **7.740M / 10M steps** (GPU utilization about 84%); the remaining lanes were approximately 7.970M, 7.170M, 7.120M, 6.615M, 5.760M, and 4.990M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:08:54 CST** follow-up, one 5080 lane advanced to approximately **4.995M / 10M steps** (GPU utilization about 83%); the remaining lanes were approximately 7.970M, 7.740M, 7.170M, 7.120M, 6.615M, and 5.760M. A100 remained active at approximately 4.295M and 6.545M, and 4080 remained active at approximately 4.340M; all observed training processes remained alive.

At the **2026-10-22 05:11:09 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.345M / 10M steps** (GPU utilization about 26%), and two 5080 lanes advanced to approximately **7.175M and 7.125M / 10M steps** (GPU utilization about 82%). The other 5080 lanes were approximately 7.975M, 7.740M, 6.615M, 5.760M, and 4.995M; A100 remained active at approximately 4.295M and 6.545M. All observed training processes remained alive.

At the **2026-10-22 05:12:02 CST** follow-up, one 5080 lane advanced to approximately **6.620M / 10M steps** (GPU utilization about 78%); the remaining lanes were approximately 7.975M, 7.740M, 7.175M, 7.125M, 5.760M, and 4.995M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:12:37 CST** follow-up, one 5080 lane advanced to approximately **7.745M / 10M steps** (GPU utilization about 86%); the remaining lanes were approximately 7.975M, 7.175M, 7.125M, 6.620M, 5.760M, and 4.995M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:13:06 CST** follow-up, two 5080 lanes advanced to approximately **7.750M and 5.000M / 10M steps** (GPU utilization about 84%); the remaining lanes were approximately 7.975M, 7.175M, 7.125M, 6.620M, and 5.760M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:13:34 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.350M / 10M steps** (GPU utilization about 28%), and the leading 5080 lane advanced to approximately **7.980M / 10M steps** (GPU utilization about 85%). The other 5080 lanes were approximately 7.750M, 7.175M, 7.125M, 6.620M, 5.760M, and 5.000M; A100 remained active at approximately 4.295M and 6.545M. All observed training processes remained alive.

At the **2026-10-22 05:15:18 CST** follow-up, two 5080 lanes advanced to approximately **7.180M and 7.130M / 10M steps** (GPU utilization about 84%); the other lanes were approximately 7.980M, 7.750M, 6.620M, 5.760M, and 5.000M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:15:50 CST** follow-up, A100 PointButton 2028 advanced to **4.300M / 10M steps** while PointButton 2030 remained at **6.545M / 10M steps** (GPU utilization about 42%/51%). Two 5080 lanes advanced to approximately **7.130M and 5.770M / 10M steps** (GPU utilization about 85%); the other lanes were approximately 7.980M, 7.180M, 7.750M, 6.620M, and 5.000M. The 4080 remained active at approximately 4.350M; all observed training processes remained alive.

At the **2026-10-22 05:16:48 CST** follow-up, A100 PointButton 2030 advanced to **6.550M / 10M steps** while PointButton 2028 remained at **4.300M / 10M steps** (GPU utilization about 34%/41%). The 4080 PointButton 2029 lane advanced to **4.355M / 10M steps** (GPU utilization about 27%). One 5080 lane advanced to approximately **6.625M / 10M steps** (GPU utilization about 83%); the remaining lanes were approximately 7.980M, 7.750M, 7.180M, 7.130M, 5.770M, and 5.000M. All observed training processes remained alive.

At the **2026-10-22 05:17:25 CST** follow-up, one 5080 lane advanced to approximately **5.005M / 10M steps** (GPU utilization about 85%); the remaining lanes were approximately 7.980M, 7.750M, 7.180M, 7.130M, 6.625M, and 5.770M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:17:55 CST** follow-up, two 5080 lanes advanced to approximately **7.985M and 5.775M / 10M steps** (GPU utilization about 87%); the remaining lanes were approximately 7.750M, 7.180M, 7.130M, 6.625M, 5.005M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:19:37 CST** follow-up, two 5080 lanes advanced to approximately **7.185M and 7.135M / 10M steps** (GPU utilization about 85%); the other lanes were approximately 7.985M, 7.750M, 6.625M, 5.775M, and 5.005M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:20:04 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.360M / 10M steps** (GPU utilization about 27%), and one 5080 lane advanced to approximately **7.135M / 10M steps** (GPU utilization about 85%). The other 5080 lanes were approximately 7.985M, 7.750M, 7.185M, 6.625M, 5.775M, and 5.005M; A100 remained active at approximately 4.300M and 6.550M. All observed training processes remained alive.

At the **2026-10-22 05:20:35 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.360M / 10M steps** (GPU utilization about 26%), and one 5080 lane advanced to approximately **6.630M / 10M steps** (GPU utilization about 83%). The other 5080 lanes were approximately 7.985M, 7.750M, 7.185M, 7.135M, 5.775M, and 5.005M; A100 remained active at approximately 4.300M and 6.550M. All observed training processes remained alive.

At the **2026-10-22 05:21:07 CST** follow-up, one 5080 lane advanced to approximately **6.630M / 10M steps** (GPU utilization about 82%); the remaining lanes were approximately 7.985M, 7.750M, 7.185M, 7.135M, 5.775M, and 5.005M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:21:33 CST** follow-up, two 5080 lanes advanced to approximately **7.755M and 5.010M / 10M steps** (GPU utilization about 83%); the other lanes were approximately 7.985M, 7.185M, 7.135M, 6.630M, and 5.775M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:22:01 CST** follow-up, the leading 5080 lane advanced to approximately **7.990M / 10M steps** (GPU utilization about 81%); the remaining lanes were approximately 7.755M, 7.185M, 7.135M, 6.630M, 5.775M, and 5.010M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:22:31 CST** follow-up, one 5080 lane advanced to approximately **5.780M / 10M steps** (GPU utilization about 85%); the remaining lanes were approximately 7.990M, 7.755M, 7.185M, 7.135M, 6.630M, and 5.010M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:23:02 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.365M / 10M steps** (GPU utilization about 28%), and one 5080 lane advanced to approximately **7.190M / 10M steps** (GPU utilization about 79%). The other 5080 lanes were approximately 7.990M, 7.755M, 7.185M, 6.630M, 5.780M, and 5.010M; A100 remained active at approximately 4.300M and 6.550M. All observed training processes remained alive.

At the **2026-10-22 05:23:30 CST** follow-up, the 4080 PointButton 2029 lane remained at approximately **4.365M / 10M steps** (GPU utilization about 25%), and one 5080 lane advanced to approximately **7.140M / 10M steps** (GPU utilization about 75%). The other 5080 lanes were approximately 7.990M, 7.755M, 7.190M, 6.630M, 5.780M, and 5.010M; A100 remained active at approximately 4.300M and 6.550M. All observed training processes remained alive.

At the **2026-10-22 05:24:25 CST** follow-up, one 5080 lane advanced to approximately **6.635M / 10M steps** (GPU utilization about 85%); the remaining lanes were approximately 7.990M, 7.755M, 7.190M, 7.140M, 5.780M, and 5.010M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:25:28 CST** follow-up, one 5080 lane advanced to approximately **7.760M / 10M steps** (GPU utilization about 86%); the remaining lanes were approximately 7.990M, 7.190M, 7.140M, 6.635M, 5.780M, and 5.010M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:26:07 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.370M / 10M steps** (GPU utilization about 23%), and three 5080 lanes advanced to approximately **7.995M, 7.760M, and 5.015M / 10M steps** (GPU utilization about 82%). The other lanes were approximately 7.190M, 7.140M, 6.635M, and 5.780M; A100 remained active at approximately 4.300M and 6.550M. All observed training processes remained alive.

At the **2026-10-22 05:26:41 CST** follow-up, two 5080 lanes advanced to approximately **5.785M and 5.015M / 10M steps** (GPU utilization about 83%); the other lanes were approximately 7.995M, 7.760M, 7.190M, 7.140M, and 6.635M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:27:09 CST** follow-up, one 5080 lane advanced to approximately **7.195M / 10M steps** (GPU utilization about 87%); the remaining lanes were approximately 7.995M, 7.760M, 7.140M, 6.635M, 5.785M, and 5.015M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:27:48 CST** follow-up, two 5080 lanes advanced to approximately **7.200M and 7.145M / 10M steps** (GPU utilization about 83%); the other lanes were approximately 7.995M, 7.760M, 6.635M, 5.785M, and 5.015M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:28:45 CST** follow-up, two 5080 lanes advanced to approximately **6.640M and 7.145M / 10M steps** (GPU utilization about 82%); the other lanes were approximately 7.995M, 7.760M, 7.195M, 5.785M, and 5.015M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:29:26 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.375M / 10M steps** (GPU utilization about 26%), and one 5080 lane advanced to approximately **6.640M / 10M steps** (GPU utilization about 86%). The other 5080 lanes were approximately 7.995M, 7.760M, 7.195M, 7.145M, 5.785M, and 5.015M; A100 remained active at approximately 4.300M and 6.550M. All observed training processes remained alive.

At the **2026-10-22 05:30:01 CST** follow-up, A100 PointButton 2028 advanced to **4.305M / 10M steps** while PointButton 2030 remained at **6.550M / 10M steps** (GPU utilization about 41%/50%). The 4080 PointButton 2029 lane advanced to **4.375M / 10M steps** (GPU utilization about 27%). Two 5080 lanes advanced to approximately **7.765M and 5.020M / 10M steps** (GPU utilization about 82%); the remaining lanes were approximately 7.995M, 7.195M, 7.145M, 6.640M, and 5.785M. All observed training processes remained alive.

At the **2026-10-22 05:31:19 CST** follow-up, A100 PointButton 2028 remained at **4.305M / 10M steps** while PointButton 2030 advanced to **6.555M / 10M steps** (GPU utilization about 42%/50%). The 4080 remained at approximately **4.375M / 10M steps** (GPU utilization about 27%). Several 5080 lanes advanced, reaching approximately **8.000M, 7.765M, 7.200M, 7.145M, 6.640M, 5.790M, and 5.020M / 10M steps** (GPU utilization about 77%). All observed training processes remained alive.

At the **2026-10-22 05:31:57 CST** follow-up, two 5080 lanes advanced to approximately **7.150M and 5.790M / 10M steps** (GPU utilization about 82%); the other lanes were approximately 8.000M, 7.765M, 7.200M, 6.640M, and 5.020M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:32:38 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.380M / 10M steps** (GPU utilization about 29%). The 5080 lanes remained approximately 8.000M, 7.765M, 7.200M, 7.150M, 6.640M, 5.790M, and 5.020M with GPU utilization about 80%; A100 remained active at approximately 4.305M and 6.555M. All observed training processes remained alive.

At the **2026-10-22 05:33:07 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.380M / 10M steps** (GPU utilization about 25%), and one 5080 lane advanced to approximately **6.645M / 10M steps** (GPU utilization about 82%). The other lanes were approximately 8.000M, 7.765M, 7.200M, 7.150M, 5.790M, and 5.020M; A100 remained active at approximately 4.305M and 6.555M. All observed training processes remained alive.

At the **2026-10-22 05:33:53 CST** follow-up, one 5080 lane advanced to approximately **7.770M / 10M steps** (GPU utilization about 83%); the remaining lanes were approximately 8.000M, 7.200M, 7.150M, 6.645M, 5.790M, and 5.020M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:34:23 CST** follow-up, three 5080 lanes advanced to approximately **8.005M, 7.775M, and 5.025M / 10M steps** (GPU utilization about 83%); the other lanes were approximately 7.200M, 7.150M, 6.645M, and 5.790M. A100 remained active at approximately 4.305M and 6.555M, and 4080 at approximately 4.380M; all observed training processes remained alive.

At the **2026-10-22 05:34:51 CST** follow-up, three 5080 lanes advanced to approximately **8.005M, 7.770M, and 5.795M / 10M steps** (GPU utilization about 82%); the other lanes were approximately 7.200M, 7.150M, 6.645M, and 5.025M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:35:18 CST** follow-up, three 5080 lanes advanced to approximately **8.005M, 7.205M, and 5.795M / 10M steps** (GPU utilization about 83%); the other lanes were approximately 7.770M, 7.150M, 6.645M, and 5.025M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:35:55 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.385M / 10M steps** (GPU utilization about 28%), and two 5080 lanes advanced to approximately **8.005M and 7.775M / 10M steps** (GPU utilization about 84%). The other lanes were approximately 7.205M, 7.150M, 6.645M, 5.795M, and 5.025M; A100 remained active at approximately 4.305M and 6.555M. All observed training processes remained alive.

At the **2026-10-22 05:36:23 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.385M / 10M steps** (GPU utilization about 27%), and the leading 5080 lane advanced to approximately **8.010M / 10M steps** (GPU utilization about 85%). The remaining lanes were approximately 7.770M, 7.205M, 7.150M, 6.645M, 5.795M, and 5.025M; A100 remained active at approximately 4.305M and 6.555M. All observed training processes remained alive.

At the **2026-10-22 05:37:05 CST** follow-up, three 5080 lanes advanced to approximately **7.210M, 7.205M, and 6.650M / 10M steps** (GPU utilization about 84%); the other lanes were approximately 8.010M, 7.770M, 5.795M, and 5.025M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:38:05 CST** follow-up, one 5080 lane advanced to approximately **6.650M / 10M steps** (GPU utilization about 87%); the remaining lanes were approximately 8.005M, 7.770M, 7.210M, 7.205M, 5.795M, and 5.025M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:38:36 CST** follow-up, three 5080 lanes advanced to approximately **8.010M, 7.775M, and 5.030M / 10M steps** (GPU utilization about 83%); the other lanes were approximately 7.205M, 7.150M, 6.650M, and 5.790M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:39:11 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.390M / 10M steps** (GPU utilization about 27%), and one 5080 lane advanced to approximately **5.800M / 10M steps** (GPU utilization about 80%). The other lanes were approximately 8.010M, 7.775M, 7.205M, 7.150M, 6.650M, and 5.030M; A100 remained active at approximately 4.305M and 6.555M. All observed training processes remained alive.

At the **2026-10-22 05:39:48 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.390M / 10M steps** (GPU utilization about 27%), and one 5080 lane advanced to approximately **7.210M / 10M steps** (GPU utilization about 81%). The other lanes were approximately 8.010M, 7.775M, 7.150M, 6.650M, 5.800M, and 5.030M; A100 remained active at approximately 4.305M and 6.555M. All observed training processes remained alive.

At the **2026-10-22 05:40:29 CST** follow-up, two 5080 lanes advanced to approximately **7.215M and 7.210M / 10M steps** (GPU utilization about 84%); the other lanes were approximately 8.010M, 7.775M, 6.650M, 5.800M, and 5.030M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:41:24 CST** follow-up, two 5080 lanes advanced to approximately **7.215M and 6.655M / 10M steps** (GPU utilization about 80%); the other lanes were approximately 8.010M, 7.775M, 7.150M, 5.800M, and 5.030M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:42:36 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.395M / 10M steps** (GPU utilization about 27%), and three 5080 lanes advanced to approximately **7.780M, 6.655M, and 5.035M / 10M steps** (GPU utilization about 83%). The other lanes were approximately 8.010M, 7.215M, 7.150M, and 5.800M; A100 remained active at approximately 4.305M and 6.555M. All observed training processes remained alive.

At the **2026-10-22 05:43:06 CST** follow-up, three 5080 lanes advanced to approximately **8.015M, 7.785M, and 5.035M / 10M steps** (GPU utilization about 86%); the other lanes were approximately 7.205M, 7.150M, 6.655M, and 5.795M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:43:47 CST** follow-up, three 5080 lanes advanced to approximately **8.015M, 7.215M, and 5.805M / 10M steps** (GPU utilization about 82%); the other lanes were approximately 7.780M, 7.150M, 6.655M, and 5.035M. A100 and 4080 remained active; all observed training processes remained alive.

At the **2026-10-22 05:44:16 CST** follow-up, A100 PointButton 2028 advanced to **4.310M / 10M steps** while PointButton 2030 remained at **6.550M / 10M steps** (GPU utilization about 36%/40%). The leading 5080 lane remained at approximately **8.015M / 10M steps** (GPU utilization about 84%); the 4080 remained active at approximately 4.395M. All observed training processes remained alive.

At the **2026-10-22 05:44:58 CST** follow-up, A100 remained at approximately **4.310M and 6.550M / 10M steps** (GPU utilization about 35%/43%). The leading 5080 lane advanced to approximately **8.020M / 10M steps** (GPU utilization about 82%), and another lane advanced to approximately **7.220M / 10M steps**; the 4080 remained active at approximately 4.395M. All observed training processes remained alive.

At the **2026-10-22 05:45:35 CST** follow-up, A100 PointButton 2030 advanced to **6.560M / 10M steps** while PointButton 2028 remained at **4.310M / 10M steps** (GPU utilization about 33%/42%). The 4080 advanced to **4.400M / 10M steps** (GPU utilization about 25%). Several 5080 lanes advanced to approximately **7.220M, 6.660M, and 5.810M / 10M steps** (GPU utilization about 81%); the leading lane was approximately 8.015M. All observed training processes remained alive.

At the **2026-10-22 05:46:11 CST** follow-up, A100 remained at approximately **4.310M and 6.560M / 10M steps** (GPU utilization about 34%/41%). The 4080 remained active at approximately **4.400M**. Two 5080 lanes advanced to approximately **7.220M and 6.660M / 10M steps** (GPU utilization about 84%); the leading lane remained around 8.015M. All observed training processes remained alive.

At the **2026-10-22 05:09:26 CST** follow-up, three 5080 lanes advanced to approximately **7.975M, 5.765M, and 4.995M / 10M steps** (GPU utilization about 82%); the other lanes were approximately 7.740M, 7.170M, 7.120M, and 6.615M. A100 and 4080 remained active at approximately 4.295M/6.545M and 4.340M respectively; all observed training processes remained alive.

At the **2026-10-22 05:48 CST** follow-up, all three machines remained healthy and training processes were alive: A100 PointButton 2028/2030 were approximately **4.310M/6.560M**, the 4080 lane was approximately **4.400M**, and the leading 5080 lane was approximately **8.020M / 10M steps** (other active lanes were in the roughly 5.0M–7.8M range). Instantaneous GPU utilization was about **42%/51%** on the two A100 devices, **26%** on the 4080, and **81%** on the 5080. No new failure or stopped process was observed.

At the **2026-10-22 05:50 CST** follow-up, the 4080 PointButton-2029 lane advanced to approximately **4.405M / 10M steps** and a previously paused Walker2d-2028 run was successfully resumed from its **2.63M / 5M** checkpoint. The 4080 GPU rose to about **44%** during restart; A100 remained around **4.310M/6.560M** and the leading 5080 lane around **8.020M**. No launch error or failed process remained.

At the **2026-10-22 05:53 CST** follow-up, the resumed 4080 Walker2d-2028 process began writing fresh progress and reached approximately **2.605M / 5M steps**; both 4080 processes remained alive and GPU utilization was about **42%**. A100 remained at 15 active processes (GPU about 33%/43%), while 5080 remained at 7 active processes with a leading lane around **8.025M / 10M** (GPU about 83%).

At the **2026-10-22 05:54 CST** follow-up, an additional 4080 Ant-2028 run was resumed from its existing **2.52M / 5M** checkpoint. The 4080 now has **3 active training processes**, GPU utilization rose to about **55%**, and the new process reached the training initialization stage without errors.

At the **2026-10-22 05:55 CST** follow-up, a fourth 4080 process (Walker2d-2029, resumed from approximately **2.605M / 5M**) was started successfully. The 4080 now has **4 active training processes** and GPU utilization increased to about **67%**; the new process reached STAR route-A initialization without errors.

At the **2026-10-22 05:57 CST** follow-up, the resumed 4080 Walker2d-2028 lane advanced to approximately **2.610M / 5M steps**. All 4 processes remained alive, with 4080 GPU utilization about **67%**; Ant-2028 and Walker2d-2029 were still in high-load initialization with no errors. The leading 5080 lane remained around **8.030M / 10M**.

At the **2026-10-22 05:58 CST** follow-up, the resumed 4080 Ant-2028 lane began writing progress at approximately **2.505M / 5M steps**; Walker2d-2028 remained around **2.610M / 5M**. Walker2d-2029 was still in active initialization (CPU about 101%) with no error output, and all 4 4080 processes remained alive (GPU about **67%**). The leading 5080 lane remained around **8.030M / 10M**.

At the **2026-10-22 05:59 CST** follow-up, the 4080 Walker2d-2029 lane also began writing progress at approximately **2.605M / 5M steps**. All four 4080 lanes now have active progress records; GPU utilization was about **65%**. A100 remained at 15 processes and 5080 at 7 processes, with the leading 5080 lane around **8.030M / 10M**.

At the **2026-10-22 06:01 CST** follow-up, the 4080 Ant-2028 lane advanced to approximately **2.510M / 5M steps**. All four 4080 processes remained alive (GPU about **61%**); A100 remained at 15 processes and 5080 at 7 processes, with the leading 5080 lane around **8.035M / 10M**.

At the **2026-10-22 06:02 CST** follow-up, the 4080 Walker2d-2029 lane advanced to approximately **2.610M / 5M steps**; its progress file was refreshed normally. All 4 4080 processes remained alive with GPU utilization about **66%**. No task completion or failure was observed on A100 or 5080.

At the **2026-10-22 06:04 CST** follow-up, the leading 5080 lane advanced to approximately **8.040M / 10M steps** (GPU utilization about **82%**). The 4080 continued with 4 active processes at about **62%** GPU utilization, and A100 continued with 15 active processes at approximately **42%/51%**. No failure or task exit was observed.

At the **2026-10-22 06:05 CST** follow-up, the 4080 PointButton-2029 lane advanced to approximately **4.430M / 10M steps**. All 4 4080 processes remained alive (GPU about **66%**); A100 remained at 15 active processes and 5080 at 7 active processes. No failure or task exit was observed.

At the **2026-10-22 06:06 CST** follow-up, 4080 PointButton-2029 advanced to approximately **4.430M** and Walker2d-2028 to **2.625M / 5M steps**. All four 4080 processes remained alive; GPU utilization was about **52%**. A100 and 5080 processes remained active, with the leading 5080 lane around **8.040M / 10M**.
