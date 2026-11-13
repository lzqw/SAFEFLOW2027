# Robot-Gap v2 — first execution snapshot

This is an interim report, not the final2.2M decision report. All new work is
simulation/offline only. Source implementation:139c4e15; baseline:5a403e04.

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
profiles; normal adaptation uses the native transition ramp. No measured speedup
is claimed until complete training time, updates and short learning are collected.
The shared A100 workload is uncontrolled and must be disclosed in timing results.

12 focused tests pass on A100, including real official-Piper serial/process
parity, transfer isolation and preserved second reward-target Polyak ordering.
The original local Python test environment lacks panda_gym; no broad package
installation or repeated validation campaign was undertaken.

## Budget, resources and outstanding decisions

Receipt refresh confirms1.80M completed old Piper candidate transitions and
90,001,920 completed old formal PointGoal transitions. Neither is newly charged
to this round. This round reserves230K for profiles and initial paired transfer;
later allowance1.97M remains unspent/unreserved. The cap is2.20M, not a target.

The A100 controller starts the paired transfer after profiles, without waiting
for another turn. Healthy foreign jobs remain untouched.4080's confirmed address
172.21.221.238 still returns no route to host. No5080 is used. New artifacts go
to `/storage_0/SAFEFLOW2027_lzqw_uid1007/robot_gap_v2`, not/home.

Still pending: transfer outcomes and paired decision; frozen risk/gradient
diagnosis and Reach adaptation;4080 Payload repair; final-selected PointGoal
64×32 mechanism study and complete efficiency/figures; Reach bundle selection,
deployment-device latency and operator-ready read-only/replay program. Existing
offline controller refuses physical transport; TCP,frames,period,speed and
workspace require现场确认. No real-world outcome or calibrated safety is claimed.

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
