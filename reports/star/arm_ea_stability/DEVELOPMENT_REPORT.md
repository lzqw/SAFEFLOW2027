# EA-STAB v2 development:running,not a repair-success claim

Scientific learner/configuration: `3c308e49d4d93d49c15ee8112071d04087e622d8`.
Recovery driver94ffff3 only permits explicit completion of matching terminal
native trials;the learner,geometry,physics,evaluation and profile remain
identical. No scientific early stop,parameter change,extra candidate or
formal-confirmation launch has occurred. Both Iso600k endpoints are now
available;all four paired Iso/EA600k endpoints,both Iso fixed1M endpoints,
and EA2028's fixed1M endpoint are now available. EA2027's fixed1M endpoint remains pending. Complete
two-seed600k-to1M stability is not yet established. EA2027 has an explicit
same-profile resource handoff at450k,not an extra scientific candidate.

## Latest completed endpoints:19:23 capture,not a repair-success claim

Three of four predeclared long lineages have completed their actual1M
endpoint and all five100-episode final conditions. EA2027 remains active at
769k in the scheduled snapshot;its incomplete result is not excluded or
replaced. All six initial300k trials and all four600k evaluations remain.

| Completed fixed1M trial | Nominal SS / violation | Four-stress SS / violation | Radius8cm violation | Accounted training violations |
|---|---:|---:|---:|---:|
| Iso2027 |95% /5%|59.75% /40.25%|95%|2597|
| Iso2028 |95% /5%|83.75% /16.25%|46%|at least2187|
| EA2028 |87% /13%|53.5% /46.5%|97%|2179|

No safe timeout occurs in any completed final condition. This is not a
whole-method training-seed mean:the EA2027 final row is still missing.
In the completed matched2028 pair,EA is worse on held-out safe success and
violations in every condition. Known logged/retained-tail accounting gives
2179 training violations versus Iso's lower bound2187,a difference of8
(0.37% of that lower bound). Unlogged interruption-tail costs are unknown,
so this is not an exact all-expenditure cost difference or evidence that
the intended350/1M training-safety target has been achieved.
The final EA radius8cm result is97 violations among100 episodes,not a
collision-only statistic. The favorable early300k results cannot substitute
for this endpoint or justify immediate new-seed confirmation.

| Seed2028 fixed1M condition | EA SS / violation | Iso SS / violation | EA return | EA capped time(s) |
|---|---:|---:|---:|---:|
| Nominal |87% /13%|95% /5%|8.694|1.108|
| Radius7cm |28% /72%|83% /17%|-4.191|3.067|
| Radius8cm |3% /97%|54% /46%|-9.791|3.900|
| Lateral+/-3cm |91% /9%|98% /2%|9.558|0.977|
| Lateral+/-6cm |92% /8%|100% /0%|9.758|0.950|

EA2028's logged600k-to1M policy-Q maximum stays below11.048 and recent
training safe success89–100%;its final policy-Q mean9.800,max10.129,
sampled replay upper-bound exceedance0,reward TD0.091,energy17.932,
clipping6.02%,and multiplier0. Training recent success at1M is100%.
Therefore a training recent-success/Q-scale snapshot alone would give a
misleadingly favorable impression of this policy's held-out behavior.
Its final failure is not accompanied by the old large positive-Q scale
near35 or persistent nominal training stagnation. These logs do not yet
identify the causal contribution of margin calibration,direction or Actor
distribution;the bounded new-policy mechanism analysis remains required.

EA2028 finishes with2179 training violations and55821 completed episodes;
998000 learner updates,1996000 reward updates,and498500 actor updates.
Its terminal native long-segment elapsed time29384.39s includes training,
evaluations and saves. The final500 episodes add6500 physical control
intervals,accounted separately from training. All large full snapshots and
policies are already on4080 4T;no local large-file staging is needed.

Current logged new training including engineering is4380210,conservative
charge4385210 with the5000 reserve,and recorded nontraining controls94523.
There are125 development condition rows and15 completed fixed1M rows.
No extra short candidate or conditional9M confirmation has started. Finish
the remaining native1M line,then complete mechanism diagnosis and the
bounded final decision;the full objective is still active.

## Runtime progression,2026-11-11 Asia/Taipei

The following three handles were directly verified at10:53:33:

| Host / GPU | Trial | Actual logged transitions | Native PID |
|---|---|---:|---:|
| A100 /1 | S-Iso2027 native300k-to1M | 360k | 1928659 |
| A100 /0 | S-Iso2028 native230k-to300k recovery | 254k | 1951046 |
| 4080 /0 | S-EA2028 native300k-to1M | 306k | 229868 |

All six initial300k endpoints are now complete. The11:23 observation verified
Iso2027 long at408k and EA2028 long at355k,and observed Iso2028's completed
300k receipt with its native process exited. Its one-shot staging receipt at
11:24 confirms the protected full snapshot was archived to4080 4T and the
declared700k queue started (controller308005). EA2027's long queue launched
native PID1997214 on A100 GPU0 at11:25. Iso2028 long initially waits for
shared CPU/memory resource admission on4080;this is not an evaluation gate.
The full long matrix is Iso/EA x2027/2028,with both methods on both hosts.
See [ISO2028_STAGING_RECEIPT.json](ISO2028_STAGING_RECEIPT.json).
At11:33 all four native identities were directly verified alive:Iso2027 at
422k and EA2027 at305k on A100;EA2028 at365k and Iso2028 at306k on4080.
The latter's first admission saw only1.84GiB free VRAM,then its5-minute
retry admitted it with14.24GiB free. No foreign task was stopped and no
already running learner was moved or restarted to achieve the four slots.

The11:53 half-hour snapshot again verified all four native identities alive:
Iso2027 453k,EA2027 316k,EA2028 386k,Iso2028 344k. There is no completed600k
evaluation yet at that capture. The read-only watcher now persists its log
and latest JSON independently of interactive tool sessions;its previous
process had genuinely exited,but the training processes were healthy and
were not restarted. See [LATEST_RUNTIME_SNAPSHOT.json](LATEST_RUNTIME_SNAPSHOT.json).
These snapshots are progress evidence,not new policy scores or a completion
percentage for the full objective.
The two other4080 jobs failed with CUDA unspecified launch failure;their
emergency CUDA-RNG checkpoint also failed. The host subsequently rebooted.
This is not evidence of a mathematical NaN or a precise hardware root cause.
No original logs or results were replaced,and no other project was stopped.

The4080 had load26.13/28CPU at10:17,so two recovery launches were refused
before any child was created. Its native checkpoints were streamed directly
to A100 (no local large-file copy). Its load returned to6.28 at10:23;future
admission reused that headroom after scientific continuation selection. The
10:45 long admission again saw a transient25.95 load;the idle own controller
was replaced without stopping a native learner,and its resource retry is now
5 minutes rather than30. External read-only inspection remains30 minutes.

The compact mirror can be slightly older than this read-only snapshot;its
recorded lower-bound counters are in [DEVELOPMENT_STATUS.json](DEVELOPMENT_STATUS.json).
Live partial steps are neither completed trials nor a whole-goal percentage.

## Complete fixed300k matrix

Each row is one training seed;nominal has30 episodes,each stress20. The stress
columns are the equal-weight mean of four conditions,not independent seeds.

| Trial | Nominal SS / violation | Stress SS / violation | Radius8cm violation | Native training violations |
|---|---:|---:|---:|---:|
| S-Iso2027 |96.67% /3.33%|71.25% /28.75%|70%|942|
| S-Iso2028 |100% /0%|95% /5%|20%|847 (accounted849)|
| S-Local2027 |100% /0%|88.75% /11.25%|40%|756|
| S-Local2028 |100% /0%|83.75% /16.25%|65%|817|
| S-EA2027 |100% /0%|97.5% /2.5%|10%|1007 (accounted at least1022)|
| S-EA2028 |100% /0%|98.75% /1.25%|5%|1001|

Both matched EA/Local seed pairs now favor EA's300k evaluation robustness,
but EA costs at least28.6% more mean training violations after retaining the
known discarded tail. This is two development seeds,not a new-seed confirmation
or proof of late stability. All initial endpoints,including the two recovered
ones,are retained. No method is declared repaired and no best intermediate
checkpoint will replace fixed1M. The unchanged first EA package is selected
for the paired long comparison;see
[CONTINUATION_SELECTION.md](CONTINUATION_SELECTION.md).

| Two-seed mean at300k | Nominal SS | Stress SS / violation | Radius8cm violation | Accounted training violations |
|---|---:|---:|---:|---:|
| S-Iso |98.33%|83.125% /16.875%|45%|at least895.5|
| S-Local |100%|86.25% /13.75%|52.5%|786.5|
| S-EA |100%|98.125% /1.875%|7.5%|at least1011.5|

No safe timeout occurred in these fixed300k tests. Stable Iso and Local both
already learn the nominal task,so this early EA comparison does not rely on
the historical collapsed Iso baseline. EA has the most favorable300k risk
results in both seeds,but about13.0% more accounted training violations than
Iso and28.6% more than Local. These descriptive means retain only two training
seeds;the test episodes are not additional independent training seeds.

There is substantial intermediate variation:Local2027 radius8cm violations
were45%,90%,40% at100/200/300k;EA2028 were60%,50%,5%. Continuing the fixed
learning curves matters. Known accounted violations756–1022 already at300k
exceed the350/1M effort target. Cumulative counts cannot decrease during
continuation;do not erase the prefix or invent final1M scores.

## First fixed600k endpoints: Iso only,2026-11-11 13:30 capture

Both native Iso continuations passed600k and produced their predeclared
evaluations:30 nominal episodes and20 per stress,layout seed start120027,
training RNG preserved. These are not additional evaluations launched by the
collector. The two evaluations consumed4491 physical control intervals,
separate from training. EA was only375k/500k in this compact capture;do not
compare its300k scores with Iso600k as a matched endpoint comparison.

Each entry below is safe success / violation / safe timeout,all percentages.

| Fixed600k condition | Iso2027 | Iso2028 |
|---|---:|---:|
| Nominal |96.67 /3.33 /0|93.33 /0 /6.67|
| Radius7cm |65 /35 /0|100 /0 /0|
| Radius8cm |35 /65 /0|90 /10 /0|
| Lateral+/-3cm |100 /0 /0|85 /0 /15|
| Lateral+/-6cm |100 /0 /0|70 /5 /25|
| Four-stress mean |75 /25 /0|86.25 /3.75 /10|

Across the two training seeds,nominal safe success is95%,stress safe success
80.625%,stress violation14.375%,stress safe timeout5%,and radius8cm violation
37.5%. These are descriptive development means,not a three-seed confirmation.
Native training violations are1508/1428 at600k;retaining the known discarded
Iso2028 tail gives1508/1430 and a mean of1469. Completed native training
episodes are32109/30583;accounted episodes32109/30641. Low stress risk in
Iso2028 is not uniformly successful behavior:the lateral6cm condition has
five safe timeouts among20 episodes. The cumulative350/1M target is already
unattainable for these lineages.

The600k learner rows alone look benign:policy-Q means9.699/9.524,maxima
10.123/10.114,zero sampled replay upper-bound exceedance,clipping5.37%/5.18%,
energy18.381/18.896,and multiplier0 for both. However,reading the intervening
native curve reveals a real transient Iso2028 failure that a milestone-only
summary would hide:

- At534k,logged signed policy-Q maximum17.958 exceeds even the global15
  task upper bound;policy/replay means are9.798/9.785,and multiplier is0.
- At548k,40.23% of sampled replay actions exceed their state-specific
  reward-Q upper bound;recent safe success is17% and mean episode length
  78.07 control intervals. At553k,recent safe success reaches0%,mean episode
  length98.02,and multiplier is0. These are recent training-episode windows,
  not held-out test scores or proof that every state has collapsed.
- By560k,recent success is90%,sampled replay bound exceedance is0,and
  by570k recent success is100%. There was no restart or scientific change.
  The600k held-out nominal result nonetheless retains two safe timeouts.

This is evidence that the first shared stability package has not eliminated
all critic/behavior excursions. Temporal association does not prove the
critic spike caused the later behavior. Nor does multiplier0 at the peak
and trough exclude its nonzero values in between. It does rule out a claim
that the entire interval is explained solely by a large persistent safety
penalty. The original complete1M continuations remain mandatory.

No short candidate has yet been launched. If the late interval shows
recurring material instability,the authorized single-factor actor-interval
2-to4 comparison is the relevant first candidate;PID,changing noise,or
silently clipping reported Q would not address the observed evidence.
Because this excursion recovered before600k,we retain the current complete
paired long matrix rather than replacing its trajectories mid-run. Final
method selection still depends on fixed1M results and the new-policy
mechanism analysis,not this partial Iso-only checkpoint.

Raw rows and episode records are in [DEVELOPMENT_RESULTS.csv](DEVELOPMENT_RESULTS.csv)
and the linked native evaluation JSONs. The refreshed curves retain the
excursion. At this capture,the logged new training lower bound including
engineering is2699210,conservative charge2703210 including the4000 reserve;
nontraining physical controls68960. There are100 development condition rows,
zero completed fixed1M rows,and no formal confirmation started.

## First matched EA/Iso600k seed pair:2028,14:32 capture

EA2028 reached its declared600k evaluation with the same development layout
seed start120027 and episode counts as Iso. Its recorded1855 evaluation
control intervals are now included separately in the nontraining budget.
This completes only one of two planned training-seed pairs at600k;EA2027
remains below that milestone. Local has no600k continuation in the selected
matrix,so its300k result cannot establish an independent600k EA gain.

Entries are safe success / violation / safe timeout percentages.

| Fixed600k condition | EA2028 | Iso2028 |
|---|---:|---:|
| Nominal |100 /0 /0|93.33 /0 /6.67|
| Radius7cm |100 /0 /0|100 /0 /0|
| Radius8cm |75 /25 /0|90 /10 /0|
| Lateral+/-3cm |100 /0 /0|85 /0 /15|
| Lateral+/-6cm |100 /0 /0|70 /5 /25|
| Four-stress mean |93.75 /6.25 /0|86.25 /3.75 /10|

EA improves stress safe success by7.5 percentage points and removes the
observed safe timeouts,but has2.5 points more average stress violations.
Its8cm violation rate is15 points worse than Iso's. EA8cm also regresses
from5% at300k to25% at600k (one versus five violations among20 episodes),
so the early robustness gain does not remain uniformly intact. This small
development sample does not establish statistical significance;the100
episodes per condition at fixed1M and both training seeds are still needed.

The increased task completion is not free training safety:EA has1564 native
training violations and32079 completed episodes at600k,versus Iso1428 native
violations (1430 accounted) and30583 native episodes (30641 accounted).
EA's accounted violation count is9.37% higher. Nominal capped time is0.689s
versus Iso0.947s,but8cm capped time is1.520s versus Iso1.046s;neither metric
supports an all-condition Pareto improvement.

EA's500k-to600k logged policy-Q means stay9.506–9.803,maxima10.075–10.627,
and recent training safe success90–100%. The maximum sampled replay
state-specific upper-bound exceedance is0.39%,versus Iso's40.23% excursion.
At600k EA policy-Q mean9.694,max10.100,energy19.305,clipping5.89%,and
multiplier0. Thus this EA seed's8cm risk regression is not accompanied by
the same large logged value excursion or training-task collapse. That
distinguishes the observed phenomena;it does not prove accurate OOD margin
geometry or a causal mechanism. No new margin query or training change was
introduced to produce this analysis.

All four original continuations remain running unchanged. The compact
capture has EA2027 at411k,Iso2027 at698k,EA2028 at602k,and Iso2028 at862k.
The total logged new training lower bound including engineering is3183210,
with conservative charge3187210 including the4000 reserve;nontraining
physical controls70815. There are105 development condition rows,zero
fixed1M rows,and no extra short candidate or formal-confirmation run.

## Complete paired600k matrix:17:24 capture

EA2027 completed its predeclared600k evaluation after the explicit450k
resource handoff;no additional development rollouts were selected by
quality. Its recorded physical evaluation count is1865 control intervals.
Its full outcomes are:

| EA2027 fixed600k condition | Safe success | Violation | Safe timeout |
|---|---:|---:|---:|
| Nominal |100%|0%|0%|
| Radius7cm |100%|0%|0%|
| Radius8cm |65%|35%|0%|
| Lateral+/-3cm |100%|0%|0%|
| Lateral+/-6cm |100%|0%|0%|

Its native training violations are1515 at32421 completed episodes;including
the initial CUDA failure's known discarded tail gives at least1530 and32774
episodes. The resource handoff adds0 known logged discarded costs,not a
cost-counter reset. Policy-Q mean9.706,max10.135,replay mean9.709,sampled
replay bound exceedance0,energy19.425,clipping6.09%,and multiplier0 at600k
do not show a simultaneous large value blow-up at this endpoint.

| Two training-seed mean at600k | Nominal SS | Stress SS | Stress violation | Stress safe timeout | Radius8cm violation | Accounted training violations |
|---|---:|---:|---:|---:|---:|---:|
| S-Iso |95%|80.625%|14.375%|5%|37.5%|1469|
| S-EA |100%|92.5%|7.5%|0%|30%|1547|

These are the complete two development seeds,not an unseen three-seed
confirmation. EA improves stress safe success by11.875 percentage points
and lowers mean stress violations by6.875 points versus matched Iso,but
uses5.31% more accounted cumulative training violations. Radius8cm's mean
gain is not consistent across seeds:EA2027 risk35% versus Iso65%,whereas
EA2028 risk25% versus Iso10%. No EA safe timeout occurs here,so its mean
evaluation advantage is not obtained solely by stalling.

Both EA seeds lose8cm robustness relative to their own300k checkpoints:
2027 rises10% to35%,2028 rises5% to25%;the seed mean rises7.5% to30%.
The600k mean misses the20%8cm effort target,despite meeting nominal and
aggregate-stress development targets. Each stress estimate still uses only
20 episodes per training seed. This is genuine unfavorable development
evidence,not omitted because the early300k comparison was stronger.
Stable-Local's two300k comparisons remain the independent endpoint-geometry
ablation;there are no Local600k results in the selected matrix and no claim
of a matched long-stage Local advantage. Fixed1M,new-policy oracle geometry
and inference-cost evidence are still needed to separate OOD geometry drift
from shared learner issues and justify any final configuration or confirmation.

No new short candidate or training change has been launched. Compact logged
steps are Iso2027 939k,EA2028 867k,EA2027 603k,and completed Iso2028 1M.
Total logged new training including engineering is4019210,with conservative
charge4024210 including the5000 reserve. Physical nontraining controls80736;
115 development condition rows and5 completed fixed1M condition rows.

## First complete fixed1M endpoint:Iso2028,15:25 capture

This native continuation completed its full700k incremental transitions and
then all five100-episode final conditions. Its terminal receipt records
998000 learner updates,1996000 reward updates,498500 actor updates,and
13108.80 seconds for the long segment including its evaluations and saves.
Final tests use the independent fixed layout seed start130027,not the
middle-checkpoint120027 layouts. No best checkpoint replaces this endpoint.

| Fixed1M condition | Safe success | Violation | Safe timeout | Return | Capped time(s) |
|---|---:|---:|---:|---:|---:|
| Nominal |95%|5%|0%|10.432|0.854|
| Radius7cm |83%|17%|0%|7.823|1.248|
| Radius8cm |54%|46%|0%|1.629|2.206|
| Lateral+/-3cm |98%|2%|0%|11.081|0.757|
| Lateral+/-6cm |100%|0%|0%|11.515|0.706|
| Four-stress mean |83.75%|16.25%|0%|8.012|1.229|

Native cumulative training violations are2185;retaining the old known
discarded tail gives at least2187. Native completed episodes53941,accounted
53999. The final nominal task is functioning,but nominal violation5%,stress
violation16.25%,8cm violation46%,and cumulative training cost all miss their
effort targets in this seed. Nominal95% is not evidence of a three-seed
average. Successful conditional times remain in FINAL_RESULTS.csv alongside
capped times;the8cm conditional mean0.679s must not hide46 failures.

From600k to1M,this seed's logged policy-Q maximum stays below11.982 and
recent training success stays85–100%;at1M policy-Q mean9.712,max10.146,
sampled replay upper-bound exceedance0,energy18.738,clipping6.38%,and
multiplier0. There is no recurrence of its534k global-Q excursion in that
logged interval. This is encouraging for nominal late learning in this one
seed,not evidence that robustness is repaired. The1M radius8cm risk is
46% versus the600k development estimate10%,with different layouts and
episode counts;this is a worse observed final endpoint,not an isolated
causal estimate of temporal drift. All final rows remain visible.

The other Iso seed also had a real post600k training excursion:at634k,
recent success4%,mean episode length35.92,and multiplier2.376;its policy-Q
mean was5.915,not a large positive mean-Q blow-up. It recovered before the
698k mirror. Thus the two observed transient failures cannot all be given
one explanation,or silently removed by reporting only recovered endpoints.
No extra PID or actor-delay candidate is launched from that temporal
association alone. The original paired complete1M comparisons continue.

## Complete Iso fixed1M results:18:23 capture

Iso2027 has now also completed its full700k native continuation and five
100-episode final conditions. Its retained final endpoint is:

| Fixed1M condition | Iso2027 SS / violation | Iso2028 SS / violation | Two-seed mean SS / violation |
|---|---:|---:|---:|
| Nominal |95% /5%|95% /5%|95% /5%|
| Radius7cm |37% /63%|83% /17%|60% /40%|
| Radius8cm |5% /95%|54% /46%|29.5% /70.5%|
| Lateral+/-3cm |98% /2%|98% /2%|98% /2%|
| Lateral+/-6cm |99% /1%|100% /0%|99.5% /0.5%|
| Four-stress mean |59.75% /40.25%|83.75% /16.25%|71.75% /28.25%|

All these final conditions have0 safe timeouts. Mean nominal return10.446
and capped time0.848s,stress return5.413 and capped time1.627s;all individual
returns and capped/conditional times remain in FINAL_RESULTS.csv. Iso2027
has2597 training violations and55282 completed episodes;Iso2028 has2185
native violations (at least2187 accounted) and53941 native episodes (53999
accounted). Accounted two-seed means are2392 violations and54640.5 episodes.

Thus the shared package produces functioning nominal final policies in both
seeds,but does not achieve training safety or robust obstacle-radius
generalization. At600k the Iso mean stress SS was80.625%,violation14.375%,
and8cm violation37.5%;the final estimates are71.75%,28.25%,and70.5%.
The independent final layouts and larger episode count prevent attributing
each delta solely to temporal change,but the observed final outcome misses
the intended robustness and late-risk goals. It must not be replaced by
the better600k checkpoint or described as a completed repair.

At1M Iso2027 policy-Q mean9.716,max10.101,sampled replay bound exceedance0,
energy19.497,clipping6.09%,multiplier0,and recent training success97%.
Its600k-to1M logged policy-Q maximum never exceeds12.117;the brief634k
recent-success trough4% was followed by recovery. Neither final seed
exhibits the old sustained positive-Q scale near35 or nominal failure at
its endpoint. That is evidence about the observed phenotype,not a causal
ablation proving which member of the shared package fixes it or that all
training excursions are eliminated. The remaining severe radius failure
cannot be excused by final nominal-task collapse.

The two EA final policies and their matched1M comparison remain pending.
The original native EA jobs continue on4080;no scientific early stop or
additional candidate is introduced on the basis of the Iso final results.
The compact capture records4259210 logged new training transitions including
engineering,4264210 conservative charge with reserve,and88023 nontraining
physical controls. There are120 development condition rows and10 completed
fixed1M rows;the full goal is not complete.

Both protected Iso2027 full snapshots (600k and1M,about201MB total),milestone
policies and compact raw evidence are now archived in a fresh4080 4T folder.
The direct SSH stream completed with both pipeline exits0 and matching
source/destination file sizes;A100 originals remain. The4080 filesystem has
about1.4TiB free. No large local file was created or historical archive erased.
See [ARCHIVE_TRANSFERS.json](ARCHIVE_TRANSFERS.json). Both Iso1M full states
are therefore available on4T for the eventual four-policy mechanism analysis.

## Resource-only acceleration:EA2027 pinned450k handoff,15:37

Completed Iso2028 released a4080 slot. Preflight observed14.4GiB free VRAM,
53GiB available RAM,and load2.61/28 on4080. The A100 EA2027 process was
advancing near10 actual transitions/s,versus the existing4080 EA workload
near25–28;GPU utilization alone was not used as the decision criterion.

The explicit plan pins the existing atomic full450k checkpoint after a
complete learner round. Only the exact owned A100 native PID1997214 was
briefly frozen while the pinned full state and compact parent records were
streamed to a fresh4080 4T archive. After destination CPU validation,the source
was intentionally terminated,not scientifically early-stopped. A new native
PID681280 resumes the remaining550k to the same1M target. The source archive,
raw logs and pinned full state are retained on A100 and4080;no large file was
written locally. No other project was signalled and no5080 resource was used.

Driver e7be7ea adds only a guarded explicit resource-handoff resume path;
learner,geometry,configuration,evaluation and runtime reconstruction are
unchanged from the original scientific package. EMA,all optimizers,replay,
Python/NumPy/Torch/CUDA RNG,episode counters and partial observed physical
states were loaded. Native first progress451k confirms449000 learner updates,
898000 reward updates,224000 actor updates and cumulative cost1286,without
resetting the lineage. This is observed-state continuation,not a claim of
bitwise-identical Bullet solver caches or cross-GPU arithmetic.

The last logged source row and pinned checkpoint are both450k,so known
discarded tail transitions,violations and episodes are all0. A separate1000
transition reserve covers any unlogged work between checkpoint and freeze;
it is not claimed as observed training. Total unlogged reserve is now5000,
and the projected complete core training charge is at least4610210 logged
plus5000 reserve,still below6M. No scientific candidate or new seed was added.
The old parent process lifetime and transfer pause remain separately visible
in TRAINING_THROUGHPUT.csv rather than being called a CUDA failure or omitted
from timing. The first2000 destination transitions averaged27.89/s;this short
prefix suggests acceleration,but a longer measured window is needed for ETA.

See [EA2027_RESOURCE_HANDOFF_RECEIPT.json](EA2027_RESOURCE_HANDOFF_RECEIPT.json)
and [the finite plan](../../../configs/star/arm_ea_stability/ea2027_resource_handoff_v2.json).
The half-hour observer now follows the new exact native identity instead of
requiring the intentionally terminal old A100 queue to remain alive.

The15:53 scheduled snapshot verified all three remaining native identities
alive:Iso2027 820k on A100,EA2028 738k and EA2027 476k on4080. Iso2028's
completed1M receipt remains visible. A20k-transition destination window now
measures26.75/s for EA2027 (477k compact capture),versus its original A100
segment near9.97/s:about2.7x observed throughput across this handoff. This
is a practical same-trial resource improvement,not a causal cross-hardware
algorithm speed benchmark. The shared scientific source files remain exactly
identical to3c308e4 by focused diff. Current remaining pure-training estimates
are2.28h Iso2027,2.69h EA2028,and5.43h EA2027;allow6–8h overall for resource
variation and milestones,then1–3h for final mechanism/collection/reporting.
The revised estimate is late evening11 November or early12 November,not the
previous next-day-morning estimate. Conditional new-seed confirmation is not
included and still requires the full scientific late-stage decision.

## Earlier fixed checkpoint retained: S-Local2027,100k

Training cumulative violations:322 at100000 actual train transitions,including
random warmup and critic preheat. Entries below are counts,not seed-averaged
conclusions. There were no safe timeouts in these five test conditions.

| Condition | Safe successes | Violating failures | Episodes |
|---|---:|---:|---:|
| Nominal | 29 | 1 | 30 |
| Radius7cm | 17 | 3 | 20 |
| Radius8cm | 11 | 9 | 20 |
| Lateral+/-3cm | 19 | 1 | 20 |
| Lateral+/-6cm | 18 | 2 | 20 |

This policy learns nominal task behavior,but its8cm violation rate is45% in
this small20-episode development test. It has not demonstrated the intended
robustness or training-safety target. At that earlier capture the other
matched results were unavailable. The fixed300k trial has since completed;
this unfavorable checkpoint remains visible rather than being replaced.

Full condition metrics,return,capped/conditional time and raw episode evidence:
[DEVELOPMENT_RESULTS.csv](DEVELOPMENT_RESULTS.csv). [FINAL_RESULTS.csv](FINAL_RESULTS.csv)
now contains both Iso seeds and EA2028's fifteen completed fixed1M conditions.
EA2027's final endpoint is still absent,not implicitly successful or waived.

## Learning curves and measured throughput

The following are generated only from compact mirrored native artifacts;they
do not run new physics queries or change any training process:

- [Fixed nominal and all four stress curves](figures/development-evaluation.png)
- [Training task behavior,episode length,cumulative cost and throughput](figures/development-training.png)
- [Signed replay/policy Q,TD,alpha,energy,multiplier and clipping](figures/development-learner.png)
- [Per-native-segment measured throughput](TRAINING_THROUGHPUT.csv)

Every training seed and native segment remains separate. Missing evaluations
are not filled with zeros,and training recent success is not a held-out score.
The plot manifest records the exact mirrored ages in
[development-curves.json](figures/development-curves.json). The table reports
actual train transitions divided by native wall time,including learner,physics,
completed evaluations and checkpoint overhead within that interval. A running
prefix does not yet include its final milestone evaluation;completed segments
use the terminal receipt so that overhead is included. A recent-window rate is
also shown,not extrapolated completion time. Hardware,shared workloads and
training age differ,so these are not causal EA-vs-Local speed comparisons.

Refresh after the serial collector with
`python -m scripts.star.arm_safe_entropy.plot_stability`. This reporting code
does not modify the immutable3c308e4 scientific source used on either server.

## Budget and remaining work

Engineering smoke training consumed3210 new transitions. The two interrupted
trials have7000 logged transitions and17 logged violations after their last
recoverable full snapshots;these discarded tails still count as real costs.
Recoveries add80000 actual transitions to complete those native endpoints,
so initial development costs at least1.807M,not1.8M. Separately reserve4000
training transitions for unlogged interruption tails;this is a conservative
budget charge,not a claim that4000 additional steps were observed. The global
counter remains a logged lower bound with that explicit reserve.
Evaluation CSV retains the native lineage counter and separately adds known
discarded-tail violations for an accounted lower bound,including descendants.
No raw counter is rewritten to hide repeated work. Predevelopment nontraining
physical probes consumed30906 control intervals and are separately itemized
in [TRIAL_LEDGER.csv](TRIAL_LEDGER.csv). Native development counters/physical
evaluation costs update as compact artifacts are mirrored. Full replay and
policies remain on the authorized servers/4080's4T,not in GitHub.

Initial planned training remains1.8M;total development training is capped6M.
Only evidence-supported bounded candidates may follow (at most4x100k),then
paired native Iso/selected-geometry long runs must cover600k-to1M. Useful
late stability and a frozen final configuration are required before any
conditional new-seed9M confirmation. There is no new5080 use.
