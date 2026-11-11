# EA-STAB v2 development:running,not a repair-success claim

Scientific learner/configuration: `3c308e49d4d93d49c15ee8112071d04087e622d8`.
Recovery driver94ffff3 only permits explicit completion of matching terminal
native trials;the learner,geometry,physics,evaluation and profile remain
identical. No scientific early stop,parameter change,extra candidate or
formal-confirmation launch has occurred. Both Iso600k endpoints are now
available;EA600k and all fixed1M endpoints remain pending. Final600k-to1M
stability is unmeasured.

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
[DEVELOPMENT_RESULTS.csv](DEVELOPMENT_RESULTS.csv). The currently empty
[FINAL_RESULTS.csv](FINAL_RESULTS.csv) has no completed fixed1M outcome yet;
it must not be read as an implicit success or omitted requirement.

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
