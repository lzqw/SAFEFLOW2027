# EA-STAB v2 development:running,not a repair-success claim

Scientific learner/configuration: `3c308e49d4d93d49c15ee8112071d04087e622d8`.
Recovery driver94ffff3 only permits explicit completion of matching terminal
native trials;the learner,geometry,physics,evaluation and profile remain
identical. No scientific early stop,parameter change,extra candidate or
formal-confirmation launch has occurred. Final600k-to1M stability is unmeasured.

## Verified runtime snapshot,2026-11-11 10:53:33 Asia/Taipei

| Host / GPU | Trial | Actual logged transitions | Native PID |
|---|---|---:|---:|
| A100 /1 | S-Iso2027 native300k-to1M | 360k | 1928659 |
| A100 /0 | S-Iso2028 native230k-to300k recovery | 254k | 1951046 |
| 4080 /0 | S-EA2028 native300k-to1M | 306k | 229868 |

All three native identities and the recovery/long queue identities were
directly observed live. EA2027's native recovery completed300k;Iso2028
continues from its230k native full checkpoint. Five of six initial endpoints
are now complete. EA2027's next700k is queued on A100 GPU0 after the occupied
recovery slot;Iso2028's700k remains to be started on4080 after its300k file
exists. Both methods therefore remain represented on both authorized hosts.
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

## Fixed300k results available so far

Each row is one training seed;nominal has30 episodes,each stress20. The stress
columns are the equal-weight mean of four conditions,not independent seeds.

| Trial | Nominal SS / violation | Stress SS / violation | Radius8cm violation | Native training violations |
|---|---:|---:|---:|---:|
| S-Iso2027 |96.67% /3.33%|71.25% /28.75%|70%|942|
| S-Local2027 |100% /0%|88.75% /11.25%|40%|756|
| S-Local2028 |100% /0%|83.75% /16.25%|65%|817|
| S-EA2027 |100% /0%|97.5% /2.5%|10%|1007 (accounted at least1022)|
| S-EA2028 |100% /0%|98.75% /1.25%|5%|1001|

Both matched EA/Local seed pairs now favor EA's300k evaluation robustness,
but EA costs at least28.6% more mean training violations after retaining the
known discarded tail. This is two development seeds,not a new-seed confirmation
or proof of late stability. Iso2028 fixed300k remains in explicit engineering
completion;it is not assigned zero or silently removed. No method is declared
repaired and no best intermediate checkpoint will replace fixed1M. The initial
EA package is selected for the planned native long comparison;see
[CONTINUATION_SELECTION.md](CONTINUATION_SELECTION.md). EA2028 can use4080
without waiting for the whole recovery wave. EA2027's A100 queue waits only
for the already occupied GPU0 recovery slot;Iso2027 continues on GPU1.

There is substantial intermediate variation:Local2027 radius8cm violations
were45%,90%,40% at100/200/300k;EA2028 were60%,50%,5%. Continuing the fixed
learning curves matters. Training violations756–1001 already at300k do not
meet the350/1M effort target;do not extrapolate these into invented1M scores.

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
