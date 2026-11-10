# EA-STAB v2 development:running,not a repair-success claim

Scientific source/configuration: `3c308e49d4d93d49c15ee8112071d04087e622d8`.
All six initial trials retain300k from fresh initialization. No scientific
early stop,parameter change,extra candidate or formal-confirmation launch
has occurred. Final600k-to1M stability is still unmeasured.

## Verified runtime snapshot,2026-11-10 23:48:22 Asia/Taipei

| Host / GPU | Trial | Actual logged transitions | Native PID |
|---|---|---:|---:|
| A100 /1 | S-Iso2027 | 101k | 1587947 |
| A100 /0 | S-EA2028 | 17k | 1627251 |
| 4080 /0 | S-EA2027 | 91k | 2559845 |
| 4080 /0 | S-Local2027 | 126k | 2560137 |

All four native identities and both queue identities were directly observed
live. S-Local2028 and S-Iso2028 remain queued,not discarded. A100's controller
handoff changed only shared-resource admission;the original native training
handle and all scientific settings were preserved. No other project was stopped.

The compact mirror can be slightly older than this read-only snapshot;its
recorded lower-bound counters are in [DEVELOPMENT_STATUS.json](DEVELOPMENT_STATUS.json).
Live partial steps are neither completed trials nor a whole-goal percentage.

## First fixed development checkpoint: S-Local2027,100k

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
robustness or training-safety target. The other matched method/seed100k
results were not yet mirrored at this capture,so no method ranking or
independent EA benefit is claimed. The policy continues to its fixed300k;
the outcome is retained rather than replaced or used as an early-stop gate.

Full condition metrics,return,capped/conditional time and raw episode evidence:
[DEVELOPMENT_RESULTS.csv](DEVELOPMENT_RESULTS.csv). The currently empty
[FINAL_RESULTS.csv](FINAL_RESULTS.csv) has no completed fixed1M outcome yet;
it must not be read as an implicit success or omitted requirement.

## Budget and remaining work

Engineering training consumed3210 new transitions. Predevelopment nontraining
physical probes consumed30906 control intervals and are separately itemized
in [TRIAL_LEDGER.csv](TRIAL_LEDGER.csv). Native development counters/physical
evaluation costs update as compact artifacts are mirrored. Full replay and
policies remain on the authorized servers/4080's4T,not in GitHub.

Initial planned training remains1.8M;total development training is capped6M.
Only evidence-supported bounded candidates may follow (at most4x100k),then
paired native Iso/selected-geometry long runs must cover600k-to1M. Useful
late stability and a frozen final configuration are required before any
conditional new-seed9M confirmation. There is no new5080 use.
