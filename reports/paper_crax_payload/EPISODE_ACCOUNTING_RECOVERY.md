# Latest Iso episode bookkeeping: preserved evidence and explicit recovery

The completed `699c5be` native-dual Iso500224 checkpoint has 256 genuine replay
truncation flags and native elapsed counter954, but runner complete-episode count0
and partial lengths1954. Thus native1000-step resets and replay bootstrap labels
were present, while episode bookkeeping missed that completion. The original
checkpoint/receipt/validation are not overwritten or rewritten as corrected data.

Two frozen-state boundary probes (256 controls each) both returned256 true done
and truncation flags and correctly updated episode accounting. The second used
asynchronous consumers. They **did not reproduce or prove the root cause**.
Raw probe results are in [episode-boundary-699](episode-boundary-699/).

Prospectively the boolean done output now uses the native DLPack object directly,
avoiding the official helper's intermediate JAXfloat cast followed by Torchbool
conversion and foreign-temporary lifetime boundary. No host/device transfer or
per-step synchronization was introduced. Native physics, reset timing, replay
terminal observation and learner loss/update budgets are unchanged. The suspect
conversion path is removed, not presented as a uniquely proven causal diagnosis.

The same-config2M continuation uses an explicit, guarded offline recovery flag.
Because the complete500224 prefix is still in chronological replay, all real
completed return/cost/goal/length statistics can be reconstructed without another
environment step. It recovered256 complete episodes and partial lengths954;
episode-cost EMA is99.4234364. Raw accumulated training cost26635.54296875 is
preserved, including its original float summation. All models, optimizers,
multiplier, replay, environment and random streams are untouched.

This operation is allowed only for `native_per_step`, where the episode EMA/count
are diagnostic and cannot change the dual update. It rejects episode-dual cases,
missing historical replay or any nonzero distance-reward scaling. One focused CPU
contract verifies reconstruction and rejection; no historical rollout is rerun.

The new native run is `crax-iso-nativedual-41027-2m-v3`, PID35500 at admission,
source`f35ccdc`; its receipt will charge only actual new transitions. Recovered
old episodes are stored separately from new episodes. Subsequent observed boundary
counts were observed at1029376 transitions:1024 total episodes, consistent with
four complete256-environment cohorts, including the256 recovered historical
episodes. The continuation's800k validation still has0 goals and mean cost9.47;
the bookkeeping repair is not a task-learning improvement claim.
