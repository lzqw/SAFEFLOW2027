# E01 stage1 validation admission, no final-test peeking

Operational update11:34: the still-idle original waiting controller3016444
was retired after exact command/child checks, with no evaluation executed.
Its replacement3036894 uses `payload_repair_v3_memory_validation.json` and
`stage1-validation-memory-v1`, observing the new serial memory-capped queue.
Only queue binding/output change; all scientific validation fields are equal.
See[resource-only amendment](MEMORY_ADMISSION_AMENDMENT.md).

The original plan already fixed five NEW20/40/60/80/100K candidates for each
of four arms,50 common layouts81027–81076 and four conditions: nominal,70mm
radius,wide payload,execution0.85. This is exactly20×4×50=4000 validation
episodes, not a new candidate or perturbation sweep. The historical failed
EA prefix is excluded as a scientific candidate and retained as325 engineering
transitions in total spending.

A lightweight controller checks real native handles/receipts every1800s. It
waits for all four actual100K completions before importing Torch or running
any validation, so it does not compete with training. It never restarts a
missing/interrupted job, never launches RL, never touches foreign processes.
All evaluation runs on CPU with CUDA hidden and output on the4080's4T disk.

Every full checkpoint's method,seed,configuration,step and SHA are bound.
All20 candidates receive the same common-validation opportunity. Frozen
learner/dual counters must stay unchanged. The native evaluator's action seed,
physical substeps,terminal classification and failure-capped time are retained.
Old `alternative_route` is explicitly excluded; this does not prove route
legality or unseen-radius extrapolation for the repaired training family.

Risk qualification remains nominal≤5%,each stress≤10%; rank safe success,
then capped time,then earlier new-step age; otherwise explicitly labelled
Pareto tradeoff. No requirement EA must win or every checkpoint be stable.
Delta training counts are relative to each true native parent, not confused
with the cumulative historical count or unequal parent snapshot ages.

IMPORTANT: stage1 validation ends with a scientific review, not automatic
final testing or another training stage. The reserved91027–91126 final layouts
stay unopened until deciding whether a conditional continuation is warranted.
Otherwise seeing final labels and then deciding on another200K would turn the
"final" test into development. After stage1 review, freeze configuration and
candidate identities before using any genuinely final test. The325 prefix
also reduces the remaining E01-specific1M allowance as already amended.
