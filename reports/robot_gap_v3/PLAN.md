# Robot-gap v3: finite repair and good-candidate evidence

Registered2026-11-14 before any v3 learning. This follows the user's supplied
E01–E14 plan; prioritize useful good-seed candidates, not forcing every seed or
checkpoint to succeed. Keep complete curves and separate candidate evidence
from fixed-budget training-seed statistics. No physical commands authorized.

## E01 stage1:400K new, maximum E01 allowance1M

One intervention only: change the nominal60mm training sphere into the shared
finite60/67.5/75mm family, round-robin at episode resets. This is NOT a learned
margin constant patch. Reward, layouts, collision geometry checks, physical
substeps, optimizer, learning rate, updates/transition, dual and action sampler
remain unchanged. The repair addresses radius robustness, not all offset risks.

| Arm | Native parent | Additional transitions | Reset distribution |
|---|---|---:|---|
| EA original control | seed41028,240K |100K|60mm|
| EA repair | same exact parent |100K|60/67.5/75mm|
| Iso repair | seed41028,120K |100K|same finite family|
| SAC repair | seed41028,300K |100K|same finite family|

Parents were previously selected using common validation, not new v3 labels.
Different parent ages are disclosed: this is equal NEW adaptation opportunity,
not a claim of equal cumulative parent training. Parent SHA256:

- EA:`628b5be167895c22917a5fac9fd81d57ba2a89feb89e8e7d35d7d7c2d5caef2d`
- Iso:`02a4417db0b70207ad12dee2f662c6a69f57ec786960633ab90a8ddb9b10166c`
- SAC:`d730265122836677179c251b1d57e7133f2ade860c6e3da5b40aa9306aa68d24`

All four explicit forks retain native learner/target/optimizer/EMA/dual/replay,
transition/episode counters and RNG. To avoid silently mutating an active
episode's collision body, ALL arms (including nominal control) start fresh
episodes from the parent's per-environment RNG. Censored parent partial
episodes are recorded with lengths/returns/costs, not labelled failures or
completed episodes, and not passed to the dual. Identical-config `--resume`
remains strict and restores physical partial states; solver caches are not
claimed bitwise reproducible. Every actual step is counted.

Save five full/native and inference candidates after NEW20/40/60/80/100K.
Keep historical30-layout development evaluations and full curves; old labels
are now development evidence. The inherited `alternative_route` condition
name does NOT prove legal alternative routes and is excluded from selection.
New common validation:50 layouts81027–81076 per condition; candidates receive
equal opportunities. Conditions: nominal, radius70mm, wide payload, scale0.85.
Risk-qualified means nominal violation≤5%, each stress condition≤10%; choose
maximum mean safe success, then minimum mean failure-capped completion time,
then earlier new step. If no candidate qualifies, label the selected tradeoff
explicitly. Before seeing final test labels freeze identities and hashes.
New independent final test:100 layouts91027–91126 per condition, identical
layout/action seeds across methods. No final-test-driven reselection.

The additional3×200K (repair EA/Iso/SAC) is CONDITIONAL, not automatically
queued: review stage1 task/risk learning and geometry failure modes first;
no requirement EA must win. Stage1 reserves400K, raising total reservation
from830K to1.23M against2.20M. Stage2 could raise this to1.83M, never automatic.

## Resources and remaining lines

4080 uses its4T data disk, maximum two shared jobs,≥6GiB free GPU memory before
each admission,≥40GiB free disk, half-hour checks; no foreign-task preemption,
no automatic retry and no scientific early stopping. Resource sharing means
historical throughput is not a completion-time guarantee.

E02 frozen Piper Payload matched-risk/frontier and E03 genuinely unseen
geometry/execution conditions use A100/CPU when available; no additional RL.
Register their equal opportunities and new layout ranges before evaluating.
E05 reuses logs first. E06–E08/E12–E14 are not silently added to this queue.
Do not repeat completed PointGoal, independent closure or mechanism tests.

The user confirmed5080 is the actual Piper deployment PC, but asks to avoid
running programs there. No5080 training or timing is launched now; only a short
explicitly arranged offline timing window may later measure E09. Hardware
transport/motion remains disabled. E10/E11 require on-site calibration/operator.
Passwords/private SSH credentials must never be committed.
