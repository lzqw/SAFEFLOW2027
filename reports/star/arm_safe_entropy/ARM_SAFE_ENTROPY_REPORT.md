# Arm safe entropy — development in progress

Start date: 2026-11-06. Starting commit `a251b7af4864212c7191e2ff80393d0f62936805`; independent branch
`codex/arm-safe-entropy-v1`. This is a new simulated Panda task, not a resumed
navigation matrix. Old outcomes remain unchanged. Current success targets are
development aims, not fabricated results or scientific stopping gates.

## Execution order

1. Three environment clips and minimal semantics checks.
2. Seed2027 unobstructed SAC / reward-only flow, initially100k transitions each;
   optionally200k if learning trends justify continuing.
3. SafeReach obstacle A/B first, then Oracle/Learned direction once foundations work.
   Two development seeds2027/2028,300k each, evaluation100k/200k/300k on common30 layouts.
4. At most two targeted revisions, four extra short runs at most; no broad sweep.
5. If useful: Iso-Low and Random-Aniso controls and fixed-state conditional action
   covariance diagnostics; freeze before formal new seeds2030/2031/2032.
6. Only after first-task evidence: payload extension and two perturbation families.

First-stage training cap is3M actual transitions including ability and engineering
smokes; no vector-env step-count multiplication ambiguity. Later larger stages
are conditional on useful signal, not automatically launched by the queue.

Initial resources: A1002x80GB idle,128 CPU threads; other workloads occupy
4080/5080 and are not interrupted. Python environment isolated in
`/home/lzqw/SAFEFLOW2027_ARM/env`; server root initially about60GiB free.
First platform is panda-gym/PyBullet, for real link-distance labels and exact
same-controller counterfactual geometry. Simulation FPS is not learner FPS.

## Current evidence and open work (2026-11-06 11:37 CST)

The three required clips and minimum semantics checks passed:

| Fixed-layout demonstration | Control steps | Reached safely | Binary cost |
|---|---:|---:|---:|
| Unobstructed reach | 16 | yes | 0 |
| Straight path, obstacle present | 2 | no | 1 |
| Collision-free detour | 33 | yes | 0 |

Primary v2 videos and native step records are in [environment_v2](environment_v2/).
Historical [environment_v1](environment_v1/) evidence remains unchanged; the
v1 detour took34 steps. v2 fixes a stale EE pose cache and completes every full
control interval before returning terminal. All new obstacle comparisons use
v2; geometry, reward, action scale and layout distribution are unchanged.
True horizon is terminal (no bootstrap), and cost is binary. Geometry/energy
and one flow save/resume check passed; see [method changes](METHOD_CHANGES.md).
No all-history validation or new disaster-recovery system was introduced.

Both first ability learners have completed:

- SAC seed2027, source0eba984, A100 GPU0: completed100k training transitions,
  **30/30 successes at25k,50k,75k and100k**, fixed common layout seeds52027–52056,
  stochastic policy evaluation. Native end-to-end runtime913.35s,98k learner
  updates and98k reward/actor updates; full replay remains on A100.
- Reward-only flow seed2027, sourcefc6b3c0, A100 GPU1: completed100k transitions,
  **30/30 successes at25k,50k,75k and100k**; runtime2126.68s,98k learner updates,
  196k reward updates and49k actor updates.

These are **unobstructed ability** results, not obstacle safety or a formal
seed comparison. The full100k budgets were retained despite early successes;
80% is not a programmatic early-stop rule. Both remain the sole behavior
policy of their learner. [DEVELOPMENT_RESULTS.csv](DEVELOPMENT_RESULTS.csv)
labels interim checkpoints explicitly; formal/mechanism/robustness result files
remain header-only. A zero-shot API compatibility check evaluated both100k
ability policies on unobstructed v2: each again succeeded30/30, with no retraining.

One scheduling comparison uses actual productive post-warmup progress:
single SAC5k–10k:111.30 transitions/s; with both learners running,
SAC35k–45k:106.15/s and flow4k–8k:46.35/s, aggregate about152.50/s.
The workloads and sample windows differ: this is an actual heterogeneous
queue capacity comparison, **not** an algorithm speed comparison or a
controlled same-workload scalability claim. Keep two independent learners,
one per currently free A100; do not equate low VRAM use with permission to
duplicate scientific seeds or interrupt foreign jobs.

### Obstacle learning

The v1 SAC-Lag2027 pilot obtained0/30 successes and0/30 violations at100k:
safe stalling, not task mastery. For a coherent Oracle/baseline interface,
both v1 pilots were stopped and preserved after the API correction described
above: SAC last logged124k (unlogged conservative bound125k), Flow-Iso14k
(upper bound15k). Original statuses/progress, checkpoints and stop markers
remain intact; no completion receipt was invented. This is not evidence that
the API bug caused poor v1 learning. v1 and v2 efficacy must not be pooled.

Fresh, matched v2 learners use sourcece0d175, seed2027,300k budgets:

| Method | Latest native checkpoint evaluation | Safe success | Violation | Status |
|---|---:|---:|---:|---|
| SAC-Lag | 100k,30 common-layout episodes | 30/30 | 0/30 | Training continues to300k |
| Safe-FLAC-Iso | Not yet at100k evaluation | — | — | About50k; recent training success0% |

SAC's return at100k was11.1936 and its training violation count97. This is
one development seed/checkpoint, not a cross-seed conclusion or proof of zero
population risk. Flow's training statistic is not substituted for held-out
evaluation. Do not expand the direction matrix until Flow foundation evidence
is available; if A learns but B does not, repair the flow backbone first.
The native [learning curves](figures/SafeReach-Obstacle-v2-learning.png)
separate held-out evaluations from recent training statistics; missing
evaluations are not plotted as zeros.

### Oracle and pending work

The exact simulator-replica oracle is integrated and a bounded interface check
passed: max margin discrepancy3.7253e-8m, minimum normal cosine1.0 among six
valid comparisons; live state unchanged. Exact queries are too costly to
launch the full repeated-target training path without acceleration work.
Two explicit controller-Jacobian approximations were examined. Nearby action
anchors reduced mean margin error to0.091mm but direction cosine averaged0.888,
with one slightly opposed normal and2 missed/2 spurious-valid gradients.
**This approximation is not qualified as an accurate Oracle upper bound.**
No Oracle/Learned training gain is claimed; all mismatch rows are retained.
See [method changes](METHOD_CHANGES.md) for the limited probes and their scope.

One interim foundation diagnostic used256 shared genuinely visited states
(128 with current-state clearance0–20mm above the safety threshold,128 farther
inside),128 independent actions per state, and the exact common normal at
each source replay action. Normal-valid states199/256; undefined directions
remain null, never replaced by a convenient axis. SAC120k and Flow60k had
mean conditional action covariance traces0.05645 and0.04903 respectively.
This does not support diagnosing Flow as effectively deterministic from these
states. Flow hard-clipping averaged7.18% per path-coordinate-step on this
mixed state pool; this differs from on-policy training clipping and includes
states visited by SAC. The policies are at different training ages: neither
relative efficacy nor a directional entropy benefit follows. Actual per-state
covariances, Gaussian marginal entropy proxies, reference covariances and true
interval risk queries are retained in
[foundation-sampling-v2](engineering/foundation-sampling-v2/diagnostic.json).
This diagnostic collected zero new training transitions and does not populate
the still-pending final mechanism table.

Budget bookkeeping retains ability200k, engineering smoke/resume2600 and v1
obstacle conservative upper bound140k. A full matched eight-run v2 matrix
would bring the projected first-stage total to2,742,600 transitions, below3M.
This is a ceiling projection, not a command to launch uninformative runs.

Cost-Q/dual, learned-margin and detached-geometry components are implemented.
Flow obstacle learning, accurate efficient Oracle training, directional-method
evidence, controls, formal runs, payload and robustness remain open. The full
requested objective is not complete. A100 project storage has about61GiB free
at the current check. Large checkpoint/replay files remain there; compact native
JSON, progress and videos are copied through this host to the new GitHub branch.

See [task definition](TASK_DEFINITION.md), [method changes](METHOD_CHANGES.md)
and configs/star/arm_safe_entropy/development_v2.json for current settings.
All primary evaluations will use fixed final checkpoints, common layouts and
per-training-seed statistics. An unsuccessful experiment will be reported as
such, not replaced by a claim that finishing a budget proves effectiveness.
