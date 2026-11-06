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

## Current evidence and open work (2026-11-06 13:10 CST)

Initial eight-run coverage is now2 completed,5 active,1 dependency-queued;
this is launch coverage, **not** completion of the scientific comparison:

| Method | Seed2027 | Seed2028 |
|---|---|---|
| SAC-Lag | Completed300k, A100 | Completed300k,5080 |
| Flow-Iso | Active, A100 | Active,5080 |
| OracleDir controller approximation | Active, A100 GPU0 | Active, same A100 GPU0 |
| LearnedDir | Waits for Flow-Iso2027 completion | Active,5080 |

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
controlled same-workload scalability claim. Initially keep two independent learners,
one per currently free A100; later use additional genuinely available host capacity.
Do not equate low VRAM use with permission to
duplicate scientific seeds or interrupt foreign jobs.

### Obstacle learning

The v1 SAC-Lag2027 pilot obtained0/30 successes and0/30 violations at100k:
safe stalling, not task mastery. For a coherent Oracle/baseline interface,
both v1 pilots were stopped and preserved after the API correction described
above: SAC last logged124k (unlogged conservative bound125k), Flow-Iso14k
(upper bound15k). Original statuses/progress, checkpoints and stop markers
remain intact; no completion receipt was invented. This is not evidence that
the API bug caused poor v1 learning. v1 and v2 efficacy must not be pooled.

Matched v2 baselines use sourcece0d175 for seed2027 and source3c1b407 for
seed2028, with300k budgets. The latter adds runtime metadata/options without
changing the selected baseline optimizer/physics settings:

| Method | Training seed | Latest native evaluation | Safe success | Violation | Status |
|---|---:|---:|---:|---:|---|
| SAC-Lag | 2027 | 300k,30 episodes | 30/30 | 0/30 | Completed; fixed final checkpoint |
| SAC-Lag | 2028 | 300k,30 episodes | 30/30 | 0/30 | Completed; retained earlier100k failure |
| Safe-FLAC-Iso | 2027 | 200k,30 episodes | 30/30 | 0/30 | Continues to300k |
| Safe-FLAC-Iso | 2028 | 200k,30 episodes | 30/30 | 0/30 | Continues to300k |

For seed2027 at a matched100k budget, SAC's return was11.1936 and training violations97;
Flow's return11.3359 and training violations137. SAC200k return11.4292,
training violations162. SAC300k return11.4191, total training violations219,
native runtime4190.35s and298k learner/reward/actor updates. For seed2028 at100k,
SAC return-1.8642/training violations32 versus Flow return11.3626/violations107.
The lower violation count of a stalled policy is not a task-preserving safety
improvement. Both Flow development seeds have now learned at the first fixed
evaluation; SAC's seed difference is retained, not hidden or treated as a final
300k failure. At200k SAC2028 reached30/30 successes, return11.3890 and training
violations119. Its earlier0/30 outcome remains in the table/curve, not discarded.
Flow2027 at200k retained30/30 successes, return11.4376 and training violations195.
SAC2028 finally completed300k with30/30 successes, return11.4195 and training
violations152 (native2595.74s,298k learner/reward/actor updates). Flow2028 at200k
also retained30/30 successes, return11.4408 and training violations244.
This is evidence of later learning, not a basis for an early scientific stop
or a premature multi-parameter change. These per-seed checkpoints do not prove zero population risk or
a directional-method gain. Flow's earlier
stalling was followed by genuine learning, so it was not a reason to terminate
the run or change several parameters prematurely. Foundation evidence now
supports moving to the two direction variants while retaining full A/B budgets.
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

On256 actually visited common state/actions, the local model's mean cosine
was0.97147 (near-boundary0.96531),195/199 exact-valid directions were valid in
both, with4 misses and4 extras. Minimum near-boundary cosine remained-0.32428;
all rows, not just successful ones, are retained. Mean margin error0.0895mm,
sign disagreements0/256. The runnable training variant is explicitly a
**privileged controller-linearization approximation**, not an error-free
model bound. A negative result from it alone cannot disprove a perfect-direction
mechanism. Exact simulator finite differences remain the independent diagnostic.

Pruning provably inactive distance planes compresses the same local linear
model, rather than changing its predictions. Oracle2008-transition smoke and
200-new-transition restoration completed; actual resumed learner throughput
was about2.0 transitions/s before final evaluation, not the cached arithmetic
rate. The small-window end-to-end rate is not a full-run completion guarantee.
Both tests are labelled engineering, not included as trained-method results.

One dependency queue has launched and one remains waiting (sourcee792e45):

| Diagnostic | Status / dependency | A100 GPU | New budget |
|---|---|---:|---:|
| OracleDir, controller-linearization backend, seed2027 | Active PID270049;10k logged, checkpoint saved | 0 | 300k |
| OracleDir, same backend, seed2028 | Active PID297991;3k logged, checkpoint saved | 0 | 300k |
| LearnedDir, seed2027; no oracle calls by its actor | Queue PID252096 waits for Flow-Iso2027 to complete300k | 1 | 300k |

Queue251920 completed its one launch and exited; child270049 and its32
single-threaded controller workers were inspected alive. The3k→4k productive
window took313.91s (~3.19 transitions/s), not the warmup-inflated headline FPS.
This is an early cached-model window, not a guaranteed late-run rate or learned
Oracle efficacy; no held-out directional result exists yet. Each
queue rechecks shared GPU/CPU/disk headroom and launches only one run, not the
entire matrix. Later controls/formal/payload stages are not blindly enqueued.

At12:56 A100 CPU load was below3, GPU0 utilization3%, available RAM108GiB.
The original matrix's Oracle2028 therefore started on the same mostly-waiting
GPU0, source3fd8bbf,32 CPU model workers, with checkpoints every1000 transitions
because model-derived geometry is slow. This changes save cadence, not the
optimizer, task, reference or300k budget. Both scientific learners were checked
alive and updating; the extra run is not a new seed/configuration. Partly
overlapping productive windows were Oracle2027 8k→10k in556.33s (~3.59/s) and
Oracle2028 2k→3k in334.24s (~2.99/s). These small, different cache/data windows
are not a controlled scalability benchmark or guaranteed full-run throughput.
After initialization, available RAM was98GiB and CPU load about6.8, retaining
headroom; no other workload was stopped. At12:56 the4080 still showed95% foreign
GPU usage and was left alone.

### Additional shared capacity:5080

At12:14 the4080 remained98% utilized by foreign workloads, so it was left
untouched. The5080 had spare capacity and approximately14GiB disk free. A new
project lives under `/home/autodrive/LZQW/SAFEFLOW2027_ARM`, with its own venv;
it read-only reuses the old SAFE conda environment's working PyTorch2.7.1+cu128
instead of duplicating CUDA wheels. Panda-gym3.0.7, PyBullet3.2.7, Gymnasium1.2.3,
NumPy1.24.3 and SB32.7.0 are installed in the new venv, not the old project.
CUDA tensor execution and package versions were checked; no three-video
revalidation or historical suite was repeated. Space remains about14GiB.

The existing matrix's seed2028 A/B runs use source3c1b407 there:

|5080 learner|PID|Latest mirrored progress|Budget|
|---|---:|---:|---:|
|SAC-Lag2028|Completed; former3890030|Final300k evaluation|300k|
|Flow-Iso2028|3890092|200k evaluation available|300k|
|LearnedDir2028|3895948|9k, actual margin/critic/actor updates|300k|

Both have productive optimizer updates, not just live startup processes. The
first post-warmup7k→8k SAC and6k→7k flow windows yielded approximately100/s
and83/s respectively; these are short throughput observations, not full-run
ETAs or a controlled hardware benchmark. A foreign Isaac workload returned
while setup was underway and remains running (~3GiB VRAM at the observation).
Each new learner used~0.36GiB VRAM. The observed98% whole-card utilization
includes that foreign job and must not be attributed entirely to this project.
All sharing continues without stopping anyone else's work. Five scientific
learners are currently active across A100+5080, without increasing budgets or
claiming unobserved seed2028 success. Native runtime versions and GPU names
are recorded in new run manifests; the5080 CUDA version differs from A100's11.8.

After SAC-Lag2028 finished its full300k budget, replacement queue3893413
(source267e848) launched LearnedDir2028 once, childPID3895948, and exited.
The child was checked live, with finite margin/critic/actor updates through9k.
It has not produced a100k held-out result yet; a small margin loss is not proof
of an accurate normal or a directional gain. This preserves two project learners on that shared card,
not a third concurrent addition. The explicit16GB-card checks are free VRAM>8GiB,
CPU headroom and disk>8GiB; shared utilization up to100% is allowed, rather than
waiting for an exclusive idle card contrary to the user's sharing choice. The
project directory is about490MB; observed total disk free fluctuated12–14GiB.
No foreign files were removed. All eight original development runs are now
completed, live or explicitly queued; no control/formal/payload expansion is implied.

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

The same analysis implementation now supports labelled interim directional
diagnosis and a strict final-mechanism mode. Final mode requires the two-seed
fixed300k directional/control cohort,256 distinct jointly visited states,
shared exact normals,128 samples/state, reference/endpoint projections,
normal validity counts, same-action normal cosines and legacy cost-weight
factor distributions. It refuses an incomplete foundation-only cohort before
creating output. The complete mechanism analysis has **not** been executed;
the formal mechanism table remains empty. Server-resident replay may be
streamed between servers for this analysis; it is not copied onto this host.

Budget bookkeeping retains ability200k, flow engineering smoke/resume2600,
oracle engineering smoke/resume2208 and v1 obstacle conservative upper bound140k.
A full matched eight-run v2 matrix would bring the projected first-stage total
to2,744,808 transitions, below3M.
This is a ceiling projection, not a command to launch uninformative runs.

Cost-Q/dual, learned-margin and detached-geometry components are implemented.
Flow obstacle learning now has a positive first held-out result. Longer-run
stability, Oracle approximation limits, directional-method evidence, trained
controls, formal runs, payload and robustness remain open. Mechanism controls
are implemented and algebra-checked, but have not been trained or assigned an
arbitrary Iso-Low variance. The full
requested objective is not complete. A100 project storage has about61GiB free
at the current check. Large checkpoint/replay files remain there; compact native
JSON, progress and videos are copied through this host to the new GitHub branch.

See [task definition](TASK_DEFINITION.md), [method changes](METHOD_CHANGES.md)
and configs/star/arm_safe_entropy/development_v2.json for current settings.
All primary evaluations will use fixed final checkpoints, common layouts and
per-training-seed statistics. An unsuccessful experiment will be reported as
such, not replaced by a claim that finishing a budget proves effectiveness.
