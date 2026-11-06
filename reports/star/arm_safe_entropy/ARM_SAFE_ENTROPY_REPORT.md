# Arm safe entropy — development in progress

Start date: 2026-11-06. Starting commit `a251b7a`; independent branch
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

## Current evidence and open work

The three required clips and minimum semantics checks passed:

| Fixed-layout demonstration | Control steps | Reached safely | Binary cost |
|---|---:|---:|---:|
| Unobstructed reach | 16 | yes | 0 |
| Straight path, obstacle present | 2 | no | 1 |
| Collision-free detour | 34 | yes | 0 |

Videos and native step records are in [environment_v1](environment_v1/).
True horizon is terminal (no bootstrap), and cost is binary. Geometry/energy
and one flow save/resume check passed; see [method changes](METHOD_CHANGES.md).
No all-history validation or new disaster-recovery system was introduced.

As of the initial Nov6 update, the first two ability learners are live:

- SAC seed2027, source0eba984, A100 GPU0: **30/30 successes at25k and50k**,
  fixed common layout seeds52027–52056, stochastic policy evaluation.
- Reward-only flow seed2027, sourcefc6b3c0, A100 GPU1: training continues
  toward100k; no held-out checkpoint result is claimed before it is written.

These are **unobstructed ability** results, not obstacle safety or a formal
seed comparison. The full100k budgets continue despite early successes;
80% is not a programmatic early-stop rule. Both remain the sole behavior
policy of their learner. [DEVELOPMENT_RESULTS.csv](DEVELOPMENT_RESULTS.csv)
labels interim checkpoints explicitly; other result files remain header-only.

One scheduling comparison uses actual productive post-warmup progress:
single SAC5k–10k:111.30 transitions/s; with both learners running,
SAC35k–45k:106.15/s and flow4k–8k:46.35/s, aggregate about152.50/s.
The workloads and sample windows differ: this is an actual heterogeneous
queue capacity comparison, **not** an algorithm speed comparison or a
controlled same-workload scalability claim. Keep two independent learners,
one per currently free A100; do not equate low VRAM use with permission to
duplicate scientific seeds or interrupt foreign jobs.

After the SAC100k ability run completes, one SAC-Lag2027 obstacle baseline
can fill that card,300k steps, with a fresh capacity/space check. This is a
specific queued baseline justified by the observed ability trend, not an
automatic direction matrix or a success-threshold gate. Flow-Iso scheduling
waits for flow ability evidence; Oracle/Learned training is not launched yet.

Cost-Q/dual, learned-margin and detached-geometry components are implemented,
but obstacle safety learning, Oracle integration, controls, formal runs,
payload extension and robustness remain pending. The full requested objective
is not complete. Large checkpoint/replay files remain on A100; compact native
JSON, progress and videos are copied through this host to the new GitHub branch.

See [task definition](TASK_DEFINITION.md), [method changes](METHOD_CHANGES.md)
and configs/star/arm_safe_entropy/development_v1.json for current settings.
All primary evaluations will use fixed final checkpoints, common layouts and
per-training-seed statistics. An unsuccessful experiment will be reported as
such, not replaced by a claim that finishing a budget proves effectiveness.
