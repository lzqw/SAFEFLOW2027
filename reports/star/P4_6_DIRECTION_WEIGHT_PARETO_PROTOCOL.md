# STAR P4.6 Direction-Loss Weight Pareto Protocol

Status: locked before any P4.6 training.

Protocol SHA-256: `c473da2ed99b31cde673a1cd038ec0477053b23ac63a130376ebccc4271ad93e`.

P4.5 produced a cross-seed safety improvement at `lambda_dir=1`, but with an unacceptable
pooled return loss and seed-specific failure modes. P4.6 changes exactly one scientific variable:
the scalar direction-loss weight. It does not change the algorithm, critics, safety band,
covariance, execution path, evaluation randomness, or environment.

## Arms and immutable controls

- `C0`: sealed P4.5 result, direction loss disabled (`lambda_dir=0` conceptually); never rerun.
- `C025`: new run, direction loss enabled with `lambda_dir=0.25`.
- `C050`: new run, direction loss enabled with `lambda_dir=0.50`.
- `C1`: sealed P4.5 result, direction loss enabled with `lambda_dir=1`; never rerun.

The P4.5 final summary and seed table are bound by SHA-256 in the machine-readable protocol.
Candidate configs must be identical except for their comments and weight, and may differ from the
P4.5 C1 config only in comments, protocol metadata, and weight. Algorithm/model files must remain
byte-identical to base commit `8791f21c41d11ace3021248e64646b75cb19fed2`.

## Formal schedule

Each new arm runs seeds 2027, 2028, and 2029 for exactly 100,000 environment steps. Batch 1 runs
the four seed-2027/2028 jobs concurrently; batch 2 runs the two seed-2029 jobs concurrently.
There is no scientific early stop. Only the already-defined engineering failures may stop a run.

Checkpoints remain 0/10k/25k/50k/75k/100k. Evaluations remain 0/25k/50k/75k/100k with the
same 20 seeds 12027--12046 and identical flow randomness. Trajectories remain fixed at seeds
12027, 12036, and 12046. Each new final checkpoint receives the unchanged 64-state by
256-action-sample A/B/C audit with audit seed 4500007.

## Locked decision rule

A candidate retains safety only if, relative to sealed C0, pooled mean cost and CVaR-20 are each
at least 10 percent lower and each is lower in at least two of three paired seeds. It retains
performance only if pooled return and success are each no more than 0.10 below C0. The targeted
robustness gate additionally requires seed-2028 path length to be no more than 1.5 times its C0
value and seed-2029 stall to be no more than 0.10 above its C0 value.

The allowed conclusions are `select_C025`, `select_C050`, `partial_tradeoff_only`, and
`no_stable_tradeoff`. If both candidates pass all gates, C025 is preferred as the lower
intervention unless C050 lowers both pooled mean cost and CVaR-20 by at least another 10 percent
relative to C025 while retaining every performance and robustness gate.

All raw seed results and the complete four-weight Pareto plots are reported regardless of the
label. The four-environment experiment is explicitly out of scope for this run; it becomes the
next stage only if P4.6 selects a final PointGoal weight.
