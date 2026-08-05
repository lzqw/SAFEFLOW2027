# P4.2 B0/B1 backbone decision

The preregistered decision is `repair_backbone_and_pause_safety`. No safety-mechanism work or
continuation beyond 50k was launched from this result, and `final_p4_verdict` remains null.

## Protocol and provenance

- Training source: `f4371320a1683bc2130cd687213ffbac2d7da9d9`.
- Frozen analysis source: `14724f3d33f15fa458b0aa3d0ce68263b59eff70`.
- Arms: B0 disables global safety and local geometry; B1 enables global safety but disables local
  geometry.
- Budget: 50k transitions, 48k updates, seeds 2027--2029, one uninterrupted segment per run.
- External evaluation: checkpoints 0 and 50k, fixed-random-flow, fixed seeds 12027--12046, 20
  episodes per checkpoint per training seed.
- Six runs consumed 26,644.11 training seconds and an estimated 10.9742 CNY.
- All six final/full checkpoints contain exactly 50k replay transitions and matching counters.

## Result

| Arm | Return 0→50k | Success 0→50k | Stall 0→50k | Net displacement 0→50k | TD gate | Overall |
|---|---:|---:|---:|---:|---|---|
| B0 | -1.343→18.845 | 0.083→0.967 | 0.017→0.000 | 2.336→1.092 | fail (3/3 seeds) | fail |
| B1 | -1.343→0.290 | 0.083→0.117 | 0.017→0.217 | 2.336→0.840 | pass | fail |

B0 clearly learned PointGoal, but the frozen all-five gate failed. Every B0 seed had a strictly
rising final four reward-TD intervals and a final/minimum ratio above two. The original net
start-to-end displacement field also declined. B1 displayed the expected conservative-collapse
signature: little task progress and a large stall increase under the global cost/dual path.

The causal branch gives B0 failure priority, so B1's collapse is recorded but not repaired yet.
The next authorized work is the B0-R1 target-flow/delayed-actor repair specified in
`P4_2_BACKBONE_REPAIR_PROTOCOL.md`.

The compact machine-readable result is `reports/star/p4_2_backbone_decision.json`; large
checkpoints, replay buffers, evaluator work directories, and raw trajectories remain on the
server and are intentionally excluded from Git.
