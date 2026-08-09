# STAR P4.6 final compact evidence bundle

This directory is the compact evidence mirror of the formal P4.6 experiment at
`seeta-nmb2:/root/autodl-tmp/SAFE/p4_6_direction_weight_pareto`.

- Protocol: `p4.6-direction-weight-pareto-v1`
- Protocol SHA-256: `c473da2ed99b31cde673a1cd038ec0477053b23ac63a130376ebccc4271ad93e`
- P4.6 base commit: `8791f21c41d11ace3021248e64646b75cb19fed2`
- Training source: clean commit `53e8c8ebdd0bea6404d56a31cc34db9d98cf5f81`
- Formal source directory: `/root/autodl-tmp/SAFE/code/SAFEFLOW2027-p4.6-53e8c8e`
- Reused baseline source: P4.5 clean commit `532d4b126f313d58e4e42d331ae6f8f672fb6e01`
- New matrix: C025/C050 at seeds 2027, 2028, and 2029; exactly 100,000 environment steps each
- Reused matrix: checksum-verified C0/C1 P4.5 evidence at the same three seeds; no reruns
- Evaluation milestones: 0, 25k, 50k, 75k, and 100k with fixed seeds 12027 through 12046
- Trajectory seeds: 12027, 12036, and 12046
- Scientific early stops: 0
- Strict conclusion: `partial_tradeoff_only`
- Selected arm / direction-loss weight: none / none

## Locked result

| arm | lambda | pooled episodes | return | success | mean cost | CVaR-20 | dangerous normal velocity | path | stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 0.00 | 60 | 0.730852 | 0.300 | 15.883333 | 70.083333 | 0.159242 | 3.365560 | 0.067 |
| C025 | 0.25 | 60 | -0.101114 | 0.283 | 17.116667 | 66.500000 | 0.163149 | 5.138978 | 0.100 |
| C050 | 0.50 | 60 | -0.134932 | 0.117 | 8.516667 | 41.750000 | 0.117940 | 3.211395 | 0.100 |
| C1 | 1.00 | 60 | 0.074086 | 0.233 | 8.616667 | 43.083333 | 0.136207 | 4.130243 | 0.117 |

C025 failed the safety gate: pooled mean cost increased by 7.765%, pooled CVaR-20 fell by
only 5.113%, and the paired-seed lower counts were 1/3 and 2/3. It also failed the performance
gate because the pooled return loss was 0.831966, although its success loss was only 0.016667.
Its locked seed-2028 path ratio (0.860) and seed-2029 stall increase (+0.050) passed robustness.

C050 passed safety: pooled mean cost and CVaR-20 fell by 46.380% and 40.428%, respectively,
and both were lower in 3/3 paired seeds. It passed the locked robustness checks (path ratio 0.934,
stall increase +0.050), but failed performance with return and success losses of 0.865784 and
0.183333. Therefore neither intermediate weight is a stable tradeoff, and P4.6 selects no final
PointGoal configuration for a four-environment expansion.

The authoritative human report and machine-readable classification are
`summary/P4_6_DIRECTION_WEIGHT_PARETO_REPORT.md` and `summary/p4_6_final_summary.json`.

## Evidence layout

- `runs/`: six receipts, exit codes, train logs, resolved configs, progress tables, all 5k
  diagnostics, held-out probability evidence, five fixed evaluations, counterfactuals, and raw
  trajectory CSVs;
- `audits/`: six deterministic 64-state x 256-action-sample A/B/C audit JSON files and logs;
- `summary/`: combined tables, checkpoint SHA-256 identities, the final report, four curve groups,
  the final Pareto plot, and three fixed-seed trajectory figures;
- `runtime/`: launch/supervisor evidence, matrix and post-processing completion markers, final logs,
  and the preserved failed first post-processing attempt;
- `checksums.sha256`: transfer seal for every artifact other than the manifest itself.

Full replay checkpoints are intentionally excluded from Git and remain on the server. Their paths,
byte sizes, and hashes are recorded in `summary/p4_6_checkpoint_sha256.csv`. No `.pt`, replay, or
audit work-directory file is present in this bundle.

## Post-processing launch repair

All six training runs completed before post-processing, and their files were not modified. The first
post-processing attempt exited with code 1 before the first audit because the absolute-path Python
invocation did not place the repository root on `sys.path`. Its launcher log, audit log, and exit code
are preserved in `runtime/`.

The retry exported the unchanged formal source directory as `PYTHONPATH` and otherwise reused the
same six checkpoints, audit seed, 64 x 256 sample, A/B/C modes, summary code, and protocol. It
completed all six audits and the summary with exit code 0. Commit
`9e3724b6b4fa211505d8cb2746b6ebe13e06ac85` makes that environment export explicit in the tracked
finalizer and adds a regression test; it does not change training or scientific analysis.

Run the following from this directory to verify the local mirror:

```bash
sha256sum --check checksums.sha256
```
