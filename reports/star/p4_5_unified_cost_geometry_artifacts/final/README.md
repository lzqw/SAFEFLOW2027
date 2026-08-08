# STAR P4.5 final compact evidence bundle

This directory is the compact evidence mirror of the formal P4.5 experiment at
`seeta-nmb2:/root/autodl-tmp/SAFE/p4_5_532d4b1`.

- Protocol: `p4.5-unified-cost-geometry-v1`
- Protocol SHA-256: `8abebc101021ec71a00fdf6d2c5580b5f97dada710e58de00936a75490e85c11`
- Training source: clean commit `532d4b126f313d58e4e42d331ae6f8f672fb6e01`
- Formal source directory: `/root/autodl-tmp/SAFE/code/SAFEFLOW2027-p4.5-532d4b1`
- Matrix: C0/C1 at seeds 2027, 2028, and 2029; exactly 100,000 environment steps each
- Evaluation milestones: 0, 25k, 50k, 75k, and 100k with fixed seeds 12027 through 12046
- Trajectory seeds: 12027, 12036, and 12046
- Scientific early stops: 0
- Strict conclusion: `safety_benefit_with_performance_cost`

## Locked result

| arm | pooled episodes | return | success | mean cost | CVaR-20 | dangerous normal velocity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 60 | 0.730852 | 0.300 | 15.883333 | 70.083333 | 0.159242 |
| C1 | 60 | 0.074086 | 0.233 | 8.616667 | 43.083333 | 0.136207 |

C1 lowered mean cost in 3/3 paired seeds, CVaR-20 in 3/3, and dangerous normal velocity in
2/3; every pooled safety value was also lower. The safety-majority condition therefore passed.
The pooled return loss was 0.656766, above the preregistered 0.10 limit, while the success-rate
loss was 0.066667. Performance was not maintained, producing the locked conclusion above.

The authoritative human report and machine-readable classification are
`summary/P4_5_UNIFIED_COST_GEOMETRY_REPORT.md` and `summary/p4_5_final_summary.json`.

## Evidence layout

- `runs/`: six receipts, train logs, resolved configs, progress tables, all 5k diagnostics,
  held-out probability evidence, five fixed evaluations, counterfactuals, and raw trajectory CSVs;
- `audits/`: six deterministic 64-state x 256-action-sample A/B/C audit JSON files and logs;
- `summary/`: combined tables, checkpoint SHA-256 identities, the final report, four curve groups,
  and three fixed-seed trajectory figures;
- `runtime/`: supervisor and post-processing logs, completion markers, failed-attempt provenance,
  the final watcher, and the exact runtime summary implementation;
- `checksums.sha256`: transfer seal for every artifact other than the manifest itself.

Full replay checkpoints are intentionally excluded from Git and remain on the server. Their paths,
byte sizes, and hashes are recorded in `summary/p4_5_checkpoint_sha256.csv`. No `.pt`, replay, or
audit work-directory file is present in this bundle.

## Engineering post-processing repairs

Training and all six audits completed without scientific intervention. Two post-processing edge
cases were repaired after training:

1. the first watcher used a path-wide `seed-*` match and counted each run's `artifacts/` and
   `checkpoints/` children; the final watcher matches only directory basenames;
2. C1 seed 2029 had constant zero final episode costs, so its Spearman diagnostic is mathematically
   undefined and serialized as JSON `null`; the summary now preserves that diagnostic as null/CSV
   empty while continuing to require every primary metric to be finite.

The report template was also corrected to name `p4_5_checkpoint_sha256.csv` exactly. The deployed
runtime summarizer is byte-identical to the tracked implementation at transfer time, SHA-256
`122914eefbdbf759975fdd9b32f8cc6fac0c49c7af1fd2eecc0fc4bf74a8aff5`. The second and third
complete audit executions produced identical hashes for all six JSON files; the earlier set is
sealed in `runtime/audit_sha256_attempt2.txt`.

Run the following from this directory to verify the local mirror:

```bash
sha256sum --check checksums.sha256
```
