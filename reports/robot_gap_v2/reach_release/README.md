# Original Piper Reach: common validation and offline release

Completed on November13,2026. This release uses the original candidates;
the separately queued100K/method geometry adaptations are not included.
All results are simulation/offline. No physical robot connection was opened.

## Selection and limitations

Each of Iso,Iso-Low,EA received five original checkpoints (60/120/180/240/300K)
and50 common validation episodes per condition, seeds61027–61076. Total2250
episodes, one training seed41027. The frozen rule first requires≤10% violation
in every condition, then compares macro safe success and capped time; if no
candidate qualifies, it returns a labelled Pareto tradeoff. No independent-test
data selected the policies.

| Method | Steps | Nominal success/violation | New geometry success/violation | Scale0.85 success/violation | Status |
|---|---:|---|---|---|---|
| Iso |180K|94%/6%|28%/72%|86%/14%|Not risk qualified|
| Iso-Low |120K|0%/0%|0%/0%|0%/0%|Risk-qualified but stagnant|
| EA |300K|98%/2%|24%/76%|88%/12%|Not risk qualified|

EA has strong nominal task success, but neither active method meets the
cross-condition risk criterion. Iso-Low's lack of violations comes with no
task success. None is marked eligible for physical deployment. Fixed300K
development scores elsewhere remain unchanged and are a different comparison.

## Delivered evidence

- `validation_results.csv`:45 condition rows covering all15 candidates.
- `selection.json` and `selection-*.json`:fixed rule, scores and selected identities.
- `bundles/{iso,iso-low,ea}/`:actual small inference weights, frozen normalizer,
  action scale, official-model identity and SHA256 manifests. These are not full
  recovery checkpoints; the EA export includes its frozen EMA margin.
- `observations.jsonl`:one simulated EA development rollout fixture,30 controls,
  600 physics substeps and632 distance queries; not a physical recording.
- `offline-*.jsonl`:actual bundle replay with raw actions, bounded proposals,
  timing and timeout records. Each has30 accepted proposals,zero deadline
  misses andzero physical commands. Acceptance does not certify physical safety.
- `latency.json`:server CPU batch1 end-to-end action generation;1thread,
  20warmup+200measured actions. Includes normalization, integration, geometry,
  tail/input gradients and host return; excludes observation acquisition,
  transport and actuation. Iso p50/p951.510/1.569ms, Iso-Low1.500/1.576ms,
  EA10.235/10.432ms. This is not deployment-computer latency.
- `planned_onsite_reach90.json`:unexecuted balanced3methods×3conditions×10 plan.
- `receipt.json`:completed offline preparation,zero training transitions and
  zero physical robot interactions.

Large validation episode JSONs remain at
`/storage_0/SAFEFLOW2027_lzqw_uid1007/robot_gap_v2/reach-release` on A100.
They were not copied into Git. No validation episodes were repeated for export.

## Offline replay

From the repository root in the established Torch environment:

```bash
python -m deployment.piper.replay_policy \
  --bundle reports/robot_gap_v2/reach_release/bundles/ea \
  --observations reports/robot_gap_v2/reach_release/observations.jsonl \
  --output /path/on/data-disk/new-offline-replay.jsonl \
  --device cpu
```

The replay code emits proposals only and has no hardware transport. TCP, frames,
period, speed/workspace limits and actual deployment-machine latency require
on-site confirmation before any separate physical activation.
