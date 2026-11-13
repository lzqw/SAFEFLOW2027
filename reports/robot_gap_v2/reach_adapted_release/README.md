# Fixed adapted Reach endpoints: independent-layout confirmation

The matched Iso/EA adaptations each added exactly100K transitions to their
original method-matched300K parents. Each sole400K endpoint was fixed before
validation; no intermediate checkpoints or alternative seeds were selected.
After50 common validation episodes/condition, both model identities and the
unchanged original Iso-Low reference were frozen before independent testing.
The original15-candidate/2250-episode release remains preserved separately.

## Independent test:100 new layouts per condition

| Policy | Nominal success / violation | 55mm geometry success / violation | Scale0.85 success / violation |
|---|---|---|---|
| Iso400K |100% /0%|72% /28%|99% /0%|
| EA400K |100% /0%|70% /30%|100% /0%|
| Original Iso-Low120K |0% /0%|0% /0%|0% /0%|

Independent layout seeds71027–71126, one training seed41027, three conditions,
900 episodes total. Iso's one scale timeout is not a violation. Iso-Low did not
receive the additional100K adaptation and is a stagnant reference, not a matched
adaptation comparison. The primary Iso/EA comparison has equal new training
budget. No training-seed mean or statistically established2-point difference
is claimed. EA does not outperform Iso on this geometry test.

Validation was Iso100%/76%/96%, EA100%/72%/100%,50 layouts61027–61076 per condition.
Validation and independent results are retained as distinct rows in `results.csv`.
The two new endpoints consumed300 validation episodes, not another2250.

55mm retains the protocol name `new_geometry`, but is in the adaptation's
45/50/55mm training family. This confirms performance on independent layouts
under geometry adaptation, **not extrapolation to an unseen radius**. Native
safety distance and every physics/distance-query substep are unchanged.
28%/30% geometry violations fail the≤10%-per-condition release rule; all
physical-deployment-eligible/motion flags remain false. Zero hard collisions
would not remove these safety-distance violations.

## Actual offline artifacts

`bundles/{iso,iso-low,ea}` contains actual inference weights, frozen normalizers,
EA frozen EMA margin, action scale/model identity and integrity manifests.
These small inference bundles are not full recovery checkpoints.
`selection.json` was frozen before the first independent label and remains
bound by SHA256 in every test record and `receipt.json`.

All three actual bundle replays logged30 proposals,zero missed deadlines and
zero physical commands. The same original simulated observation fixture is
used; replay acceptance is not calibrated safety. Server CPU batch1 p50/p95:
Iso1.529/1.598ms, Iso-Low1.538/1.594ms, EA10.419/10.655ms. Each has20warmup+200
measurements, including complete action generation/tail/input gradients.
This is not timing on the actual deployment computer.

From the repository root, offline only:

```bash
python -m deployment.piper.replay_policy \
  --bundle reports/robot_gap_v2/reach_adapted_release/bundles/ea \
  --observations reports/robot_gap_v2/reach_release/observations.jsonl \
  --output /path/on/data-disk/new-adapted-replay.jsonl --device cpu
```

The original `reach_release/observations.jsonl` remains the source fixture.
`planned_onsite_reach90.json` remains an unexecuted, balanced on-site plan.
No CAN/SDK/physical transport is opened. TCP, frames, control period, workspace,
speed limits, deployment-computer latency and explicit operator authorization
remain pending.

New evaluation work:56,697 native controls,1,133,940 physics substeps and
1,193,040 distance queries;zero RL updates or physical robot interactions.
Raw per-episode validation/test JSONs stay on the A100 data disk under
`/storage_0/SAFEFLOW2027_lzqw_uid1007/robot_gap_v2/reach-adapted-release`.
Only compact results, real bundles and offline replay records are in Git.
