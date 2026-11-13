# Completed Piper transfer and geometry adaptation

`piper_new_learning_curve.csv` retains all36 condition/checkpoint rows from
the first four completed branches; `piper_new_efficiency.csv` records exact
new transitions, elapsed time, stage timing, update counts and new violations.
`receipts/` contains the corresponding genuine completion receipts. Raw
episode trajectories remain on the A100 data disk and in the local source
mirror, not Git. No simulation or training was added to produce these files.

![Learning and adaptation](piper_learning_and_adaptation.png)

One training seed41027,30 fixed development episodes/condition. No independent
test, training-seed uncertainty or physical safety claim. The original Reach
300K comparison is from the same development protocol, not the separately
selected50-episode common-validation candidates.

Transfer learns but regresses: Iso60K safe success was100% nominal,96.67% new
geometry and100% scale0.85; at100K it is50%/3.33%/56.67%. EA100K is
66.67%/10%/70%. Retain all checkpoints and do not relabel an intermediate maximum
as a fixed-budget result. EA's100K training violations366 exceed Iso226.

Equal100K Reach radius adaptation improves new-geometry success from30→80%
(Iso) and33.33→73.33% (EA). Both nominal/scale reach100%; geometry violations
remain20%/26.67%, above the prior release's10% risk qualification threshold.
There is no universal EA advantage or physical deployability demonstrated.
The condition retains its original name `new_geometry`, but55mm is included
in the adaptation training-radius family. This is geometry-distribution repair,
not extrapolation to an unseen radius. Independent-layout tests are separate.

The planned continuation uses each fixed100K Payload endpoint, with200K more
per method and identical full-state resume. No curriculum, geometry change or
additional candidate search. Its results will be appended only when complete.

Regenerate from retained input records:

```bash
python -m scripts.paper_crax_payload.robot_gap_results \
  --runs /path/to/campaign_runs \
  --old-finals reports/robot_gap_v2/piper_existing_finals.csv \
  --output reports/robot_gap_v2/piper_results
```

Complete segment SPS includes simulation, replay/learner, evaluation and save.
Engineering profile SPS elsewhere excludes its separate post-profile eval;
shared GPU load and different optimizer-update counts preclude an exact
optimization-equivalence or hardware-only speed claim.
