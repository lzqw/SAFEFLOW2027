# Completed Piper transfer and geometry adaptation

`piper_new_learning_curve.csv` retains all48 condition/checkpoint rows from
five completed branches, including the Iso300K continuation;
`piper_new_efficiency.csv` records exact
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

The continuation uses each fixed100K Payload endpoint, with200K more per
method and identical full-state resume. No curriculum, geometry change or
additional candidate search. Iso is complete: its300K endpoint has100% nominal,
10% geometry and66.67% scale safe success, with0%/90%/33.33% violations.
Its geometry success was66.67% at120K,43.33% at180K and56.67% at240K;
the poor300K endpoint is preserved rather than silently replaced. Nominal
learning has stabilized more than robustness. The200K continuation took
3019.12s (66.24SPS) with456 new training violations; cumulative Payload
training violations are682 including the first100K. These are development
results, not the pending common-validation/independent closure.

EA's continuation was still live at239K/300K at02:19+08 November14. Its
partially completed segment is deliberately not folded into this completed-run
efficiency/curve table. The figure uses identical x-axis scales for both
methods; missing later EA points mean pending results, not zero success.
Do not compare the two methods as if both300K endpoints were already available.

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
