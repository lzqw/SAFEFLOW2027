# Native Piper simulation training and candidate export

Status: source implemented; two constructive task routes and three focused
contracts passed. No six-trial development training has been launched by this
implementation. One explicitly approved3210→3220 engineering continuation is
pending a source commit at this report revision. No physical robot transport or
motion is enabled, and an engineering checkpoint is not a deployment candidate.

## Model and genuine task feasibility first

The user identified standard Piper and standard gripper. The trainer loads the
official pinned URDF/meshes through `deployment.piper.simulation`; it never loads
Panda weights. Model `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`, source URDF SHA256
`c1a5c29ae9b91160557ed02bddd47365a71f0eb1653abb4997eb34f124021486`.
TCP calibration, firmware and physical limits remain pending. These are simulated
values, not calibrated hardware settings or an authorization.

The provisional simulator's200-iteration IK initially stopped124.6mm short when
checking the cross-workspace goal and placed links inside the obstacle. The
failed static gate is preserved in `piper-constructive-feasibility-v1.json`.
Offline repeated seed updates converged to the same requested goal; this was
solver convergence, not proof the official robot could never reach it.

The NEW training-only wrapper uses the same kinematic solver with1200 maximum
iterations, without changing the URDF, start/goal, obstacle, reward scale, action
scale or control period. It adds common static start/goal clearance and reachable
endpoint rejection, never actor-performance selection or action reranking. The
same predeclared high detour then actually completed Reach and Payload in58
control intervals each, with minimum44.6mm clearance and zero violations. See
`piper-constructive-feasibility-v2.json`, including actual trajectories. This is
feasibility evidence, not learned-policy performance. A stationary zero-action
smoke is explicitly not used as proof of learnability.

`piper_helpers.py` wraps, but does not edit, the deployment simulator. Payload
reward/completion uses its actual centroid, not just the tool pose. Both tasks
retain the explicit34D observation, fixed attitude, normalized deltaXYZ in robot
base,10mm full action scale,40ms control interval,100step horizon, full arm AND
payload interval-clearance costs and shared binary budget0.1. Allowed grasp
contacts alone remain exempt. All methods use the same task and admissibility.

## Native methods and variance/energy units

`piper_train.py` subclasses the stable learner with observation34. All weights
start fresh. Margin/target/replay/dual/geometry handling is native Piper data, not
relabelled Panda checkpoints. The optional Q-bound diagnostic uses Piper
TCP12:15 and goal15:18 instead of the parent's Panda indices; it never clips Q or
changes an objective.

| Control | Actual actor AND target variance | Energy target | Initial alpha |
|---|---:|---:|---:|
| Iso |.04|18.75|.0008|
| Iso-Low |.01|75|.0002|
| EA |.04|18.75|.0008|

Iso-Low changes both sampling and inverse-reference energy consistently. It is
not trace-matched to Iso/EA and is not the same checkpoint renamed. The same
scalar alpha parameter and optimizer are kept; settings/native variance enforce
exact resume identity. Frozen EMA geometry is exported for EA. Native geometry
ramp remains the shared10k→30k transition schedule for these simulated Arm tasks.

Configuration `piper_sim_v1.json`: two tasks ×three methods ×one common seed41027,
planned300k each =1.8M, subject to the total3M Piper simulation allowance including
engineering. Main coordinator admits jobs only when appropriate shared resources
are available; the trainer does not claim a global distributed budget lock.
All methods retain five equal fractions and their fixed final endpoint. Fresh
validation61027 and independent test71027 ranges are distinct from development
650000 and engineering layouts; independent testing has not been consumed here.

## Full continuation and frozen evaluation

Full checkpoints retain Actor/targets/critics/margin EMA/all optimizers/replay,
Python/NumPy/Torch CPU/CUDA RNG, partial episode returns/costs/lengths, all six arm
and two finger joint positions/velocities, payload pose/velocities, task fields
and environment RNG. Observed reconstruction makes no simulation step and does
not reset partial episodes. Next action replaces motor targets. Bullet solver
warm-start caches are not claimed bitwise portable.

Three focused tests passed1.74s: one tests all three models' actual covariance
trace, energy/alpha units, frozen input-gradient sampling and native state load;
two verify Reach/Payload joint/finger/payload observed restore and the next real
control interval. Physics6control intervals total, not4 cumulative counters per
task: a restored cumulative counter is not a newly executed query. JUnit:
`piper-focused-training-contracts-v1.xml`. Synthetic learner tensor updates are
not environment training transitions.

Frozen evaluation isolates training random streams, changes no optimizer/dual,
and reports success, violation, timeout, arm/payload collisions, return, minimum
clearance, capped and successful completion time separately for nominal/new
geometry/action-scale0.85. These are not yet measured candidate scores.

## Commands and provenance

All source modules and official simulator digest are recorded; source commit
must be pinned by the coordinator before a real run. Example native training:

```bash
python -m scripts.paper_crax_payload.piper_train train \
  --config configs/paper_crax_payload/piper_sim_v1.json \
  --assets-receipt PINNED_EXTERNAL_ASSET_RECEIPT.json \
  --task Payload --method Iso-Low --seed 41027 --steps 300000 \
  --device cuda:0 --source-commit EXACT_COMMIT --output NEW_4T_RUN
```

Use `Reach`/`Payload`, methods `Iso`/`Iso-Low`/`EA`. Native continuation uses the
same arguments and `--resume NATIVE_FULL.pt` in a new output directory. Engineering
uses `--engineering` and a total target at most10000; those transitions count
against the3M allowance, not zero-cost setup.

```bash
python -m scripts.paper_crax_payload.piper_train evaluate \
  --config configs/paper_crax_payload/piper_sim_v1.json \
  --assets-receipt PINNED_EXTERNAL_ASSET_RECEIPT.json \
  --checkpoint NATIVE_FULL.pt --split validation --output NEW_EVAL.json
```

`--split independent_test` is reserved for policies selected using validation.
An engineering split is limited to two nominal episodes and has separate layout
seeds. No test has been relabelled as validation. A final candidate requires the
same validation rule/checkpoint opportunity for all methods; otherwise its
Pareto tradeoff must be explicitly labelled.

```bash
python -m scripts.paper_crax_payload.piper_train export \
  --config configs/paper_crax_payload/piper_sim_v1.json \
  --assets-receipt PINNED_EXTERNAL_ASSET_RECEIPT.json \
  --checkpoint SELECTED_NATIVE_FULL.pt \
  --training-receipt MATCHING_NATIVE_TRAINING_RECEIPT.json \
  --selection-record COMMON_RULE_SELECTION.json --output NEW_BUNDLE
```

Selection records must identify Piper/task/method/checkpoint steps, validation
selection, no independent-test use, and either risk qualification or explicit
`pareto_tradeoff_not_risk_qualified`. Milestone training receipts are saved for
eligible intermediate checkpoints as well as the final native receipt. Export
checks actual method/source/model/step identity and delegates integrity/versioned
Actor/frozen margin/normalizer/scaling packing to `deployment.piper.bundle`.
Engineering checkpoints are rejected as deployment candidates. All exports keep
`physical_motion_enabled=false` and `hardware_matched=false`; there is no SDK/CAN
connection. Actual candidate generation awaits real training and common selection.

## New query accounting before approved engineering training

| Item | New train transitions | Non-training controls | Physics substeps | Scene distance evaluations |
|---|---:|---:|---:|---:|
| Failed static gate and offline IK diagnosis |0|0|0|7|
| Actual two-task constructive detour |0|116|2320|2440|
| Two restore/next-interval focused contracts |0|6|120|130|
| Total at this revision |0|122|2440|2577|

Distance evaluations include reset/goal static and interval samples; they are
not additional control transitions or training steps. Earlier deployment-agent
Piper8control/160substep smoke is separate and must not be double-counted here.
Physical robot interactions remain0. Long-run candidate performance, completed
1.8M simulation training, export quality and real-hardware deployment are not
claimed from these implementation checks.
