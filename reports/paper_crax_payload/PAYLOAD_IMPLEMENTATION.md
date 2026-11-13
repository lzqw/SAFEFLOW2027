# SafePayloadTransport-v1: implementation and bounded engineering evidence

Status: environment and three-method training runner implemented; initial six
300k development trials are not claimed complete by this implementation report.
No old Reach weights are transferred, and historical code/reports are unchanged.

## Task semantics

`scripts/paper_crax_payload/payload_environment.py` registers
`SafePayloadTransport-v1`. The robot already holds a dynamic50g box, half extents
25×18×20mm, using a fixed Bullet grasp constraint on Panda's EE link. This is not
Reach renamed: collision geometry, actual transported-body centroid reward,
payload observations, inertia, constraint dynamics and restoration are explicit.
The policy commands three EE displacement coordinates,25mm full scale/control,
with the inherited fixed attitude Rx(pi),40ms control period and100step horizon.
Start, goal and obstacle are independently randomized with the shared historical
relative-layout distribution. Reset validates both arm/payload start and reachable
goal clearance before training; the object is placed with the robot before reset
settling rather than dragged across the scene by a stale grasp constraint.

The label is `safety_distance - min(arm_distance,payload_distance)` over the
initial configuration and every one of20 physical substeps of the real action
interval. Both signed distances and both collision classes are reported. First
clearance violation/collision terminates with cost1; all other interval costs0.
Only hand/gripper–payload grasp contacts on links8–11 are exempted. This does not
exempt either the robot or payload from obstacle collisions. There is no action
reranking, oracle-selected policy action, protected physical robot, or learned
grasp in this task.

Every method receives the same45D observation: native25 plus payload world
position3, quaternion4, half extents3, mass1, linear/angular velocity6 and payload
goal3. Reward progress and goal completion use the actual payload centroid, not
the gripper's distance while ignoring its object. The target marker is also the
payload goal. All pose/dimension/velocity additions use explicit SI units.

## Learners, seeds, retained checkpoints

`payload_models.py` reuses the existing Gaussian SAC/TwinQ, stable Flow actor,
binary-cost critic, margin ensemble and true injection-site pullback. Iso and EA
share the existing stable learner; the only added diagnostic correction uses the
actual payload-to-goal distance in the reward-Q upper-bound diagnostic. The SAC
wrapper matches the stable critic settings, critic-only preheat and reward-update
count while retaining the Gaussian entropy objective rather than disguising it as
Flow reference energy. All three methods share task/cost budget0.1, width128,
actor LR1e-4, critic LR3e-4, batch256 and UTD1 in the initial configuration.
This is the `episodic_binary` task mode; it is explicitly not the CRAX cumulative
cost model or its budget25. New configuration is `payload_v1.json`.

Development seeds41027/41028, confirmation seeds51027/51028/51029 agree with the
master registration. All development policies start fresh. Each method gets the
same five checkpoint fractions. Initial300k checkpoints are60k/120k/180k/240k/300k;
the fixed endpoint is retained even if an earlier checkpoint validates better.
Confirmation0.6M/optional shared1M is controlled by the master protocol, not by an
automatic success-triggered extension in this runner. Overall budget admission
belongs to the coordinator's ledger/finite queue; this runner does not claim a
distributed global budget lock.

`payload_train.py` saves learner, all optimizers, replay, RNG, margin EMA, episode
returns/costs/lengths and both robot and payload observed state. Native recovery
checks exact configuration/method/seed and rejects a matching still-live parent.
Restoration neither resets partial episodes nor advances physics. Bullet solver
warm-start caches are not claimed bitwise portable. Each source module SHA256 is
recorded, including for the pre-commit engineering smoke. Large full replay files
remain on4080's4T filesystem. Failures are preserved and not automatically retried.

## Actual engineering evidence

4080 project: `/home/carla/4T/SAFEFLOW2027_PAPER/project`.
Native unchanged Python: `/home/carla/4T/SAFEFLOW2027_ARM/env/bin/python`.

- Four local focused contracts passed1.52s. The isolated local panda-gym3.0.7
  package was installed under `/tmp`, not into the user's base environment.
- The same four native4080 contracts passed1.59s, including an actual
  payload-only collision while arm-obstacle collision distance stays positive,
  21sample interval minima and full payload/robot partial-state restoration.
  An initial pytest invocation failed before test collection through unrelated
  typeguard autoload. Retrying with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` fixed test
  discovery without installing/upgrading dependencies; that failed invocation
  performed zero physics/learning.
- One identical constructive high detour succeeded safely in each of five
  candidate conditions: nominal, radius7cm, wider payload, action-scale0.85,
  and offset obstacle. These are **controller feasibility checks**, not learned
  policy scores. Actual cost206 non-training control intervals. This proves one
  alternate path exists; it does not yet prove the offset blocks a learned
  policy's commonly used route. That interpretation requires later route evidence.
- A genuine EA engineering run collected3210 transitions,1210 learner rounds,
  2420 reward-critic updates and105 actor updates. A saved full checkpoint then
  continued3210→3220 with10 additional transitions,1220/2440/110 updates. This
  includes actual random warmup and critic-only preheat; it is not zero-cost
  preparation. It is not a300k policy result or an active-geometry performance
  benchmark: this early smoke is before the geometry ramp.
- Initial smoke elapsed17.83s including its data loop; cumulative average180SPS
  includes2000 fast random transitions and is not steady EA learning throughput.
  The10step resumed segment is too short for a speed claim. Record real long-run
  steady throughput separately.

Compact raw receipts, manifests, progress and native JUnit evidence are under
`payload-engineering-v1/`. Full checkpoints remain server-only. Actual counters:

| Item | Actual new training | Non-training control intervals |
|---|---:|---:|
| Local exploratory reset/step and next-interval probe |0|6|
| Local focused contracts |0|5|
| Native focused contracts |0|5|
| Native common constructive route,5conditions |0|206|
| Native EA engineering run |3210|0|
| Native full continuation |10|0|
| Total to this report |3220|222|

Physical robot interaction remains0. Synthetic learner unit-test tensors do not
represent environment transitions. Control intervals count real `env.step` or
oracle `_advance` calls; reset-settling physical substeps are not relabelled as
training transitions. Long-run metrics, inference and simulations are not inferred
from these semantic checks.

## Native training command

After the coordinator commits/pins source and admits a slot:

```bash
cd /home/carla/4T/SAFEFLOW2027_PAPER/project
CUDA_VISIBLE_DEVICES=0 /home/carla/4T/SAFEFLOW2027_ARM/env/bin/python \
  -m scripts.paper_crax_payload.payload_train \
  --config configs/paper_crax_payload/payload_v1.json \
  --method sac_lag --seed 41027 --steps 300000 --device cuda:0 \
  --source-commit EXACT_COMMIT --output NEW_RUN_DIRECTORY
```

Use `s_iso` or `s_ea` for the other methods, both registered seeds, and a distinct
output per trial. No checkpoint/seed substitution is performed. A slots-free
finite queue may start the next declared run immediately; it must not stop
foreign tasks, use5080, or retry failed training invisibly.
