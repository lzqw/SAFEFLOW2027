# SafeReach-Obstacle-v2 (simulation only)

Project task variant using panda-gym 3.0.7 and PyBullet 3.2.7; it is not
reported as an unmodified PandaReachDense or Safe-Panda benchmark.
The two A100 GPUs were idle at initial inspection. 4080/5080 other-project
loads are not interrupted. Independent environment: `/home/lzqw/SAFEFLOW2027_ARM/env`.

Panda has a fixed closed gripper and fixed target orientation. Normalized
actions in [-1,1]^3 specify world-frame EE displacements, 0.025 m per axis
per action unit. Control period is 0.04 s (25 Hz), with twenty 0.002 s physics
substeps. Commands are workspace-clipped to x[-.45,.10], y[-.35,.35],
z[.08,.55] m before inverse kinematics and real joint position-motor simulation.
These are simulation settings, not certified physical robot velocity limits.
Maximum commanded displacement per axis corresponds to 0.625 m/s; actual
motion depends on IK and joint motor response. Robot hardware is never addressed.

State: seven q/pi, seven qdot/4, world EE xyz, goal xyz, obstacle xyz,
obstacle radius and normalized remaining time (25 scalar observations).
All learners use the same state, control mapping, rewards and layouts.

Initial design: a 6 cm radius static sphere separates an opposite-side goal
and start, nominally (-.25,-.17,.25) and (-.25,.17,.25) m. Common small layout
translations are sampled per episode. Exact fixed-layout no-obstacle,
straight-line violation and safe-detour clips must be produced before training.
Start reset rejects an initial unsafe protected-link distance. Main state/goal
layout diversity may be revised as an explicitly versioned task for all methods,
not only the proposed learner.

Protected robot links: PyBullet link indices 2–11, including arm, wrist,
hand and fingers; table/base support contact is not an obstacle violation.
Every physics frame queries true signed robot-link/sphere closest distance,
including the initial frame. Complete the full twenty-frame control interval;
if its minimum is below 0.015 m OR contact occurs, terminate that control step
and emit cost 1. No subsequent control action is executed after a violation.
Other steps cost 0, so episode cost is binary. Contact and clearance violations
are reported separately. Clearance termination often precedes actual contact;
low contact rate alone must not be reported as low overall violation rate.
Local training margin label = .015 - minimum distance over the executed interval.

Horizon =100 control steps, with remaining time observed. Success, first
violation and real horizon all use `terminated=True`, `truncated=False`:
no bootstrap across the true finite-horizon boundary.

Reward = 5*(clipped previous goal distance - clipped next distance) - .01;
distances in this potential are capped at 1 m, success within .035 m adds 10,
violation subtracts 10 plus all remaining per-step charges, and an unsuccessful
safe horizon subtracts 2. The bounded potential can worsen by at most 5;
remaining step charges total at most 1 and timeout costs 2. The failure charge
therefore removes an incentive to terminate early simply to escape negative
per-step rewards. The same reward applies to every method.

Budget d_C=.10 means a target 10% **episode** violation rate, not cost limit10
and not a real-world permission. Oracle uses counterfactual PyBullet state
save/restore with the actual action-to-IK-to-motor mapping, not an unconverted
task-space normal. Its exact finite-difference implementation is diagnostic;
its query throughput is not learner throughput. An approximate cached controller
Jacobian is available and explicitly labelled. Both original development seeds
have completed300k and the bounded rho.50 revision has completed100k per seed.
Its errors are retained in the accuracy reports; it is not an error-free upper
bound. On the final10-policy shared pool, its mean signed cosine to the exact
interval-gradient axis is .935 near the boundary (108 valid comparisons), not1.

The existing Gym-style `SafeReachEnv.reset/step/close` boundary is the reserved
control-adapter interface. `step` accepts normalized world-frame EE commands;
the simulation adapter alone implements the mapping and interval labels above.
A future hardware adapter must supply independently calibrated limits, workspace,
contact-force monitoring and emergency stop under explicit hardware permission.
Neither this interface reservation nor the nominal .625m/s command scale is
permission to execute exploration on a real robot.

## Version correction and retained evidence

v1 read Bullet's default link-state cache while q/dq could already reflect the
next integrated frame; moving states differed by approximately 0.3–0.4 mm after
counterfactual restoration. v2 explicitly recomputes forward kinematics for every
EE observation/control/reward use. v1 also stopped physics at the violating
substep; v2 defines and labels a consistent complete 40 ms interval. Geometry,
reward, action scale, layout distribution and cost budget are otherwise unchanged.
All obstacle comparisons must share v2. v1 pilots and their weak outcomes remain
in the archive, marked superseded, never pooled with v2 efficacy results. These
API corrections do not establish the cause of any v1 learning failure.

The old unobstructed 100k SAC/flow policies each obtained 30/30 successes when
evaluated zero-shot on v2; no additional ability training was used for this check.
The three v2 clips take 16, 2 and 33 control steps respectively; see
[environment_v2](environment_v2/). The exact controller replica check found maximum
margin difference 3.7253e-8 m and normal cosine 1.0 for its six valid comparisons;
this is a bounded interface check, not proof of global oracle accuracy or efficacy.

Upstream platform references:
[panda-gym](https://github.com/qgallouedec/panda-gym),
[Panda Cartesian control](https://panda-gym.readthedocs.io/en/stable/robots/panda.html).

## Bounded relative-layout screening: SafeReach-RelativeLayout-v3

After the complete frozen-model obstacle evaluation showed no consistent
Learned robustness advantage, the new user instruction selects a task-only
extension, not9M more nominal training. Start, goal and obstacle positions are
drawn independently before shared feasibility rejection:

| Object | Center xyz (m) | Independent uniform jitter halfwidth xyz (m) |
|---|---|---|
| Start | (-.25,-.17,.25) | (.04,.02,.03) |
| Goal | (-.25,.17,.25) | (.05,.02,.03) |
| Obstacle | (-.25,0,.25) | (.045,.015,.025) |

These are not a shared scene translation or a midpoint-constrained obstacle.
Accepted layouts condition these proposals on the existing safe-start test
and a reachable, safe kinematic goal. The goal test saves/restores simulation
state during reset only; it is not an action filter, extra learned policy,
behavior-time oracle, reward change or teacher label. All methods use the same
sampler and costs. The25-state observation, 6cm sphere, first-violation full
interval, controller and100-step horizon remain unchanged. The original v2
sampler's random draw order and semantics remain unchanged by the layout hooks.

The single screening configuration uses fresh SAC-Lag/Iso/Learned models,
seed2027,300k each, total0.9M, common30 layouts82027..82056 at100k/200k/300k.
No new algorithm module, weight search, scientific early stop or automatic
budget extension is introduced. Formal seeds2030..2032 remain reserved.
