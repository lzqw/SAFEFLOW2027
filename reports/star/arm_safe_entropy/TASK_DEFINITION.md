# SafeReach-Obstacle-v1 (simulation only)

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
Every physics frame queries true signed robot-link/sphere closest distance.
At the first distance below 0.015 m OR contact, terminate and emit cost 1.
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
task-space normal. Its initial finite-difference implementation is diagnostic;
its training throughput must be measured before scheduling OracleDir runs.

Upstream platform references:
[panda-gym](https://github.com/qgallouedec/panda-gym),
[Panda Cartesian control](https://panda-gym.readthedocs.io/en/stable/robots/panda.html).
