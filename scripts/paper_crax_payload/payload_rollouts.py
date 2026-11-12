"""Frozen Payload rollouts with optional private action/initial-state witnesses."""
from pathlib import Path
import json


class FrozenRolloutInterrupted(RuntimeError):
    def __init__(self, error, receipt):
        super().__init__(str(error))
        self.receipt = receipt


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def assert_initial_robot_payload(env, state):
    """Explicit gripper/joint/body checks in addition to the policy observation."""
    import numpy as np
    for joint, position, velocity in zip(state['joints'], state['positions'], state['velocities']):
        actual=env.client.getJointState(env.body,joint)
        np.testing.assert_allclose(actual[:2],[position,velocity],rtol=0.,atol=2e-6)
    position,orientation=env.client.getBasePositionAndOrientation(env.payload_body)
    linear,angular=env.client.getBaseVelocity(env.payload_body)
    for actual,expected in ((position,state['payload_position']),(orientation,state['payload_orientation']),
                            (linear,state['payload_linear_velocity']),(angular,state['payload_angular_velocity']),
                            (env.goal,state['goal'])):
        np.testing.assert_allclose(actual,expected,rtol=0.,atol=2e-6)


def evaluate_frozen_rollouts(learner, config, episodes, seed_start, conditions,
                             trace_directory=None, trace_episodes=0, provenance=None):
    import numpy as np
    import torch
    from scripts.paper_crax_payload.payload_environment import (
        PayloadConfig, SafePayloadTransport, capture_payload_environment)
    from dataclasses import asdict
    from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng
    results = []
    with isolated_rng():
        for ci, condition in enumerate(conditions):
            torch.manual_seed(config['evaluation']['action_sampling_seed']+ci)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config['evaluation']['action_sampling_seed']+ci)
            task = dict(config['task']) | condition.get('task_overrides', {})
            if 'obstacle_offset' in condition:
                task['obstacle_offset'] = condition['obstacle_offset']
            env = SafePayloadTransport(PayloadConfig(**task))
            records = []
            try:
                for index in range(episodes):
                    trace_path = None
                    if index < trace_episodes:
                        if trace_directory is None:
                            raise ValueError('Explicit external trace directory required')
                        trace_path = Path(trace_directory)/(condition['name']+f'-{seed_start+index}.jsonl')
                        if trace_path.exists():
                            raise FileExistsError('Preserve partial/completed action witnesses before any new physics')
                        trace_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        observation, initial = env.reset(seed=seed_start+index)
                    except RuntimeError as error:
                        # A declared held-out layout can be physically
                        # unconstructable even after the environment's bounded
                        # layout attempts. Preserve it as an explicit test
                        # failure instead of aborting the whole locked split or
                        # silently changing the seed/layout distribution.
                        records.append(dict(
                            layout_seed=seed_start+index,
                            episode_return=0.0,
                            length=0,
                            minimum_distance_m=0.0,
                            trajectory_payload_xyz=[],
                            action_witness_file=None,
                            time_to_goal_seconds=None,
                            time_to_goal_capped_seconds=env.config.horizon*env.control_dt,
                            safe_success=False,
                            violation=True,
                            collision=False,
                            arm_collision=False,
                            payload_collision=False,
                            clearance_violation=True,
                            true_horizon_terminal=False,
                            reset_failure=True,
                            reset_failure_type=type(error).__name__,
                            reset_failure_message=str(error)))
                        continue
                    stream = trace_path.open('x') if trace_path else None
                    if stream:
                        stream.write(json.dumps(jsonable(dict(kind='initial_state', layout_seed=seed_start+index,
                            condition=condition, task_configuration=asdict(env.config),
                            initial_state=capture_payload_environment(env), provenance=provenance)))+'\n')
                        stream.flush()
                    total, minimum = 0., initial['minimum_distance']
                    trajectory = []
                    try:
                        for step in range(env.config.horizon):
                            raw_action = learner.action(observation[None])[0]
                            executed = np.clip(raw_action*condition.get('action_scale', 1.), -1., 1.)
                            before = observation.copy()
                            observation, reward, done, truncated, info = env.step(executed)
                            total += reward; minimum = min(minimum, info['minimum_distance'])
                            trajectory.append(env.payload_position().tolist())
                            if stream:
                                stream.write(json.dumps(jsonable(dict(kind='executed_interval', step=step+1,
                                    raw_action=raw_action, executed_action=executed,
                                    observation_before=before, observation_after=observation,
                                    tcp_world_xyz=env.ee_position(), reward=reward, info=info)))+'\n')
                                stream.flush()
                            if done or truncated:
                                break
                    finally:
                        if stream:
                            stream.close()
                    records.append(dict(layout_seed=seed_start+index, episode_return=total, length=step+1,
                        minimum_distance_m=minimum, trajectory_payload_xyz=trajectory,
                        action_witness_file=str(trace_path) if trace_path else None,
                        time_to_goal_seconds=(step+1)*env.control_dt if info['safe_success'] else None,
                        time_to_goal_capped_seconds=((step+1) if info['safe_success'] else env.config.horizon)*env.control_dt,
                        **{k: info[k] for k in ('safe_success', 'violation', 'collision', 'arm_collision',
                            'payload_collision', 'clearance_violation', 'true_horizon_terminal')}))
            except BaseException as error:
                raise FrozenRolloutInterrupted(error, dict(status='evaluation_interruption',
                    error_type=type(error).__name__, error=str(error), provenance=provenance,
                    completed_conditions=results, partial_condition=condition,
                    completed_episode_records=records,
                    physical_evaluation_transitions=sum(r['physical_evaluation_transitions'] for r in results)+env.physical_control_intervals,
                    started_interval_charged_on_exception=True,
                    training_transitions=0, adaptation_updates=0, automatic_retry=False)) from error
            finally:
                env.close()
            successful = [r['time_to_goal_seconds'] for r in records if r['safe_success']]
            results.append(dict(condition=condition['name'], episodes=episodes,
                safe_success_count=sum(r['safe_success'] for r in records),
                violation_count=sum(r['violation'] for r in records),
                safe_timeout_count=sum(not r['safe_success'] and not r['violation'] for r in records),
                reset_failure_count=sum(bool(r.get('reset_failure', False)) for r in records),
                physical_evaluation_transitions=sum(r['length'] for r in records),
                time_to_goal_seconds=float(np.mean(successful)) if successful else None,
                **{key: float(np.mean([r[key] for r in records])) for key in ('safe_success', 'violation',
                    'collision', 'arm_collision', 'payload_collision', 'episode_return',
                    'minimum_distance_m', 'time_to_goal_capped_seconds')}, records=records))
    return dict(task='SafePayloadTransport-v1', conditions=results, training_transitions=0,
        adaptation_updates=0, training_rng_preserved=True,
        physical_evaluation_transitions=sum(r['physical_evaluation_transitions'] for r in results),
        oracle_used_for_action_selection=False, reset_settling_not_relabelled_as_training=True)


def replay_open_loop_witness(witness_path, env, offset):
    """Real simulator interval labels under offset, NEVER actions chosen by truth.

    Exact nominal observed reconstruction is followed by the declared explicit
    obstacle translation and assertions. This is diagnostic state setup, not a
    strict native training resume with a silently changed configuration.
    """
    import numpy as np
    from scripts.paper_crax_payload.payload_environment import restore_payload_environment
    lines = [json.loads(line) for line in Path(witness_path).read_text().splitlines()]
    header = lines[0]
    if header['kind'] != 'initial_state' or header['condition']['name'] != 'nominal':
        raise ValueError('Route proof requires nominal witness, not a fitted offset rollout')
    from dataclasses import asdict
    if jsonable(asdict(env.config)) != header['task_configuration']:
        raise ValueError('Replay must use exactly the nominal task configuration before explicit state perturbation')
    nominal = np.asarray(header['initial_state']['observation'])
    restored = restore_payload_environment(env, header['initial_state'])
    assert_initial_robot_payload(env,header['initial_state'])
    np.testing.assert_allclose(restored, nominal, rtol=0., atol=2e-6)
    nominal_obstacle = np.asarray(header['initial_state']['obstacle'])
    env.obstacle_position = nominal_obstacle+np.asarray(offset)
    env.sim.set_base_pose('obstacle', env.obstacle_position, np.array([0., 0., 0., 1.]))
    env.client.performCollisionDetection()
    changed = env._obs(); mask = np.ones(len(changed), dtype=bool); mask[20:23] = False
    np.testing.assert_allclose(changed[mask], nominal[mask], rtol=0., atol=2e-6)
    np.testing.assert_allclose(changed[20:23], nominal_obstacle+offset, rtol=0., atol=2e-6)
    actual_body = env.client.getBasePositionAndOrientation(env.obstacle_body)[0]
    np.testing.assert_allclose(actual_body, nominal_obstacle+offset, rtol=0., atol=2e-6)
    minimum, _ = env._distance()
    initial_unsafe = minimum < env.config.safety_distance
    used = 0; success = False; violation = initial_unsafe
    for line in ([] if initial_unsafe else lines[1:]):
        if line['kind'] != 'executed_interval' or line['step'] != used+1:
            raise ValueError('Incomplete or reordered nominal execution stream')
        _, _, done, truncated, info = env.step(line['executed_action'])
        used += 1; minimum = min(minimum, info['minimum_distance'])
        success, violation = bool(info['safe_success']), bool(info['violation'])
        if done or truncated:
            break
    return dict(layout_seed=header['layout_seed'], shifted_minimum_distance=float(minimum),
        shifted_violation=violation, shifted_safe_success=success,
        initial_unsafe_after_declared_offset=initial_unsafe,
        unchanged_initial_robot_gripper_payload_goal_asserted=True,
        obstacle_translation_applied_after_full_restore=True,
        new_control_intervals=used, new_static_initial_distance_queries=1,
        new_interval_physics_substeps=used*env.config.substeps,
        native_interval_distance_labels=True, oracle_used_for_action_selection=False,
        replay_mode='real_open_loop_native_physics',
        native_interval_geometry_covers_all_physical_substeps=True,
        solver_warmstart_bitwise_identity_claim=False)


def run_predeclared_alternative(env, initial_state, offset):
    """Same fixed high detour, separate diagnostic controls; no oracle actions."""
    import numpy as np
    from scripts.paper_crax_payload.payload_environment import restore_payload_environment
    restore_payload_environment(env, initial_state)
    assert_initial_robot_payload(env,initial_state)
    original = env._obs().copy()
    env.obstacle_position = np.asarray(initial_state['obstacle'])+np.asarray(offset)
    env.sim.set_base_pose('obstacle', env.obstacle_position, np.array([0.,0.,0.,1.]))
    env.client.performCollisionDetection()
    mask=np.ones(45,dtype=bool);mask[20:23]=False
    np.testing.assert_allclose(env._obs()[mask], original[mask], rtol=0., atol=2e-6)
    np.testing.assert_allclose(env._obs()[20:23], np.asarray(initial_state['obstacle'])+offset, rtol=0., atol=2e-6)
    minimum, _ = env._distance()
    before = env.physical_control_intervals
    if minimum < env.config.safety_distance:
        return dict(safe_success=False, initial_unsafe=True, new_control_intervals=0,
                    new_static_initial_distance_queries=1)
    waypoints=[env.start+[0.,-.07,0.],env.start+[.03,-.07,.20],env.goal+[.03,0.,.20],env.goal]
    target=0
    for step in range(env.config.horizon):
        position=env.ee_position()
        if np.linalg.norm(position-waypoints[target])<.025 and target<3:target+=1
        action=np.clip((waypoints[target]-position)/env.config.displacement_scale,-1.,1.)
        _,_,done,truncated,info=env.step(action)
        minimum=min(minimum,info['minimum_distance'])
        if done or truncated:break
    return dict(safe_success=bool(info['safe_success']),initial_unsafe=False,minimum_distance=float(minimum),
        new_control_intervals=env.physical_control_intervals-before,
        new_static_initial_distance_queries=1,oracle_used_for_action_selection=False,
        controller='same predeclared high detour as Payload feasibility probe;not learned-policy score')
