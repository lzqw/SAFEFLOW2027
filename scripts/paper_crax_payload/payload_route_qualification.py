"""Diagnostic-only offset route proof helpers. Never a behavior action selector."""
from collections import Counter
import math
from statistics import mean


def route_sector(payload_path, obstacle_position):
    """Predefined8-sector x/z crossing proxy; not global exploration entropy."""
    point = min(payload_path, key=lambda p: abs(p[1]-obstacle_position[1]))
    angle = math.atan2(point[2]-obstacle_position[2], point[0]-obstacle_position[0])
    return int((angle+math.pi)/(2*math.pi)*8)%8


def replay_offset_distance(counterfactual_env, source_env, offset):
    """Static true arm/payload geometry for ONE actual source physical substep.

    Caller must invoke on every recorded nominal substep in a separate client.
    This advances NO physics, resets NO live source state, reads no policy logits
    and chooses NO action. Count each call as a static diagnostic query. A TCP
    or payload-centroid-only reconstruction is not sufficient for arm safety.
    """
    import numpy as np
    if counterfactual_env is source_env:
        raise ValueError('Never modify the live source environment for truth replay')
    if (tuple(counterfactual_env.config.payload_half_extents) != tuple(source_env.config.payload_half_extents) or
            counterfactual_env.config.obstacle_radius != source_env.config.obstacle_radius):
        raise ValueError('Replay changes only obstacle position, not body geometry')
    for joint in range(7):
        state = source_env.client.getJointState(source_env.body, joint)
        counterfactual_env.client.resetJointState(counterfactual_env.body, joint, state[0], targetVelocity=state[1])
    for joint in (9, 10):
        state = source_env.client.getJointState(source_env.body, joint)
        counterfactual_env.client.resetJointState(counterfactual_env.body, joint, state[0], targetVelocity=state[1])
    position, orientation = source_env.client.getBasePositionAndOrientation(source_env.payload_body)
    linear, angular = source_env.client.getBaseVelocity(source_env.payload_body)
    counterfactual_env.client.resetBasePositionAndOrientation(counterfactual_env.payload_body, position, orientation)
    counterfactual_env.client.resetBaseVelocity(counterfactual_env.payload_body, linear, angular)
    shifted = source_env.obstacle_position+np.asarray(offset)
    counterfactual_env.obstacle_position = shifted
    counterfactual_env.sim.set_base_pose('obstacle', shifted, np.array([0., 0., 0., 1.]))
    counterfactual_env.client.performCollisionDetection()
    arm, payload, _ = counterfactual_env._component_distance()
    return dict(minimum_distance=min(arm, payload), arm_distance=arm, payload_distance=payload,
                new_control_intervals=0, new_static_scene_queries=1)


def qualify_offset_route(protocol, evidence, selected_policy_hashes, freeze_digest):
    methods = list(selected_policy_hashes)
    if (evidence.get('source_split') != 'route_calibration_validation_only' or
            evidence.get('oracle_used_for_action_selection') is not False or
            evidence.get('native_interval_geometry_covers_all_physical_substeps') is not True or
            evidence.get('replay_mode') != 'real_open_loop_native_physics'):
        raise ValueError('Need validation-only real openloop native interval truth; no XYZ-only/oracle-selected behavior')
    if (not freeze_digest or evidence.get('selection_freeze_digest') != freeze_digest or
            evidence.get('selected_policy_sha256') != selected_policy_hashes or
            evidence.get('obstacle_offset') != protocol['fixed_obstacle_offset']):
        raise ValueError('Route proof needs exact selected policy/fixed offset provenance, not test-fitted geometry')
    wanted = list(range(protocol['calibration_layout_seed_start'],
                        protocol['calibration_layout_seed_start']+protocol['calibration_layout_count']))
    records = evidence['nominal_trajectories']
    if not all(r.get('native_interval_distance_labels') is True for r in records):
        raise ValueError('Every reconstructed route needs actual native interval geometry, not line intersections')
    if not all(r.get('initial_unsafe_after_declared_offset') is False for r in records):
        return dict(status='offset_initial_admissibility_not_verified', alternative_route_claim_verified=False)
    histograms = {}
    for method in methods:
        rows = [r for r in records if r['method']==method]
        if sorted(r['layout_seed'] for r in rows) != wanted:
            raise ValueError('Every frozen selected method gets the same registered calibration layouts')
        successes = [r for r in rows if r['source_safe_success']]
        if not successes:
            return dict(status='inconclusive_no_successful_nominal_route', alternative_route_claim_verified=False,
                method_without_route=method)
        histogram = Counter(r['sector'] for r in successes)
        histograms[method] = {sector: count/len(successes) for sector, count in histogram.items()}
    # Equal method weights: a method with more successes does not define all of
    # the benchmark's route calibration. Sector tie-break is deterministic.
    sector = min(range(8), key=lambda k: (-mean(h.get(k, 0.) for h in histograms.values()), k))
    blocked = []
    for method in methods:
        usual = [r for r in records if r['method']==method and r['source_safe_success'] and r['sector']==sector]
        if not usual:
            return dict(status='inconclusive_pooled_sector_missing_for_method', alternative_route_claim_verified=False)
        blocked.append(mean(r['shifted_minimum_distance'] < evidence['safety_distance_m'] for r in usual))
    alternative = evidence['predeclared_alternative_rollouts']
    if sorted(r['layout_seed'] for r in alternative) != wanted:
        raise ValueError('Alternative feasibility must use identical calibration layouts, no seed rejection')
    if evidence.get('alternative_controller_changed_after_results') is not False:
        raise ValueError('Alternative path cannot be oracle-reranked or adapted to test responses')
    if not all(r.get('initial_unsafe') is False for r in alternative):
        return dict(status='alternative_initial_admissibility_not_verified', alternative_route_claim_verified=False)
    alternative_success = mean(bool(r['safe_success']) for r in alternative)
    verified = mean(blocked) >= protocol['minimum_blocked_usual_route_fraction'] and alternative_success >= protocol['minimum_alternative_safe_success_fraction']
    return dict(status='verified_alternative_route_condition' if verified else protocol['failure_label'],
        alternative_route_claim_verified=verified, pooled_usual_sector=sector,
        equal_method_mean_usual_route_blocked_fraction=mean(blocked),
        per_method_usual_route_blocked_fraction=dict(zip(methods, blocked)),
        predeclared_alternative_safe_success=alternative_success,
        oracle_used_for_action_selection=False, independent_test_used=False)
