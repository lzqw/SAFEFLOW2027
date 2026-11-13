"""Common old-Reach validation and offline candidate release; no training.

All five original checkpoints of all three controls receive the same 50
validation episodes/condition. No independent-test/development score selects
the winner. Risk qualification requires <=0.1 violations in every condition.
The release is an offline preparation artifact, never a hardware enable.
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from deployment.piper.bundle import FrozenPiperPolicy
from deployment.piper.replay_policy import benchmark, read_observations, replay_bundle
from scripts.paper_crax_payload.piper_train import evaluate_piper, export_command, load_frozen, make_piper_env
from scripts.star.arm_safe_entropy.train import atomic_json


METHODS = ('Iso', 'Iso-Low', 'EA')
STEPS = (60000, 120000, 180000, 240000, 300000)


def summarize(evaluation, risk_limit=.1):
    conditions = evaluation['conditions']
    return dict(risk_qualified=all(c['violation'] <= risk_limit for c in conditions),
        task_progress_observed=any(c['safe_success'] > 0 for c in conditions),
        macro_safe_success=sum(c['safe_success'] for c in conditions)/len(conditions),
        macro_capped_seconds=sum(c['time_to_goal_capped_seconds'] for c in conditions)/len(conditions),
        per_condition_violation=[c['violation'] for c in conditions],
        per_condition_safe_success=[c['safe_success'] for c in conditions])


def choose(candidates):
    eligible = [c for c in candidates if c['score']['risk_qualified']]
    front = [c for c in candidates if not any(
        other['score']['macro_safe_success'] >= c['score']['macro_safe_success'] and
        all(a <= b for a, b in zip(other['score']['per_condition_violation'], c['score']['per_condition_violation'])) and
        (other['score']['macro_safe_success'] > c['score']['macro_safe_success'] or
         other['score']['per_condition_violation'] != c['score']['per_condition_violation'])
        for other in candidates)]
    winner = min(eligible or front, key=lambda c: (-c['score']['macro_safe_success'],
        c['score']['macro_capped_seconds'], c['steps']))
    status = ('risk_qualified' if winner['score']['task_progress_observed'] else 'risk_qualified_but_stagnant') if eligible else 'pareto_tradeoff_not_risk_qualified'
    return winner, status


def write_observation_fixture(args, candidate):
    config, assets, _, learner = load_frozen(SimpleNamespace(config=args.config,
        assets_receipt=args.assets_receipt, checkpoint=Path(candidate['checkpoint']), device='cpu'))
    env = make_piper_env(config, assets, 'Reach')
    output = args.output/'observations.jsonl'
    try:
        torch.manual_seed(694000)
        obs, _ = env.reset(seed=650000)
        with output.open('x') as stream:
            stream.write(json.dumps(dict(record_type='metadata', robot_family='Piper',
                frame='robot_base', observation_dimension=34, source='native simulation development episode',
                layout_seed=650000, physical_robot_interactions=0))+'\n')
            for step in range(env.config.horizon):
                stream.write(json.dumps(dict(timestamp_s=step*env.control_dt, observation=obs.tolist()))+'\n')
                obs, _, done, truncated, _ = env.step(learner.action(obs[None])[0])
                if done or truncated:
                    break
        return dict(controls=env.total_control_intervals, substeps=env.total_physics_substeps,
            distance_queries=env.total_distance_queries, source_is_not_physical_recording=True)
    finally:
        env.close()


def reach_schedule():
    # Every condition/repeat receives each method; rotate method order.
    rows = []
    for repeat in range(10):
        for ci, condition in enumerate(('Nominal', 'UnseenGeometry', 'ExecutionVariation')):
            offset = (repeat+ci) % len(METHODS)
            for method in METHODS[offset:]+METHODS[:offset]:
                rows.append(dict(order=len(rows)+1, repeat=repeat+1, task='Reach', condition=condition,
                    method=method, status='planned_not_executed'))
    return dict(rows=rows, episodes=90, physical_motion_authorized=False,
        activation_requires='on-site TCP/frame/period/workspace/speed calibration and explicit operator confirmation',
        required_metrics=['safe_success', 'unassisted_safe_success', 'protection_intervention_rate', 'completion_seconds'],
        physical_robot_interactions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not repeat an existing validation/release')
    config = json.loads(args.config.read_text())
    plan = config['evaluation']
    count, first = plan['validation_episodes_per_condition'], plan['validation_layout_seed_start']
    if count != 50 or first != 61027 or len(plan['conditions']) != 3:
        raise ValueError('Use the existing common validation stream and three conditions')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    candidates = []
    for method in METHODS:
        run = args.original_root/'runs'/f'piper-reach-{method.lower()}-41027-v1'
        for step in STEPS:
            checkpoint, receipt = run/f'full_{step}.pt', run/f'training_receipt_{step}.json'
            if not checkpoint.is_file() or not receipt.is_file():
                raise FileNotFoundError(f'Complete equal candidate opportunity required: {checkpoint}')
            candidates.append(dict(method=method, steps=step, checkpoint=str(checkpoint), receipt=str(receipt)))
    atomic_json(args.output/'plan.json', dict(cohort='original_native_Reach_seed41027',
        methods=METHODS, steps=STEPS, candidates=candidates, validation_layouts=list(range(first, first+count)),
        conditions=plan['conditions'], risk_limit_each_condition=.1,
        selection='risk-qualified then macro safe success then capped time then earliest checkpoint; otherwise Pareto',
        independent_test_used_for_selection=False, new_training_transitions=0))
    result_rows = []
    for index, candidate in enumerate(candidates):
        atomic_json(args.output/'status.json', dict(state='validation', completed_candidates=index,
            total_candidates=len(candidates), active_candidate=candidate))
        _, assets, saved, learner = load_frozen(SimpleNamespace(config=args.config,
            assets_receipt=args.assets_receipt, checkpoint=Path(candidate['checkpoint']), device='cpu'))
        if (saved['task'] != 'Reach' or saved['method'] != candidate['method'] or
                saved['steps'] != candidate['steps'] or saved.get('engineering')):
            raise ValueError('Original native candidate identity mismatch')
        evaluation = evaluate_piper(learner, config, assets, 'Reach', count, first)
        evaluation.update(split='validation', method=candidate['method'], checkpoint_steps=candidate['steps'],
            training_seed=41027, independent_test_used_for_selection=False)
        name = f'validation-{candidate["method"].lower()}-{candidate["steps"]}.json'
        atomic_json(args.output/name, evaluation)
        candidate['score'] = summarize(evaluation)
        candidate['validation_file'] = name
        for condition in evaluation['conditions']:
            result_rows.append(dict(method=candidate['method'], checkpoint_steps=candidate['steps'],
                split='validation', condition=condition['condition'], episodes=count,
                safe_success=condition['safe_success'], violation=condition['violation'],
                capped_seconds=condition['time_to_goal_capped_seconds']))
        del saved, learner
    selected = {}
    for method in METHODS:
        winner, status = choose([c for c in candidates if c['method'] == method])
        selection = dict(robot_family='Piper', task='Reach', method=method,
            selected_checkpoint_steps=winner['steps'], selection_split='validation',
            independent_test_used_for_selection=False, risk_qualified=winner['score']['risk_qualified'],
            status=status, candidate_count=5, score=winner['score'],
            physical_deployment_eligible=False, hardware_calibration_pending=True)
        selection_record = args.output/f'selection-{method.lower()}.json'
        atomic_json(selection_record, selection)
        export_command(SimpleNamespace(config=args.config, assets_receipt=args.assets_receipt,
            checkpoint=Path(winner['checkpoint']), device='cpu', selection_record=selection_record,
            training_receipt=Path(winner['receipt']), output=args.output/'bundles'/method.lower()))
        selected[method] = dict(candidate=winner, selection=selection)
    fixture_work = write_observation_fixture(args, selected['EA']['candidate'])
    _, observations = read_observations(args.output/'observations.jsonl')
    timings, replays = {}, {}
    for method in METHODS:
        bundle = args.output/'bundles'/method.lower()
        policy = FrozenPiperPolicy(bundle)
        timings[method] = benchmark(policy, observations)
        replays[method] = replay_bundle(bundle, args.output/'observations.jsonl',
            args.output/f'offline-{method.lower()}.jsonl')
    with (args.output/'validation_results.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0])); writer.writeheader(); writer.writerows(result_rows)
    atomic_json(args.output/'selection.json', dict(methods=selected, candidates=candidates,
        independent_test_used_for_selection=False, fixed_300K_endpoints_not_replaced=True))
    atomic_json(args.output/'latency.json', timings)
    atomic_json(args.output/'planned_onsite_reach90.json', reach_schedule())
    atomic_json(args.output/'receipt.json', dict(status='completed_offline_preparation',
        validation_candidates=len(candidates), validation_episodes=15*3*count,
        bundles=list(METHODS), fixture_work=fixture_work, offline_replays=replays,
        training_transitions=0, independent_test_executed=False, physical_robot_interactions=0,
        real_deployment_machine_latency_pending=True, physical_motion_enabled=False))
    atomic_json(args.output/'status.json', dict(state='completed_offline_preparation',
        completed_candidates=len(candidates), total_candidates=len(candidates)))


if __name__ == '__main__':
    main()
