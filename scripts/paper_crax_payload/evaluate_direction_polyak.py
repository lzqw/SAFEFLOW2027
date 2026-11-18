"""One registered fixed250K target-Actor deployment intervention, DEV only.

The existing target parameters are used as the sole deployed Actor; there is
no mixture, second action, new network, training, or best-window selection.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import torch

from scripts.paper_crax_payload.direction_annealing import build as build_v6, sha256
from scripts.paper_crax_payload.direction_optimization import build as build_v7
from scripts.paper_crax_payload.direction_refinement_v8 import build as build_v8
from scripts.paper_crax_payload.evaluate_direction_annealing import evaluate
from scripts.paper_crax_payload.summarize_direction_annealing import csv_output


def actor_digest(actor):
    digest = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def target_deployment(learner):
    original = learner.actor
    if (learner.target_actor.num_steps != original.num_steps
            or learner.target_actor.rho != original.rho
            or learner.target_actor.variance != original.variance):
        raise ValueError('Deployment must retain the native sampler settings')
    learner.actor = learner.target_actor
    try:
        yield
    finally:
        learner.actor = original


def final_counts(value):
    records = [r for c in value['conditions'] for r in c['records']]
    if len(records) != 90:
        raise ValueError('All90 original paired DEV episodes required')
    return dict(safe_successes=sum(r['safe_success'] for r in records),
        violations=sum(r['violation'] for r in records),
        timeouts=sum(r['safe_timeout'] for r in records),
        execution085_successes=sum(r['safe_success'] for c in value['conditions']
            if c['condition']=='execution085' for r in c['records']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (args.output.exists() or not re.fullmatch('[0-9a-f]{40}', args.source_commit)
            or plan['protocol'] != 'fixed250K_existing_target_actor_development_intervention'
            or plan['target_polyak'] != .001 or plan['checkpoint_steps'] != 250000
            or plan['new_training_steps'] != 0 or plan['independent_test_opened']):
        parser.error('Use the exact registered fixed-endpoint intervention in a new directory')
    if shutil.disk_usage(args.output.parent).free < 40*1024**3:
        parser.error('Preserve data-disk reserve')
    torch.set_num_threads(1)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.10, torch.device(args.device))
    from scripts.star.arm_safe_entropy.train import atomic_json
    args.output.mkdir()
    atomic_json(args.output/'registration.json', dict(plan=plan, plan_sha256=sha256(args.plan),
        source_commit=args.source_commit, native_pid=os.getpid(), new_training_steps=0, physical_robot_interactions=0))
    rows, inputs, controls = [], [], 0
    started = time.monotonic()
    builders = {'direction_annealing_v6_full': build_v6,
                'direction_optimization_v7_full': build_v7,
                'direction_refinement_v8_full': build_v8}
    for case in plan['cases']:
        atomic_json(args.output/'status.json', dict(state='running', native_pid=os.getpid(),
            active_case=case['name'], completed_cases=len(rows), registered_cases=len(plan['cases']),
            actual_new_simulation_controls_in_completed_cases=controls, new_training_steps=0))
        checkpoint = Path(case['checkpoint'])
        if sha256(checkpoint) != case['checkpoint_sha256']:
            raise ValueError('Require the exact fixed250K original full checkpoint')
        native_path = checkpoint.parent/'receipt.json'
        if sha256(native_path) != case['receipt_sha256']:
            raise ValueError('Original native training receipt changed')
        native = json.loads(native_path.read_text())
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        identity = saved.get('candidate', saved.get('arm'))
        if (identity['name'] != case['name'] or saved['steps'] != 250000
                or saved['format'] != case['format'] or saved['seed'] != 2028
                or native['status'] != 'completed' or native['steps'] != 250000
                or native['engineering_only'] or native['source_commit'] != saved['source_commit']
                or saved['configuration']['task_version'] != plan['task_version']
                or saved['learner']['ea_stability_v2']['settings']['target_polyak'] != .001):
            raise ValueError('Retain the registered native identity and target averaging')
        config = saved['configuration']
        if (config['evaluation'] != case['evaluation'] or any(
                config['evaluation'].get(key) != value for key, value in plan['evaluation'].items())):
            raise ValueError('Do not change action/layout seeds, conditions, or episodes')
        learner = builders[saved['format']](config, identity, args.device)
        learner.load_state_dict(saved['learner'])
        del saved
        original_actor, original_target = actor_digest(learner.actor), actor_digest(learner.target_actor)
        prefix = native['parent'].get('common100K', native['parent'].get('prefix'))
        training = {key: native[counter]-prefix[counter] for key, counter in
            [('training_successes', 'safe_successes'), ('training_violations', 'training_violations'), ('training_timeouts', 'safe_timeouts')]}
        with target_deployment(learner):
            if actor_digest(learner.actor) != original_target:
                raise ValueError('Only the original target Actor may be deployed')
            value = evaluate(learner, config, 250000)
        if actor_digest(learner.actor) != original_actor or actor_digest(learner.target_actor) != original_target:
            raise RuntimeError('Frozen intervention mutated original online/target parameters')
        value.update(deployment='sole_existing_target_actor_polyak001', split='fixed_endpoint_development_intervention',
            independent_confirmation=False, no_weight_window_search=True)
        directory = args.output/case['name']
        directory.mkdir()
        atomic_json(directory/'evaluation_250000.json', value)
        controls += value['native_controls']
        online_path = checkpoint.parent/'evaluation_250000.json'
        online = final_counts(json.loads(online_path.read_text()))
        target = final_counts(value)
        goal = plan['goals']
        eligible = (target['safe_successes'] >= goal['safe_successes']
            and target['execution085_successes'] >= goal['execution085_successes']
            and training['training_violations'] <= goal['comparative_training_violations']
            and training['training_successes'] >= goal['comparative_training_successes']
            and training['training_timeouts'] <= goal['comparative_training_timeouts_reference'])
        row = dict(name=case['name'], **training, **{'online_'+k:v for k,v in online.items()},
            **{'target_'+k:v for k,v in target.items()}, joint_development_target_entered=eligible,
            online_actor_sha256=original_actor, target_actor_sha256=original_target)
        rows.append(row)
        inputs.extend(dict(path=str(p), sha256=sha256(p)) for p in (checkpoint, native_path, online_path))
        atomic_json(directory/'receipt.json', dict(status='completed', **row, new_training_steps=0,
            actual_new_simulation_controls=value['native_controls'], original_training_source_commit=native['source_commit']))
        print(json.dumps(row), flush=True)
        del learner
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
    csv_output(args.output/'fixed_endpoint_comparison.csv', rows)
    atomic_json(args.output/'receipt.json', dict(status='completed', registered_cases=len(plan['cases']),
        completed_cases=len(rows), plan_sha256=sha256(args.plan), source_commit=args.source_commit,
        new_training_steps=0, actual_new_simulation_controls=controls, elapsed_seconds=time.monotonic()-started,
        eligible_development_candidates=[r['name'] for r in rows if r['joint_development_target_entered']],
        new_seed_confirmation_executed=False, independent_test_opened=False, all_cases_retained=True,
        no_best_checkpoint_or_window_selection=True, github_uploads=0, inputs=inputs))
    atomic_json(args.output/'status.json', dict(state='completed', native_pid=os.getpid(),
        completed_cases=len(rows), registered_cases=len(plan['cases']), new_training_steps=0,
        actual_new_simulation_controls=controls))


if __name__ == '__main__':
    main()
