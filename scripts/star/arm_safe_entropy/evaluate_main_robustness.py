"""Frozen formal episode tests, reusing native nominal records verbatim.

Prepare the shared physical-layout evidence once, then evaluate one completed
main run at ALL four declared checkpoints. This never launches training.
"""
import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from scripts.star.arm_safe_entropy.evaluate_obstacle_robustness import (
    atomic_json, load_policy, physics_worker, rollout_wave, summarize)
from scripts.star.arm_safe_entropy.run_protocol import validate_main_freeze


@contextmanager
def physics_pool(count):
    context = mp.get_context('spawn')
    pipes, workers = [], []
    try:
        for _ in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=physics_worker, args=(child,))
            process.start()
            child.close()
            pipes.append(parent)
            workers.append(process)
        yield pipes
    finally:
        for pipe in pipes:
            try:
                pipe.send(('close', None))
            except (OSError, EOFError):
                pass
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()  # Only this invocation's own simulation child.
                process.join()
        for pipe in pipes:
            pipe.close()


def check_plan(frozen, plan):
    validate_main_freeze(frozen, plan['training_methods'][0], plan['training_seeds'][0],
                         .04, task_version=plan['task_version'])
    if plan.get('preparation_only') or frozen.get('robustness_plan') != plan:
        raise ValueError('The formal robustness protocol is not frozen into the native training configuration')
    required = dict(training_methods=frozen['methods'], training_seeds=frozen['training_seeds'],
                    evaluation_steps=frozen['evaluation_steps'],
                    episodes_per_condition_per_training_seed=frozen['evaluation_episodes'],
                    layout_seed_start=frozen['evaluation_layout_seed_start'], task_version=frozen['task_version'])
    if any(plan.get(key) != value for key, value in required.items()):
        raise ValueError('Robustness and native main protocols disagree')


def check_layouts(records, condition, fixed):
    for record in records:
        reference = fixed[(condition['name'], record['layout_seed'])]
        for key in ['start', 'goal', 'obstacle', 'radius_m']:
            if not np.allclose(record['layout'][key], reference[key], rtol=0., atol=1e-9):
                raise ValueError('Do not compare different actual layouts or silently remove a failed episode')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-config', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-layouts', action='store_true')
    modes.add_argument('--run', type=Path, help='One completed1M main run; all four checkpoint policies are required')
    parser.add_argument('--layout-validity', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    frozen = json.loads(args.frozen_config.read_text())
    plan = json.loads(args.plan.read_text())
    try:
        check_plan(frozen, plan)
    except ValueError as error:
        parser.error(str(error))
    episodes = plan['episodes_per_condition_per_training_seed']
    batch_size = plan['action_batch_size']
    batches = [list(range(i, min(i+batch_size, episodes))) for i in range(0, episodes, batch_size)]
    expected = {(c['name'], seed) for c in plan['conditions']
                for seed in range(plan['layout_seed_start'], plan['layout_seed_start']+episodes)}
    fixed = None
    if args.run:
        if args.layout_validity is None:
            parser.error('Reuse the shared complete layout evidence; do not repeat it per policy')
        proof = json.loads(args.layout_validity.read_text())
        if (proof['plan'] != plan or proof['safe_detours'] != len(expected)
                or len(proof['records']) != len(expected)):
            parser.error('Use the complete shared physical-layout evidence for this exact frozen protocol')
        fixed = {(r['condition'], r['layout_seed']): r['layout'] for r in proof['records']}
        if set(fixed) != expected:
            parser.error('The physical-layout evidence has a missing or duplicated condition/layout')
        config = json.loads((args.run / 'run_config.json').read_text())
        receipt = json.loads((args.run / 'receipt.json').read_text())
        if (config.get('stage') != 'main' or config.get('frozen_config') != frozen
                or receipt.get('status') != 'completed' or receipt.get('steps') != 1000000):
            parser.error('Use the identical completed frozen1M main run, not a development or selected run')
    if args.output.exists():
        parser.error('Never overwrite earlier evaluation evidence')
    args.output.mkdir(parents=True)
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    try:
        with physics_pool(batch_size) as pipes:
            if args.prepare_layouts:
                records = []
                for condition in plan['conditions']:
                    for indices in batches:
                        records.extend(dict(condition=condition['name'], **row)
                                       for row in rollout_wave(pipes, condition, indices, plan))
                proof = dict(plan=plan, records=records, safe_detours=sum(r['safe_success'] for r in records),
                    validation_transitions=sum(r['length'] for r in records), source_commit=args.source_commit,
                    training_transitions=0, adaptation_updates=0)
                atomic_json(args.output / 'layout_validity.json', proof)
                if proof['safe_detours'] != len(expected):
                    raise RuntimeError('The fixed physical route did not establish every layout; preserve failures,do not run policy comparisons')
                atomic_json(args.output / 'receipt.json', dict(status='completed', layouts=len(expected),
                            training_transitions=0, validation_transitions=proof['validation_transitions']))
                return
            results = []
            for step in plan['evaluation_steps']:
                actor, margin, provenance = load_policy(args.run, args.device, step, frozen)
                for condition in plan['conditions']:
                    native_file = None
                    if condition['name'] == 'nominal':
                        native_file = args.run / f'evaluation_{step}.json'
                        native = json.loads(native_file.read_text())
                        if (native['milestone_steps'] != step or native['episodes'] != episodes or len(native['records']) != episodes or
                                native['fixed_layout_seeds'] != list(range(plan['layout_seed_start'], plan['layout_seed_start']+episodes))):
                            raise ValueError('Do not replace the fixed native nominal episodes with a new stochastic result')
                        records = [dict(**row, layout_seed=row['seed']) for row in native['records']]
                    else:
                        records = []
                        for indices in batches:
                            records.extend(rollout_wave(pipes, condition, indices, plan, actor, margin,
                                                        provenance['training_seed']))
                    check_layouts(records, condition, fixed)
                    result = dict(**provenance, task_version=plan['task_version'], condition=condition,
                        **summarize(records), records=records, evaluation_source_commit=args.source_commit,
                        nominal_source=str(native_file) if native_file else None,
                        nominal_reused_without_new_rollout=bool(native_file),
                        training_transitions=0, adaptation_updates=0,
                        status='predeclared_frozen_main_checkpoint_zero_shot')
                    atomic_json(args.output / f'evaluation_{step}_{condition["name"]}.json', result)
                    results.append(result)
                    atomic_json(args.output / 'status.json', dict(state='running', completed_conditions=len(results),
                                expected_conditions=20, elapsed_seconds=time.monotonic()-started))
                    print(json.dumps({key: result[key] for key in
                                      ['method', 'training_seed', 'checkpoint_steps', 'condition', 'safe_success', 'violation']}), flush=True)
                del actor, margin
            atomic_json(args.output / 'receipt.json', dict(status='completed', plan=plan,
                frozen_config=frozen, run=str(args.run), source_commit=args.source_commit,
                method=config['method'], training_seed=config['training_seed'],
                condition_rows=len(results), policy_episodes=sum(r['episodes'] for r in results),
                native_nominal_episodes_reused=sum(r['episodes'] for r in results if r['nominal_reused_without_new_rollout']),
                new_stress_episodes=sum(r['episodes'] for r in results if not r['nominal_reused_without_new_rollout']),
                new_evaluation_transitions=sum(r['evaluation_transitions'] for r in results if not r['nominal_reused_without_new_rollout']),
                training_transitions=0, adaptation_updates=0, layout_validity=str(args.layout_validity),
                elapsed_seconds=time.monotonic()-started))
            atomic_json(args.output / 'status.json', dict(state='completed', completed_conditions=len(results)))
    except BaseException as error:
        atomic_json(args.output / 'status.json', dict(state='engineering_interruption',
                    error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
