"""Summarize all registered v5 development arms, including pending/failed runs.

Only compact run records are read. No model, simulator, tests or new training.
Output is explicitly developmental; selection/confirmation are separate work.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def summarize(config, root):
    inventory = []; training = []; evaluation = []; initializations = set(); sources = set()
    for seed in config['development_training_seeds']:
        for arm in config['arms']:
            directory = root/f"{arm['name']}-seed{seed}"
            manifest = load(directory/'run_config.json')
            status = load(directory/'status.json') or dict(state='not_started', steps=0)
            receipt = load(directory/'receipt.json')
            if manifest:
                if (manifest['configuration'] != config or manifest['arm'] != arm
                        or manifest['training_seed'] != seed or manifest['engineering_only']):
                    raise ValueError(f'Wrong registered scientific run: {directory}')
                initializations.add((seed, manifest['initial_network_sha256']))
                sources.add(manifest['source_commit'])
            if receipt and (receipt['status'] != 'completed' or receipt['steps'] != config['steps_per_run']
                            or receipt['independent_endpoint_test_executed'] or receipt['engineering_only']):
                raise ValueError('Do not accept incomplete engineering or independent results as development completion')
            progress_path = directory/'progress.jsonl'
            progress = []
            if progress_path.exists():
                for line in progress_path.read_text().splitlines():
                    try: progress.append(json.loads(line))
                    except json.JSONDecodeError: raise ValueError('Copy only flushed compact progress files')
            if any(a['steps'] >= b['steps'] for a, b in zip(progress, progress[1:])):
                raise ValueError('Duplicate or backwards native progress; inspect resume lineage')
            for row in progress:
                training.append(dict(arm=arm['name'], training_seed=seed, **row))
            last = progress[-1] if progress else {}
            inventory.append(dict(arm=arm['name'], training_seed=seed, state=status['state'],
                actual_steps=receipt['steps'] if receipt else status['steps'], target_steps=config['steps_per_run'],
                training_violations=receipt['training_violations'] if receipt else last.get('training_violations'),
                safe_successes=receipt['safe_successes'] if receipt else last.get('safe_successes'),
                safe_timeouts=receipt['safe_timeouts'] if receipt else last.get('safe_timeouts'),
                completed=bool(receipt), source_commit=manifest['source_commit'] if manifest else None,
                initial_network_sha256=manifest['initial_network_sha256'] if manifest else None))
            for milestone in config['evaluation']['milestones']:
                result = load(directory/f'evaluation_{milestone}.json')
                if result is None: continue
                if (result['split'] != 'development_only' or result['independent_confirmation']
                        or result['milestone'] != milestone):
                    raise ValueError('Never merge independent outcomes into the new development selection table')
                for condition in result['conditions']:
                    records = condition['records']
                    if len(records) != config['evaluation']['development_episodes_each']:
                        raise ValueError('Incomplete development episode denominator')
                    evaluation.append(dict(arm=arm['name'], training_seed=seed, milestone=milestone,
                        condition=condition['condition'], episodes=len(records),
                        safe_successes=sum(r['safe_success'] for r in records),
                        violations=sum(r['violation'] for r in records),
                        safe_timeouts=sum(r['safe_timeout'] for r in records),
                        mean_return=sum(r['episode_return'] for r in records)/len(records),
                        mean_capped_seconds=sum(r['capped_seconds'] for r in records)/len(records),
                        split='development_only'))
    for seed in config['development_training_seeds']:
        if len({digest for s, digest in initializations if s == seed}) > 1:
            raise ValueError('Paired fresh initialization mismatch')
    if len(sources) > 1: raise ValueError('Source revisions differ; explicit lineage review required')
    return inventory, training, evaluation


def write_csv(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); config = load(args.config)
    inventory, training, evaluation = summarize(config, args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'training_inventory.csv', inventory)
    write_csv(args.output/'training_curve.csv', training)
    write_csv(args.output/'development_evaluations.csv', evaluation)
    inputs = [dict(path=str(path.relative_to(args.runs)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(args.runs.rglob('*')) if path.is_file()]
    value = dict(protocol=config['protocol'], completed_runs=sum(r['completed'] for r in inventory),
        expected_runs=len(inventory), actual_recorded_training_steps=sum(r['actual_steps'] for r in inventory),
        finite_development_budget=config['new_development_training_budget'],
        development_condition_rows=len(evaluation), training_seed_count=len(config['development_training_seeds']),
        independent_confirmation=False, method_selected=False, new_training_transitions=0,
        source_commits=sorted({r['source_commit'] for r in inventory if r['source_commit']}),
        inputs=inputs)
    (args.output/'receipt.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'inputs'}))


if __name__ == '__main__': main()
