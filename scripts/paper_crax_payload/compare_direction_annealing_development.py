"""Paired DEV outcomes and actual failure severity; no new controls or selection.

Missing branches remain missing rather than being scored as failed. Conditions
reuse layouts and must not be treated as independent samples for inference.
This analysis-only entry never imports or runs the simulator or learner.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def outcome(record):
    labels = [key for key in ('safe_success', 'violation', 'safe_timeout') if record[key]]
    if len(labels) != 1:
        raise ValueError('Each executed episode has exactly one terminal outcome')
    return labels[0]


def analyze(config, root, output):
    arms = [arm['name'] for arm in config['arms']]
    count = config['evaluation']['development_episodes_each']
    expected_seeds = set(range(config['evaluation']['development_layout_seed_start'],
                               config['evaluation']['development_layout_seed_start'] + count))
    observations = {}; failures = []; inputs = []; coverage = []
    for arm in arms:
        for step in config['evaluation']['milestones']:
            path = root / arm / f'evaluation_{step}.json'
            coverage.append(dict(arm=arm, steps=step, available=path.exists()))
            if not path.exists():
                continue
            raw = path.read_bytes(); result = json.loads(raw)
            inputs.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest()))
            if (result['milestone'] != step or result['split'] != 'paired_fork_development_only'
                    or result['independent_confirmation']):
                raise ValueError('Only registered paired DEV nodes may be compared')
            expected_conditions = {c['name'] for c in config['evaluation']['conditions']}
            if {c['condition'] for c in result['conditions']} != expected_conditions:
                raise ValueError('Keep all registered conditions, including execution085')
            for condition in result['conditions']:
                records = condition['records']
                if len(records) != count or {r['layout_seed'] for r in records} != expected_seeds:
                    raise ValueError('Keep all paired layout outcomes without duplication')
                for record in records:
                    label = outcome(record)
                    key = (step, condition['condition'], record['layout_seed'])
                    observations.setdefault(key, {})[arm] = record
                    controls = record['control_details']
                    if len(controls) != record['length']:
                        raise ValueError('Every actual control must have failure diagnostics')
                    violating = [c for c in controls if c['violation']]
                    if bool(violating) != bool(record['violation']):
                        raise ValueError('Episode/control violation mismatch')
                    if label == 'violation':
                        for control in violating:
                            failures.append(dict(arm=arm, steps=step, condition=condition['condition'],
                                layout_seed=record['layout_seed'], actual_control=control['step'],
                                initial_clearance_mm=1000*record['layout']['initial_minimum_distance_m'],
                                episode_minimum_clearance_mm=1000*record['minimum_distance_m'],
                                control_minimum_clearance_mm=1000*control['minimum_distance_m'],
                                clearance_deficit_mm=1000*control['safety_clearance_deficit_m'],
                                closest_protected_link=control['closest_protected_link'],
                                raw_norm=control['raw_norm'], executed_norm=control['executed_norm'],
                                actual_tcp_displacement_mm=1000*control['actual_tcp_displacement_norm_m'],
                                task_progress_mm=1000*control['task_progress_m'], collision=control['collision'],
                                source=str(path.relative_to(root))))
    paired = []; comparisons = []
    for (step, condition, seed), records in sorted(observations.items()):
        row = dict(steps=step, condition=condition, layout_seed=seed)
        for arm in arms:
            record = records.get(arm)
            row[arm+'_outcome'] = outcome(record) if record else 'not_collected'
            for name in ('minimum_distance_m', 'length', 'episode_return', 'capped_seconds'):
                row[arm+'_'+name] = record[name] if record else None
        paired.append(row)
    for step in config['evaluation']['milestones']:
        for condition in config['evaluation']['conditions']:
            for left, right in itertools.combinations(arms, 2):
                records = [r for (s, c, _), r in observations.items()
                           if s == step and c == condition['name'] and left in r and right in r]
                if not records:
                    continue
                comparisons.append(dict(steps=step, condition=condition['name'], left=left, right=right,
                    paired_layouts=len(records),
                    both_success=sum(r[left]['safe_success'] and r[right]['safe_success'] for r in records),
                    left_only_success=sum(r[left]['safe_success'] and not r[right]['safe_success'] for r in records),
                    right_only_success=sum(r[right]['safe_success'] and not r[left]['safe_success'] for r in records),
                    neither_success=sum(not r[left]['safe_success'] and not r[right]['safe_success'] for r in records)))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'paired_development_outcomes.csv', paired)
    write_csv(output/'paired_development_discordance.csv', comparisons)
    write_csv(output/'actual_failure_severity.csv', failures)
    receipt = dict(coverage=coverage, executed_episode_records=sum(len(r) for r in observations.values()),
        violating_controls=len(failures), fixed_unique_layouts=count, training_seed_count=1,
        independent_confirmation=False, method_selected=False, inputs=inputs,
        new_simulation_controls=0, new_training_transitions=0,
        scope='paired_development_only_conditions_reuse_layouts_no_independence_claim',
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'paired_development_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key:receipt[key] for key in ('executed_episode_records', 'violating_controls',
                                                'method_selected', 'new_simulation_controls')}))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(json.loads(args.config.read_text()), args.runs, args.output)


if __name__ == '__main__':
    main()
