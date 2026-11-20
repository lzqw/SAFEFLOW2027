"""Combine BOTH complete cross-seed DEV summaries without selecting seeds/modes.

No checkpoints loaded, simulator calls, training, freeze, or independent tests.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256
from scripts.paper_crax_payload.summarize_simulation_v12 import output


ARMS = ('P5-fixed', 'P5-activation2mm', 'Metric-only')
SEEDS = (41028, 41029)
STEPS = (20000, 40000, 60000, 80000, 100000, 150000)
CONDITIONS = ('nominal', 'radius_0575', 'radius_060', 'execution_scale_085')


def modes(step):
    return ('sampled', 'zero_noise') if step in (100000, 150000) else ('sampled',)


def aggregate(nodes):
    expected = {(seed, arm, step, condition, mode)
        for seed in SEEDS for arm in ARMS for step in STEPS
        for condition in CONDITIONS for mode in modes(step)}
    indexed = {}
    for row in nodes:
        key = (int(row['training_seed']), row['arm'], int(row['new_steps']),
               row['condition'], row['deployment_mode'])
        if key in indexed or key not in expected or int(row['episodes']) != 100:
            raise ValueError('Duplicate/unregistered node or unequal episode budget')
        if str(row['independent_test_used']) != 'False':
            raise ValueError('This is development evidence only')
        values = {k: float(row[k]) for k in ('safe_successes', 'violations', 'timeouts',
                  'stall_count', 'mean_return', 'mean_capped_seconds')}
        if (any(not math.isfinite(v) for v in values.values())
                or values['safe_successes']+values['violations']+values['timeouts'] != 100
                or any(not 0 <= values[k] <= 100 for k in
                       ('safe_successes', 'violations', 'timeouts', 'stall_count'))):
            raise ValueError('Keep finite metrics and every terminal class')
        indexed[key] = values
    if set(indexed) != expected:
        raise ValueError('Require both seeds, all arms, six nodes and every deployment mode')
    rows = []
    for arm in ARMS:
        for step in STEPS:
            for condition in CONDITIONS:
                for mode in modes(step):
                    records = [indexed[(seed, arm, step, condition, mode)] for seed in SEEDS]
                    success = [r['safe_successes']/100 for r in records]
                    row = dict(arm=arm, new_steps=step, condition=condition,
                        deployment_mode=mode, training_seed_count=2,
                        episodes_per_seed=100, success_mean=statistics.mean(success),
                        success_worst=min(success), success_best=max(success),
                        success_sd_sample=statistics.stdev(success), independent_test_used=False)
                    for key in ('violations', 'timeouts', 'stall_count', 'mean_return', 'mean_capped_seconds'):
                        row[key+'_mean'] = statistics.mean(r[key] for r in records)
                        for seed, record in zip(SEEDS, records):
                            row[f'{key}_{seed}'] = record[key]
                    row['success_41028'], row['success_41029'] = success
                    rows.append(row)
    return rows


def comparisons(rows):
    index = {(r['arm'], r['new_steps'], r['condition'], r['deployment_mode']): r for r in rows}
    comparisons = []
    for row in rows:
        if row['arm'] == 'Metric-only':
            continue
        baseline = index[('Metric-only', row['new_steps'], row['condition'], row['deployment_mode'])]
        success = row['success_mean'] >= baseline['success_mean']
        both_seeds_success = all(row[f'success_{seed}'] >= baseline[f'success_{seed}'] for seed in SEEDS)
        no_extra_failure = all(row[f'{k}_{seed}'] <= baseline[f'{k}_{seed}']
                              for k in ('violations', 'timeouts', 'stall_count') for seed in SEEDS)
        # Relative return tolerance remains well defined if baseline return is negative.
        return_ok = row['mean_return_mean'] >= baseline['mean_return_mean']-.01*abs(baseline['mean_return_mean'])
        both_seeds_return_ok = all(row[f'mean_return_{seed}'] >= baseline[f'mean_return_{seed}']
                                  -.01*abs(baseline[f'mean_return_{seed}']) for seed in SEEDS)
        target = .93 if row['condition'] == 'radius_0575' else .90 if row['condition'] == 'radius_060' else None
        comparisons.append(dict(arm=row['arm'], new_steps=row['new_steps'], condition=row['condition'],
            deployment_mode=row['deployment_mode'], success_delta=row['success_mean']-baseline['success_mean'],
            violations_delta=row['violations_mean']-baseline['violations_mean'],
            timeouts_delta=row['timeouts_mean']-baseline['timeouts_mean'],
            stalls_delta=row['stall_count_mean']-baseline['stall_count_mean'],
            return_delta=row['mean_return_mean']-baseline['mean_return_mean'],
            success_not_lower_than_metric=success,
            both_seeds_success_not_lower_than_metric=both_seeds_success,
            no_extra_eval_violations_timeouts_stalls=no_extra_failure, return_within_one_percent=return_ok,
            both_seeds_return_within_one_percent=both_seeds_return_ok,
            reach_success_target=target,
            reach_target_satisfied=(row['success_mean'] >= target) if target is not None else None,
            reach_worst_seed_at_least_85percent=(row['success_worst'] >= .85) if target is not None else None,
            success_seed_sd_at_most_5pp=row['success_sd_sample'] <= .05+1e-12,
            joint_development_task_checks=both_seeds_success and no_extra_failure and both_seeds_return_ok,
            independent_confirmation=False))
    return comparisons


def training_comparisons(endpoints):
    index = {(r['training_seed'], r['arm']): r for r in endpoints}
    rows = []
    for seed in SEEDS:
        baseline = index[(seed, 'Metric-only')]
        for arm in ARMS[:2]:
            candidate = index[(seed, arm)]
            rows.append(dict(training_seed=seed, arm=arm, new_steps=150000,
                violations=candidate['violations'], metric_violations=baseline['violations'],
                violations_at_most_80percent_metric=candidate['violations'] <= .8*baseline['violations'],
                successes_not_lower=candidate['safe_successes'] >= baseline['safe_successes'],
                timeouts_not_higher=candidate['safe_timeouts'] <= baseline['safe_timeouts'],
                stalls_not_higher=candidate['training_stall_episodes'] <= baseline['training_stall_episodes'],
                return_within_one_percent=candidate['training_cumulative_return'] >=
                    baseline['training_cumulative_return']-.01*abs(baseline['training_cumulative_return']),
                scope='400K-to550K suffix only; inherited risks not removed',
                iso_comparison_available=False, independent_confirmation=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a100', type=Path, required=True)
    parser.add_argument('--rtx4080', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    admission = json.loads(args.registration.read_text())
    nodes, provenance, endpoints = [], [], []
    for directory, seed, host in ((args.a100, 41028, 'CTAlab'), (args.rtx4080, 41029, '447')):
        receipt = json.loads((directory/'receipt.json').read_text())
        jobs = [j for j in admission['jobs'] if j['seed'] == seed]
        if (receipt['status'] != 'complete_crossseed_one_host_development_summary'
                or receipt['training_seed'] != seed or receipt['host'] != host
                or receipt['completed_runs'] != 3 or receipt['completed_new_training_transitions'] != 450000
                or receipt['development_nodes'] != 96 or receipt['development_episodes'] != 9600
                or receipt['independent_test_used'] is not False or receipt['winner_confirmation'] is not False
                or {p['arm'] for p in receipt['provenance']} != set(ARMS)
                or any(j['host'] != host for j in jobs)):
            raise ValueError('Both complete registered native host summaries required')
        for native in receipt['native_receipts']:
            if (native['new_training_steps'] != 150000 or native['steps'] != 550000
                    or native['status'] != 'completed' or native['source_commit'] != admission['runner_source_commit']):
                raise ValueError('No partial or substituted training endpoints')
            endpoints.append(dict(training_seed=seed, arm=native['arm'],
                **native['segment_counts'], **native['training_metrics']))
        with (directory/'development_nodes.csv').open(newline='') as stream:
            nodes.extend(csv.DictReader(stream))
        provenance.append(dict(seed=seed, host=host, summary_sha256=sha256(directory/'receipt.json'),
            nodes_sha256=sha256(directory/'development_nodes.csv'), native=receipt['provenance']))
    if sorted((r['training_seed'], r['arm']) for r in endpoints) != sorted(
            (seed, arm) for seed in SEEDS for arm in ARMS):
        raise ValueError('All six unique completed method/seed endpoints required')
    result = aggregate(nodes)
    args.output.mkdir(parents=True, exist_ok=False)
    output(args.output/'equal_seed_development.csv', result)
    output(args.output/'comparisons_vs_metric.csv', comparisons(result))
    output(args.output/'training_endpoints.csv', endpoints)
    output(args.output/'training_comparisons_vs_metric.csv', training_comparisons(endpoints))
    (args.output/'receipt.json').write_text(json.dumps(dict(
        status='complete_two_seed_three_arm_development_review',
        completed_new_training_transitions=900000, training_seeds=list(SEEDS),
        completed_runs=6, development_nodes=192, development_episodes=19200,
        provenance=provenance, execution_registration_sha256=sha256(args.registration),
        new_training_steps=0, new_simulation_controls=0, independent_test_used=False,
        winner_confirmation=False, global_objective_complete=False,
        missing_confirmation=['Panda final-configuration regression', 'Payload optimization',
                              'SafeReach from-start superiority', 'third independent training seed',
                              'frozen unseen evaluation', 'same-campaign Iso comparison'],
        physical_robot_interactions=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
