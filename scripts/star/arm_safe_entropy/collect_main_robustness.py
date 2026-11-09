"""Mirror/export existing frozen main tests; never train or rerun an episode.

Keep MAIN_ROBUSTNESS_RESULTS.csv separate from the original negative v2
ROBUSTNESS_RESULTS.csv. Incomplete evaluation invocations remain visible.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess


COLUMNS = ['stage', 'task_version', 'method', 'training_seed', 'checkpoint_steps',
           'primary_final_checkpoint', 'condition', 'perturbation', 'severity',
           'radius_m', 'lateral_offset_m', 'evaluation_episodes', 'safe_success',
           'task_success', 'violation_rate', 'collision_rate', 'return',
           'time_to_goal_seconds', 'time_to_goal_successes', 'time_to_goal_capped_seconds',
           'minimum_distance_m', 'safe_success_change_from_nominal',
           'nominal_reused_without_new_rollout', 'policy_source_commit',
           'evaluation_source_commit', 'native_run', 'evidence', 'run_evaluation_status']


def read_json(path):
    return json.loads(path.read_text())


def export_results(report, evidence, frozen, require_complete=False):
    plan = frozen['robustness_plan']
    conditions = {condition['name']: condition for condition in plan['conditions']}
    pairs = {(method, seed) for method in frozen['methods'] for seed in frozen['training_seeds']}
    milestones = set(frozen['evaluation_steps'])
    expected = {(method, seed, step, condition) for method, seed in pairs
                for step in milestones for condition in conditions}
    layout_seeds = set(range(plan['layout_seed_start'],
                             plan['layout_seed_start'] + frozen['evaluation_episodes']))
    native, completed_pairs, indexed = {}, set(), {}
    for config_path in sorted((report / 'main').glob('*/run_config.json')):
        run = config_path.parent
        if not (run / 'receipt.json').exists():
            continue
        config, receipt = read_json(config_path), read_json(run / 'receipt.json')
        if receipt.get('status') != 'completed' or receipt.get('steps') != 1000000:
            continue
        pair = (config['method'], config['training_seed'])
        if pair not in pairs or config.get('frozen_config') != frozen or pair in native:
            raise ValueError('Missing frozen identity or duplicate completed native main method/seed')
        native[pair] = (run, config)
    for folder in sorted(evidence.iterdir()) if evidence.exists() else []:
        if not folder.is_dir():
            continue
        paths = sorted(folder.glob('evaluation_*_*.json'))
        receipt = read_json(folder / 'receipt.json') if (folder / 'receipt.json').exists() else {}
        invocation = set()
        for path in paths:
            result = read_json(path)
            pair = (result['method'], result['training_seed'])
            condition = result['condition']
            key = (*pair, result['checkpoint_steps'], condition['name'])
            if (key not in expected or key in indexed or pair not in native
                    or result.get('task_version') != frozen['task_version']
                    or condition != conditions[condition['name']]
                    or result.get('training_transitions') != 0 or result.get('adaptation_updates') != 0
                    or result['episodes'] != frozen['evaluation_episodes']
                    or len(result['records']) != frozen['evaluation_episodes']
                    or {r['layout_seed'] for r in result['records']} != layout_seeds):
                raise ValueError(f'Incomplete, duplicate or non-frozen formal condition: {path}')
            run, config = native[pair]
            if result['policy_source_commit'] != config['source_commit']:
                raise ValueError('Evaluation policy source differs from the completed native run')
            nominal = condition['name'] == 'nominal'
            if result.get('nominal_reused_without_new_rollout') != nominal:
                raise ValueError('Reuse nominal exactly; only stress conditions execute new episodes')
            if nominal:
                original = read_json(run / f'evaluation_{result["checkpoint_steps"]}.json')
                if result['records'] != [dict(**r, layout_seed=r['seed']) for r in original['records']]:
                    raise ValueError('Nominal records differ from their original fixed native evaluation')
            indexed[key] = (result, path, receipt.get('status') == 'completed', run)
            invocation.add(key)
        if receipt.get('status') == 'completed':
            pair = (receipt['method'], receipt['training_seed'])
            if (pair not in pairs or pair not in native
                    or receipt.get('plan') != plan or receipt.get('frozen_config') != frozen
                    or receipt.get('condition_rows') != 20 or receipt.get('policy_episodes') != 2000
                    or receipt.get('native_nominal_episodes_reused') != 400
                    or receipt.get('new_stress_episodes') != 1600
                    or receipt.get('training_transitions') != 0 or receipt.get('adaptation_updates') != 0
                    or invocation != {key for key in expected if key[:2] == pair}
                    or pair in completed_pairs):
                raise ValueError('A completed evaluation must retain every declared checkpoint and condition')
            completed_pairs.add(pair)
    if require_complete and (set(indexed) != expected or completed_pairs != pairs):
        raise ValueError(f'Formal tests are not complete: {len(indexed)}/180 rows, {len(completed_pairs)}/9 receipts')
    rows = []
    for key, (result, path, complete, run) in sorted(indexed.items()):
        condition = result['condition']
        nominal = indexed.get((*key[:3], 'nominal'))
        rows.append(dict(stage=plan['stage'], task_version=frozen['task_version'],
            method=key[0], training_seed=key[1], checkpoint_steps=key[2],
            primary_final_checkpoint=key[2] == 1000000,
            condition=key[3], perturbation=condition['family'], severity=condition['severity'],
            radius_m=condition['radius_m'], lateral_offset_m=condition['lateral_offset_m'],
            evaluation_episodes=result['episodes'], safe_success=result['safe_success'],
            task_success=result['success'], violation_rate=result['violation'],
            collision_rate=result['collision'], **{'return': result['episode_return']},
            time_to_goal_seconds=result['time_to_goal_seconds'],
            time_to_goal_successes=result['time_to_goal_successes'],
            time_to_goal_capped_seconds=result['time_to_goal_capped_seconds'],
            minimum_distance_m=result['minimum_distance_m'],
            safe_success_change_from_nominal=(result['safe_success'] - nominal[0]['safe_success']) if nominal else None,
            nominal_reused_without_new_rollout=result['nominal_reused_without_new_rollout'],
            policy_source_commit=result['policy_source_commit'],
            evaluation_source_commit=result['evaluation_source_commit'],
            native_run=str(run.relative_to(report)), evidence=str(path.relative_to(report)),
            run_evaluation_status='completed_all20_predeclared_conditions' if complete else 'partial_invocation_not_complete'))
    temporary = report / 'MAIN_ROBUSTNESS_RESULTS.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(report / 'MAIN_ROBUSTNESS_RESULTS.csv')
    summary = dict(condition_rows=len(rows), expected_condition_rows=len(expected),
        completed_run_evaluations=len(completed_pairs), expected_run_evaluations=len(pairs),
        policy_episodes=sum(r['evaluation_episodes'] for r in rows),
        expected_policy_episodes=plan['expected_policy_episodes'],
        training_transitions=0, adaptation_updates=0, new_evaluations_launched=0,
        historical_v2_table_unchanged='ROBUSTNESS_RESULTS.csv',
        complete=set(indexed) == expected and completed_pairs == pairs)
    temporary = report / 'MAIN_ROBUSTNESS_STATUS.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(report / 'MAIN_ROBUSTNESS_STATUS.json')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Optional SSH alias; otherwise export already mirrored JSON only')
    parser.add_argument('--remote-analysis', help='Parent containing per-run formal robustness output directories')
    parser.add_argument('--require-complete', action='store_true', help='Final audit: require all9 runs and180 conditions')
    args = parser.parse_args()
    if bool(args.host) != bool(args.remote_analysis):
        parser.error('Specify both --host and --remote-analysis, or neither')
    repo = Path(__file__).resolve().parents[3]
    report = repo / 'reports/star/arm_safe_entropy'
    evidence = report / 'main_robustness'
    if args.host:
        evidence.mkdir(exist_ok=True)
        subprocess.run(['rsync', '-a', '--include=*/', '--include=*.json', '--exclude=*',
            '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=8',
            args.host + ':' + args.remote_analysis.rstrip('/') + '/', str(evidence) + '/'],
            check=True, timeout=120)
    frozen = read_json(repo / 'configs/star/arm_safe_entropy/main_plan.json')
    print(json.dumps(export_results(report, evidence, frozen, args.require_complete)))


if __name__ == '__main__':
    main()
