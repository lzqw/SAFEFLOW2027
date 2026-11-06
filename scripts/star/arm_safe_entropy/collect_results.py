"""Copy compact native arm evidence server -> host; never copy large replay."""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['ability', 'obstacle_development', 'mechanism_control', 'main'], default='ability')
    parser.add_argument('--host', default='tjucta-a100')
    parser.add_argument('--remote-project', default='/home/lzqw/SAFEFLOW2027_ARM')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    report = repo / 'reports/star/arm_safe_entropy'
    target = report / args.phase
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--include=*/', '--include=*.json', '--include=progress.csv',
                    '--exclude=*', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=8',
                    args.host + ':' + args.remote_project.rstrip('/') + '/runs/' + args.phase + '/', str(target) + '/'],
                   check=True, timeout=120)
    rows, main_rows, seen_main_pairs = [], [], set()
    main_freeze_record = None
    # Retain all mirrored versions: never drop weak runs or metric-select checkpoints.
    for phase in ['ability', 'obstacle_development', 'mechanism_control', 'main']:
        for config_path in sorted((report / phase).glob('*/run_config.json')):
            run = config_path.parent
            config = json.loads(config_path.read_text())
            is_main = config.get('stage') == 'main'
            if phase == 'main' and not is_main:
                raise ValueError('A development manifest in the main folder is not a formal result')
            if is_main:
                from scripts.star.arm_safe_entropy.run_protocol import resolve_phase, validate_main_freeze
                resolve_phase(config['method'], config['training_seed'], config['total_transition_budget'],
                              config['evaluation_episodes'], phase='main')
                validate_main_freeze(config.get('frozen_config') or {}, config['method'],
                                     config['training_seed'], config.get('reference_variance') or .04)
                if main_freeze_record is None:
                    main_freeze_record = config['frozen_config']
                elif main_freeze_record != config['frozen_config']:
                    raise ValueError('Do not combine different frozen configurations in one main comparison')
            receipt = json.loads((run / 'receipt.json').read_text()) if (run / 'receipt.json').exists() else {}
            lineage = json.loads((run / 'resume_lineage.json').read_text()) if (run / 'resume_lineage.json').exists() else {}
            progress = {}
            if (run / 'progress.csv').is_file():
                with (run / 'progress.csv').open(newline='') as stream:
                    for record in csv.DictReader(stream):
                        try:
                            progress[int(record['steps'])] = record
                        except (ValueError, TypeError):
                            continue  # A mutable live log may end in a partial row.
            for path in sorted(run.glob('evaluation_*.json')):
                e = json.loads(path.read_text())
                step = e['milestone_steps']
                final = (step == config['total_transition_budget'] and receipt.get('status') == 'completed'
                         and receipt.get('steps') == step)
                superseded = (run / 'engineering_stop.json').exists()
                row = dict(stage=config['stage'], task_version=config['task_config']['version'],
                    method=config['method'], geometry_source=config.get('geometry_source', 'none'),
                    oracle_backend=config.get('oracle_backend') or '',
                    training_seed=config['training_seed'], transitions=step,
                    evaluation_episodes=e['episodes'], safe_success=e['safe_success'], task_success=e['success'],
                    violation_rate=e['violation'], collision_rate=e['collision'], return_=e['episode_return'],
                    training_violation_count=e.get('training_violation_count', progress.get(step, {}).get('training_violation_count', '')),
                    training_violation_count_scope=('retained_checkpoint_lineage_only; lost_tail_not_included'
                        if lineage.get('lost_tail_violation_count_unknown') else 'native_run_counter'),
                    source_commit=config['source_commit'], run=str(run.relative_to(report)),
                    status='superseded_v1_engineering_pilot' if superseded else
                           'fixed_final_frozen_main' if is_main and final else
                           'final_native_not_formal_matrix' if final else 'interim_native')
                row['return'] = row.pop('return_')
                if is_main:
                    if final:
                        if step != 1000000 or e['episodes'] != 100:
                            raise ValueError('Main table is only the fixed final1M checkpoint,100 episodes per training seed')
                        start = config['evaluation_layout_seed_start']
                        if e['fixed_layout_seeds'] != list(range(start, start + 100)) or len(e['records']) != 100:
                            raise ValueError('Final main evaluation must retain all100 fixed common-layout episodes')
                        pair = (config['method'], config['training_seed'])
                        if pair in seen_main_pairs:
                            raise ValueError('Duplicate final main method/seed: do not silently choose the better run')
                        seen_main_pairs.add(pair)
                        main_rows.append(row)
                else:
                    rows.append(row)
    columns = ['stage', 'task_version', 'method', 'geometry_source', 'oracle_backend', 'training_seed', 'transitions', 'evaluation_episodes',
               'safe_success', 'task_success', 'violation_rate', 'collision_rate', 'return',
               'training_violation_count', 'training_violation_count_scope', 'source_commit', 'run', 'status']
    temporary = report / 'DEVELOPMENT_RESULTS.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(report / 'DEVELOPMENT_RESULTS.csv')
    main_columns = ['task_version', 'method', 'training_seed', 'transitions', 'evaluation_episodes',
                    'safe_success', 'task_success', 'violation_rate', 'collision_rate', 'return',
                    'training_violation_count', 'source_commit', 'run', 'status']
    temporary = report / 'MAIN_RESULTS.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=main_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(main_rows)
    temporary.replace(report / 'MAIN_RESULTS.csv')
    result = dict(host=args.host, remote_project=args.remote_project,
                          native_evaluations=len(rows), completed_main_rows=len(main_rows), expected_main_rows=9,
                          large_checkpoints_copied=0)
    if args.verbose:
        result['rows'] = [{key: row[key] for key in ['method', 'training_seed', 'transitions', 'task_success', 'violation_rate', 'status']} for row in rows]
    print(json.dumps(result))


if __name__ == '__main__':
    main()
