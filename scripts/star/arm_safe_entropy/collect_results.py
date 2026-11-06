"""Copy compact native arm evidence server -> host; never copy large replay."""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['ability', 'obstacle_development'], default='ability')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    report = repo / 'reports/star/arm_safe_entropy'
    target = report / args.phase
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--include=*/', '--include=*.json', '--include=progress.csv',
                    '--exclude=*', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=8',
                    'tjucta-a100:/home/lzqw/SAFEFLOW2027_ARM/runs/' + args.phase + '/', str(target) + '/'],
                   check=True, timeout=120)
    rows = []
    # Retain all mirrored versions: never drop weak runs or metric-select checkpoints.
    for phase in ['ability', 'obstacle_development']:
        for config_path in sorted((report / phase).glob('*/run_config.json')):
            run = config_path.parent
            config = json.loads(config_path.read_text())
            receipt = json.loads((run / 'receipt.json').read_text()) if (run / 'receipt.json').exists() else {}
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
                final = step == config['total_transition_budget'] and receipt.get('status') == 'completed'
                rows.append(dict(stage=config['stage'], task_version=config['task_config']['version'],
                    method=config['method'], training_seed=config['training_seed'], transitions=step,
                    evaluation_episodes=e['episodes'], safe_success=e['safe_success'], task_success=e['success'],
                    violation_rate=e['violation'], collision_rate=e['collision'], return_=e['episode_return'],
                    training_violation_count=progress.get(step, {}).get('training_violation_count', ''),
                    source_commit=config['source_commit'], run=str(run.relative_to(report)),
                    status='final_native_not_formal_matrix' if final else 'interim_native'))
    columns = ['stage', 'task_version', 'method', 'training_seed', 'transitions', 'evaluation_episodes',
               'safe_success', 'task_success', 'violation_rate', 'collision_rate', 'return',
               'training_violation_count', 'source_commit', 'run', 'status']
    for row in rows:
        row['return'] = row.pop('return_')
    temporary = report / 'DEVELOPMENT_RESULTS.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(report / 'DEVELOPMENT_RESULTS.csv')
    print(json.dumps(dict(native_evaluations=len(rows), large_checkpoints_copied=0,
                          rows=[{key: row[key] for key in ['method', 'training_seed', 'transitions', 'task_success', 'violation_rate', 'status']} for row in rows])))


if __name__ == '__main__':
    main()
