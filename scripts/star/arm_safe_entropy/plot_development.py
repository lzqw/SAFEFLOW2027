"""Compact native learning curves; versions/seeds are never silently averaged."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-version', default='SafeReach-Obstacle-v2')
    parser.add_argument('--phase', choices=['development', 'main'], default='development',
                        help='Main writes separate per-seed figures and an all-milestone table, not the final-only primary table')
    parser.add_argument('--include-controls', action='store_true',
                        help='Separate nominal-method/control figure; original R1 curves remain in the default development figure')
    args = parser.parse_args()
    if args.phase == 'main' and args.include_controls:
        parser.error('Do not pool development controls with the held-out main cohort')
    report = Path(__file__).resolve().parents[3] / 'reports/star/arm_safe_entropy'
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    episodes = 100 if args.phase == 'main' else 30
    titles = [f'Held-out safe success ({episodes} episodes/checkpoint)', 'Held-out violation rate',
              'Training recent success (last 100 completed episodes)', 'Cumulative training violations']
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=.2)
    included, excluded_engineering, milestone_rows = [], [], []
    phases = (['obstacle_development', 'mechanism_control'] if args.include_controls else ['obstacle_development'])
    if args.task_version == 'SafeReach-RelativeLayout-v3':
        phases = ['relative_layout_screening']
    if args.phase == 'main':
        phases = ['main']
    config_paths = [path for phase in phases for path in sorted((report / phase).glob('*/run_config.json'))]
    for config_path in config_paths:
        config = json.loads(config_path.read_text())
        run = config_path.parent
        if config['task_config']['version'] != args.task_version or (run / 'engineering_stop.json').exists():
            continue
        marker = run / 'engineering_interruption.json'
        if args.phase == 'main' and marker.exists() and json.loads(marker.read_text()).get('status') == 'stopped_before_first_evaluation_engineering_not_primary':
            excluded_engineering.append(dict(run=str(run.relative_to(report)),
                marker=str(marker.relative_to(report)), reason='Pre-evaluation deployment interruption; raw data retained, not a scientific result'))
            continue
        if args.include_controls and config.get('revision_id'):
            continue  # Predeclared nominal control comparison, not selection by outcome.
        with (run / 'progress.csv').open(newline='') as stream:
            progress = []
            for row in csv.DictReader(stream):
                try:
                    progress.append({k: float(row[k]) for k in
                        ['steps', 'recent_success', 'training_violation_count']})
                except (ValueError, TypeError):
                    continue
        if not progress:
            continue
        label = f"{config['method']} seed{config['training_seed']}"
        if (run / 'resume_lineage.json').exists():
            label += ' (resumed; lost-tail excluded)'
        elif (run / 'engineering_interruption.json').exists():
            label += ' (interrupted)'
        if config['method'] == 'flow_oracle':
            label += f" ({config.get('oracle_backend', 'exact')})"
        if config.get('revision_id'):
            label += f" [R1 rho={config.get('normal_variance_ratio', .25):g}]"
        base_color = {'flow_iso': 0, 'flow_learned': 2, 'flow_oracle': 4, 'sac_lag': 6,
                      'flow_iso_low': 8, 'flow_random_aniso': 10}.get(config['method'], 12)
        color_index = base_color + int(config['training_seed'] == 2028)
        color = plt.get_cmap('tab20')(color_index) if args.include_controls else f'C{color_index}'
        style = '--' if config.get('revision_id') else '-'
        marker = 's' if config.get('revision_id') else 'o'
        if args.phase == 'main':
            style, marker = {2030: ('-', 'o'), 2031: ('--', 's'), 2032: (':', '^')}[config['training_seed']]
        x = [row['steps'] / 1000 for row in progress]
        axes[1, 0].plot(x, [r['recent_success'] * 100 for r in progress], color=color, linestyle=style, label=label)
        axes[1, 1].plot(x, [r['training_violation_count'] for r in progress], color=color, linestyle=style)
        evaluations = sorted([json.loads(p.read_text()) for p in run.glob('evaluation_*.json')],
                             key=lambda e: e['milestone_steps'])
        if args.phase == 'main':
            for evaluation in evaluations:
                if evaluation['episodes'] != 100 or len(evaluation['records']) != 100:
                    raise ValueError('Do not label incomplete or development episodes as a main milestone')
                milestone_rows.append(dict(task_version=config['task_config']['version'], method=config['method'],
                    training_seed=config['training_seed'], transitions=evaluation['milestone_steps'],
                    evaluation_episodes=evaluation['episodes'], safe_success=evaluation['safe_success'],
                    violation_rate=evaluation['violation'], collision_rate=evaluation['collision'],
                    episode_return=evaluation['episode_return'], training_violation_count=evaluation['training_violation_count'],
                    time_to_goal_seconds=evaluation['time_to_goal_seconds'],
                    time_to_goal_successes=evaluation['time_to_goal_successes'],
                    time_to_goal_capped_seconds=evaluation['time_to_goal_capped_seconds'],
                    minimum_distance_m=evaluation['minimum_distance_m'],
                    source_commit=config['source_commit'], run=str(run.relative_to(report)),
                    status='predeclared_main_milestone_not_primary_table'))
        for axis, key in zip(axes[0], ['safe_success', 'violation']):
            if evaluations:
                axis.plot([e['milestone_steps'] / 1000 for e in evaluations],
                          [e[key] * 100 for e in evaluations], marker=marker, linestyle=style, color=color, label=label)
        included.append(dict(run=str(run.relative_to(report)), source_commit=config['source_commit'],
            latest_logged_transitions=int(progress[-1]['steps']),
            evaluation_milestones=[e['milestone_steps'] for e in evaluations]))
    for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        axis.set_ylim(-3, 103)
        axis.set_ylabel('%')
    for axis in axes[1]:
        axis.set_xlabel('Retained training transitions (thousands)')
    if included:
        handles, labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=7)
    scope = 'nominal development + mechanism controls' if args.include_controls else 'interim development'
    if args.phase == 'main':
        scope = 'frozen main, predeclared milestones'
    fig.suptitle(f'{args.task_version}: {scope}, per seed; no smoothing or extrapolation',
                 fontsize=10 if args.include_controls else 11)
    fig.tight_layout(rect=(0, .18 if included else 0, 1, .96))
    destination = report / 'figures'
    destination.mkdir(exist_ok=True)
    stem = f'{args.task_version}-mechanism-learning' if args.include_controls else f'{args.task_version}-learning'
    if args.phase == 'main':
        stem = f'{args.task_version}-main-learning'
        if milestone_rows:
            temporary = report / 'MAIN_LEARNING_RESULTS.csv.tmp'
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(milestone_rows[0]))
                writer.writeheader()
                writer.writerows(milestone_rows)
            temporary.replace(report / 'MAIN_LEARNING_RESULTS.csv')
    fig.savefig(destination / f'{stem}.png', dpi=150)
    plt.close(fig)
    (destination / f'{stem}.json').write_text(json.dumps(dict(
        task_version=args.task_version, phases=phases, runs=included, excluded_engineering_runs=excluded_engineering,
        comparison_scope=('Frozen held-out main runs; native fixed milestones only, original engineering attempts retained separately'
                          if args.phase == 'main' else
                          'Nominal methods and requested controls; R1 retained separately in the default development figure'
                          if args.include_controls else 'All development variants, including R1'),
        note='Top panels: stochastic held-out evaluation only. Bottom panels: training statistics, not held-out performance. Missing checkpoints are not assigned zeros.'), indent=2) + '\n')
    print(json.dumps(dict(task_version=args.task_version, plotted_runs=included)))


if __name__ == '__main__':
    main()
