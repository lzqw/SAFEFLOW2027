"""Plot existing fixed formal tests without pooling training seeds or rerunning physics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHODS = [('sac_lag', 'SAC-Lag', 'C6'), ('flow_iso', 'Flow-Iso', 'C0'),
           ('flow_learned', 'LearnedDir', 'C2')]
METRICS = [('safe_success', 'Safe success (%)', 100.),
           ('violation_rate', 'Violation episode rate (%)', 100.),
           ('time_to_goal_capped_seconds', 'Horizon-capped time-to-goal (s)', 1.)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true',
                        help='Final delivery only: refuse a figure lacking any of the nine completed run evaluations')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    report = repo / 'reports/star/arm_safe_entropy'
    frozen = json.loads((repo / 'configs/star/arm_safe_entropy/main_plan.json').read_text())
    plan = frozen['robustness_plan']
    source = report / 'MAIN_ROBUSTNESS_RESULTS.csv'
    with source.open(newline='') as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if r['run_evaluation_status'] == 'completed_all20_predeclared_conditions']
    conditions = [c['name'] for c in plan['conditions']]
    stress_conditions = [c for c in conditions if c != 'nominal']
    steps = frozen['evaluation_steps']
    expected_pairs = {(method, seed) for method in frozen['methods'] for seed in frozen['training_seeds']}
    indexed, pairs = {}, set()
    for row in rows:
        pair = (row['method'], int(row['training_seed']))
        key = (*pair, int(row['checkpoint_steps']), row['condition'])
        if (pair not in expected_pairs or key[2] not in steps or key[3] not in conditions
                or row['task_version'] != frozen['task_version']
                or int(row['evaluation_episodes']) != frozen['evaluation_episodes'] or key in indexed):
            parser.error('Use the canonical non-duplicated frozen formal episode table')
        indexed[key] = row
        pairs.add(pair)
    expected_cases = {(step, condition) for step in steps for condition in conditions}
    for pair in pairs:
        if {(key[2], key[3]) for key in indexed if key[:2] == pair} != expected_cases:
            parser.error('A completed run evaluation must retain all four checkpoints and five conditions')
    if not pairs or (args.require_complete and pairs != expected_pairs):
        parser.error(f'Only {len(pairs)}/9 complete formal run evaluations are available; never fill missing results with zeros')
    seeds = [seed for seed in frozen['training_seeds'] if any(p[1] == seed for p in pairs)]
    destination = report / 'figures'
    destination.mkdir(exist_ok=True)
    stem = frozen['task_version'] + '-main-robustness'
    curve_points = []
    for method, seed in sorted(pairs):
        for step in steps:
            records = [indexed[(method, seed, step, condition)] for condition in stress_conditions]
            count = sum(int(r['evaluation_episodes']) for r in records)
            successes = sum(int(r['time_to_goal_successes']) for r in records)
            successful_time = (sum(float(r['time_to_goal_seconds'])*int(r['time_to_goal_successes'])
                for r in records if int(r['time_to_goal_successes'])) / successes) if successes else None
            curve_points.append(dict(method=method, training_seed=seed, checkpoint_steps=step,
                stress_episodes=count, time_to_goal_successes=successes,
                time_to_goal_seconds=successful_time,
                **{metric: sum(float(r[metric])*int(r['evaluation_episodes'])
                    for r in records)/count for metric, _, _ in METRICS}))
    curve_index = {(r['method'], r['training_seed'], r['checkpoint_steps']): r for r in curve_points}
    labels = {'nominal': 'Nominal', 'radius_070': 'Radius7cm', 'radius_080': 'Radius8cm',
              'lateral_030': 'Lateral±3cm', 'lateral_060': 'Lateral±6cm'}
    artifacts = []
    for learning in [False, True]:
        fig, axes = plt.subplots(len(seeds), 3, figsize=(12, 3.1*len(seeds)+.9), squeeze=False)
        legends = {}
        for row_index, seed in enumerate(seeds):
            available = sum((method, seed) in pairs for method in frozen['methods'])
            for col, (metric, title, scale) in enumerate(METRICS):
                axis = axes[row_index, col]
                for method, label, color in METHODS:
                    if (method, seed) not in pairs:
                        continue
                    if learning:
                        x = [step/1000 for step in steps]
                        y = [curve_index[(method, seed, step)][metric]*scale for step in steps]
                    else:
                        x = list(range(len(conditions)))
                        y = [float(indexed[(method, seed, frozen['steps_per_seed'], condition)][metric])*scale
                             for condition in conditions]
                    handle, = axis.plot(x, y, marker='o', color=color, label=label)
                    legends[label] = handle
                short_title = 'Capped time-to-goal (s)' if scale == 1. else title
                axis.set_title(f'Seed{seed} — {available}/3 methods\n{short_title}', fontsize=9)
                axis.grid(alpha=.2)
                axis.set_ylim((-.03, 4.1) if scale == 1. else (-3., 103.))
                if learning:
                    axis.set_xticks([step/1000 for step in steps])
                    axis.set_xlabel('Training transitions (×1000)')
                else:
                    axis.set_xticks(range(len(conditions)), [labels[c] for c in conditions], rotation=20)
        ordered = [label for _, label, _ in METHODS if label in legends]
        fig.legend([legends[label] for label in ordered], ordered, loc='lower center', ncol=len(ordered))
        scope = 'Four predeclared stress conditions,400 episodes/point' if learning else 'Fixed1M primary,100 episodes/condition'
        fig.suptitle(f'{scope}; {len(pairs)}/9 run evaluations complete\n'
                     'Per training seed; missing results are not zeros; no checkpoint selection', fontsize=10)
        fig.tight_layout(rect=(0, .09, 1, .96))
        filename = stem + ('-learning.png' if learning else '-fixed-final.png')
        fig.savefig(destination / filename, dpi=150)
        plt.close(fig)
        artifacts.append(str((destination / filename).relative_to(report)))
    provenance = dict(task_version=frozen['task_version'], source=str(source.relative_to(report)),
        complete=pairs == expected_pairs, completed_run_evaluations=len(pairs), expected_run_evaluations=9,
        plotted_method_seeds=[dict(method=method, training_seed=seed) for method, seed in sorted(pairs)],
        pending_method_seeds=[dict(method=method, training_seed=seed) for method, seed in sorted(expected_pairs-pairs)],
        complete_condition_rows=len(rows), excluded_incomplete_condition_rows=len(all_rows)-len(rows),
        excluded_incomplete_scope='Only receipt completion determines inclusion,never reward or violation outcome; raw incomplete rows remain in the canonical CSV',
        evaluation_steps=steps, primary_checkpoint_steps=frozen['steps_per_seed'], conditions=plan['conditions'],
        stress_curve_points=curve_points, artifacts=artifacts,
        note='Fixed-final figure uses all5conditions. Learning figure equally weights the four predeclared100-episode stress conditions within each method/seed/checkpoint,never pools training seeds; nominal is retained in the CSV and native learning figure. Time includes failures at the4s cap,not just successes.',
        training_transitions=0, new_evaluation_episodes=0)
    (destination / (stem + '.json')).write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(completed_run_evaluations=len(pairs), expected_run_evaluations=9,
        plotted_training_seeds=seeds, artifacts=artifacts, new_evaluation_episodes=0, training_transitions=0)))


if __name__ == '__main__':
    main()
