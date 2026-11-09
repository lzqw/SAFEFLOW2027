"""Export the completed predeclared episode comparison; never rerun physics."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.input / 'receipt.json').read_text())
    if receipt['status'] != 'completed' or receipt['policy_episodes'] != 2000:
        parser.error('Export only the complete2000-episode comparison, not a favorable subset')
    plan = receipt['plan']
    results = [json.loads(path.read_text()) for path in sorted(args.input.glob('*-seed*.json'))]
    expected = {(method, seed, condition['name']) for method in plan['training_methods']
                for seed in plan['training_seeds'] for condition in plan['conditions']}
    if (len(results) != 40 or {(r['method'], r['training_seed'], r['condition']['name']) for r in results} != expected
            or any(r['episodes'] != 50 or len(r['records']) != 50 for r in results)):
        parser.error('Missing or duplicate method/seed/condition, or incomplete native episodes')
    indexed = {(r['method'], r['training_seed'], r['condition']['name']): r for r in results}
    rows = []
    for result in results:
        condition = result['condition']
        nominal = indexed[(result['method'], result['training_seed'], 'nominal')]
        rows.append(dict(task_version=plan['task_version'], method=result['method'],
            training_seed=result['training_seed'], perturbation=condition['family'], severity=condition['severity'],
            condition=condition['name'], radius_m=condition['radius_m'], lateral_offset_m=condition['lateral_offset_m'],
            evaluation_episodes=result['episodes'], safe_success=result['safe_success'],
            task_success=result['success'], violation_rate=result['violation'], collision_rate=result['collision'],
            time_to_goal_seconds=result['time_to_goal_seconds'], time_to_goal_successes=result['time_to_goal_successes'],
            time_to_goal_capped_seconds=result['time_to_goal_capped_seconds'],
            return_=result['episode_return'], minimum_distance_m=result['minimum_distance_m'],
            safe_success_change_from_nominal=result['safe_success']-nominal['safe_success'],
            source_commit=result['policy_source_commit'], evaluation_source_commit=result['evaluation_source_commit'],
            checkpoint=result['checkpoint'], checkpoint_steps=result['checkpoint_steps'],
            training_counter_scope='retained_checkpoint_lineage_only; lost_tail_not_included' if
                (result.get('resume_lineage') or {}).get('lost_tail_violation_count_unknown') else 'native_run_counter',
            status='fixed_final300k_frozen_zero_shot_50_native_episodes'))
        rows[-1]['return'] = rows[-1].pop('return_')
    report = Path(__file__).resolve().parents[3] / 'reports/star/arm_safe_entropy'
    temporary = report / 'ROBUSTNESS_RESULTS.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(report / 'ROBUSTNESS_RESULTS.csv')
    summaries = []
    perturbed = [c['name'] for c in plan['conditions'] if c['name'] != 'nominal']
    for method in plan['training_methods']:
        for seed in plan['training_seeds']:
            native = [indexed[(method, seed, condition)] for condition in perturbed]
            summaries.append(dict(method=method, training_seed=seed, perturbed_episodes=200,
                nominal_safe_success=indexed[(method, seed, 'nominal')]['safe_success'],
                perturbed_safe_success=sum(r['safe_success'] for r in native)/4,
                perturbed_violation_rate=sum(r['violation'] for r in native)/4,
                perturbed_horizon_capped_time_seconds=sum(r['time_to_goal_capped_seconds'] for r in native)/4))
    (args.input / 'comparison_summary.json').write_text(json.dumps(dict(
        policy_episodes=2000, training_transitions=0, summaries=summaries,
        caution='Equal-weight predeclared conditions; two development seeds, not independent formal-seed evidence; no adaptive stronger conditions'), indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    labels = [c['name'] for c in plan['conditions']]
    for col, seed in enumerate(plan['training_seeds']):
        for index, method in enumerate(plan['training_methods']):
            values = [indexed[(method, seed, condition)] for condition in labels]
            for row, metric in enumerate(['safe_success', 'violation', 'time_to_goal_capped_seconds']):
                axes[row, col].plot(labels, [r[metric] for r in values], marker='o',
                                    color=f'C{index}', label=method)
        axes[0, col].set_title(f'Fixed300k seed{seed};50 common episodes/condition')
        for row, title in enumerate(['Safe success', 'Violation episode rate', 'Horizon-capped time to goal (s)']):
            axes[row, col].set_ylabel(title)
            axes[row, col].grid(alpha=.2)
            if row < 2:
                axes[row, col].set_ylim(-.03, 1.03)
        axes[2, col].tick_params(axis='x', rotation=25)
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='lower center', ncol=4)
    fig.suptitle('Frozen native policies: obstacle-size and balanced-position shifts\nNo retraining, no checkpoint selection, no adaptive stress tuning')
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(args.input / 'obstacle-robustness.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(canonical_rows=len(rows), policy_episodes=2000, training_transitions=0,
                          summaries=summaries)))


if __name__ == '__main__':
    main()
