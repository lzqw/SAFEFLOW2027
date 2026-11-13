"""Paper figures from completed formal receipts and locked validation/test data.

No training, simulation, model selection, smoothing, or best-seed substitution.
Fixed10M summaries use three training seeds; selected independent tests remain
one selected policy/method and use five evaluation seeds only for display.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('sac_lag', 'stable_iso', 'ea')
LABELS = dict(sac_lag='SAC-Lag', stable_iso='Stable-Iso', ea='EA')
COLORS = dict(sac_lag='#b45309', stable_iso='#2563eb', ea='#15803d')
METRICS = ('mean_native_return', 'mean_episode_cost', 'mean_goals_per_episode', 'mean_budget_compliant_fraction')


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--independent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve published figures; choose an explicit new release')
    selection, independent = json.loads(args.selection.read_text()), json.loads(args.independent.read_text())
    if (selection['freeze_digest'] != independent['selection_freeze_digest'] or
            independent['independent_test_used_for_selection'] is not False):
        raise ValueError('Selection/test identity mismatch')
    candidates = selection['candidates']
    if len(candidates) != 45:
        raise ValueError('Retain the complete three methods xthree seeds xfive checkpoints')
    curves, fixed, efficiency, sources = [], [], [], []
    for candidate in candidates:
        row = {k:candidate[k] for k in ('method', 'run', 'training_seed', 'checkpoint_steps', *METRICS)}
        row.update(validation_seeds=5, validation_episodes_per_seed=50,
            role='fixed_common_validation_curve_not_independent_test')
        curves.append(row)
    for method in METHODS:
        for seed in (51027, 51028, 51029):
            pool = sorted([c for c in curves if c['method']==method and c['training_seed']==seed],
                          key=lambda c:c['checkpoint_steps'])
            if len(pool) != 5:
                raise ValueError('Do not replace a missing training-seed curve')
            endpoint = pool[-1]
            path = args.run_root/endpoint['run']/'receipt.json'
            receipt = json.loads(path.read_text())
            if receipt['status'] != 'completed' or receipt['steps'] != endpoint['checkpoint_steps']:
                raise ValueError('Completed fixed endpoint receipt mismatch')
            cost = receipt.get('total_training_cumulative_cost', receipt.get('training_cumulative_cost'))
            elapsed = receipt['elapsed_seconds']
            row = dict(endpoint, training_cumulative_cost=cost,
                total_training_seconds=elapsed, complete_training_sps=receipt['full_training_including_compile_eval_save_sps'])
            fixed.append(row)
            efficiency.append(dict(method=method, training_seed=seed, run=endpoint['run'],
                actual_transitions=receipt['steps'], total_training_seconds=elapsed,
                complete_training_sps=receipt['full_training_including_compile_eval_save_sps'],
                compile_reset_seconds=receipt.get('reset_compile_seconds'),
                compile_first_step_seconds=receipt.get('first_step_compile_seconds'),
                evaluation_seconds=receipt.get('evaluation_seconds'), save_seconds=receipt.get('save_seconds'),
                learner_updates=receipt.get('learner_updates'), reward_updates=receipt.get('reward_updates'),
                actor_updates=receipt.get('actor_updates'), base_flow_integration_steps=4 if method!='sac_lag' else None,
                training_cumulative_cost=cost, fixed10m_mean_goals=endpoint['mean_goals_per_episode'],
                fixed10m_mean_cost=endpoint['mean_episode_cost'],
                timing_scope='original complete training including compile/eval/save; shared load uncontrolled',
                batch1_full_inference_latency='separate frozen-policy measurement; not training SPS',
                source_commit=receipt['source_commit']))
            sources.append(dict(run=endpoint['run'], receipt_sha256=digest(path)))
    summary = []
    for method in METHODS:
        pool = [r for r in fixed if r['method']==method]
        row = dict(method=method, training_seeds=3, nominal_training_budget=10000000,
            role='fixed_endpoint_mean_and_sample_SD_across_training_seeds')
        for metric in (*METRICS, 'training_cumulative_cost', 'total_training_seconds', 'complete_training_sps'):
            values = np.array([r[metric] for r in pool], dtype=float)
            row[metric+'_mean'], row[metric+'_sd'] = float(values.mean()), float(values.std(ddof=1))
        summary.append(row)
    args.output.mkdir(parents=True)
    csv_write(args.output/'pointgoal_learning_curves.csv', curves)
    csv_write(args.output/'pointgoal_fixed_10m_by_seed.csv', fixed)
    csv_write(args.output/'pointgoal_fixed_10m_summary.csv', summary)
    csv_write(args.output/'pointgoal_training_efficiency.csv', efficiency)
    selected = []
    for candidate in independent['candidates']:
        selected.append({k:candidate[k] for k in ('method','run','training_seed','checkpoint_steps','episodes',*METRICS)} |
            dict(role='one_validation_selected_policy_per_method; not_three_training_seed_mean'))
    csv_write(args.output/'pointgoal_selected_independent.csv', selected)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout='constrained')
    for method in METHODS:
        matrix = []
        for seed in (51027, 51028, 51029):
            points = sorted((r for r in curves if r['method']==method and r['training_seed']==seed),
                            key=lambda r:r['checkpoint_steps'])
            matrix.append(points)
            for ax, metric in zip(axes, ('mean_goals_per_episode','mean_episode_cost')):
                ax.plot([r['checkpoint_steps']/1e6 for r in points], [r[metric] for r in points],
                        color=COLORS[method], alpha=.25, linewidth=.8)
        x = np.array([[r['checkpoint_steps']/1e6 for r in points] for points in matrix]).mean(0)
        for ax, metric in zip(axes, ('mean_goals_per_episode','mean_episode_cost')):
            values = np.array([[r[metric] for r in points] for points in matrix])
            center, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(x, center, color=COLORS[method], marker='o', markersize=4, label=LABELS[method])
            ax.fill_between(x, center-sd, center+sd, color=COLORS[method], alpha=.12)
    for ax, ylabel in zip(axes, ('Goals / native episode','Native episode cost')):
        ax.set(xlabel='Actual training transitions (millions)', ylabel=ylabel, xlim=(1.8,10.2))
        ax.grid(alpha=.18); ax.legend(frameon=False)
    axes[1].axhline(25, color='#666666', linestyle='--', linewidth=1)
    fig.suptitle('Fixed formal training: all three training seeds, five common validation checkpoints\n'
                 'Thin lines: each training seed; band: sample SD across three training seeds', fontsize=11)
    for suffix in ('png','pdf'):
        fig.savefig(args.output/f'pointgoal_three_seed_learning.{suffix}', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6,4.5), layout='constrained')
    for candidate in independent['candidates']:
        method = candidate['method']
        ax.scatter([r['episode_cost'] for r in candidate['per_seed']],
                   [r['goals_per_episode'] for r in candidate['per_seed']],
                   color=COLORS[method], alpha=.35, s=22)
        ax.scatter(candidate['mean_episode_cost'], candidate['mean_goals_per_episode'],
                   color=COLORS[method], s=70, marker='D', label=LABELS[method])
    ax.axvline(25, color='#666666', linestyle='--', linewidth=1, label='Cost budget25')
    ax.set(xlabel='Native episode cost (lower is better)', ylabel='Goals / native episode (higher is better)',
        title='Locked independent tests: one selected policy per method\n'
              'Diamonds:500-episode mean; dots:five test-seed means')
    ax.grid(alpha=.18); ax.legend(frameon=False)
    for suffix in ('png','pdf'):
        fig.savefig(args.output/f'pointgoal_selected_independent.{suffix}', dpi=180)
    plt.close(fig)
    (args.output/'figure_receipt.json').write_text(json.dumps(dict(training_transitions=0,
        physical_controls=0, selection_repeated=False, training_seeds_are_replication_unit=True,
        smoothing=False, missing_values_filled=False, selected_policy_replaces_fixed_endpoint=False,
        selection_sha256=digest(args.selection), independent_sha256=digest(args.independent),
        receipts=sources, shared_GPU_load_uncontrolled=True), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
