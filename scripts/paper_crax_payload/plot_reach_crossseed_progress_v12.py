"""Plot every archived cross-seed DEV node and training prefix; no new physics."""
import argparse
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.paper_crax_payload.plot_simulation_v12 import read
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256


ARMS = {'P5-fixed': ('Fixed P5', '#666666'),
        'P5-activation2mm': ('P5 + 2 mm', '#009E73'),
        'Metric-only': ('Metric-only', '#D55E00')}
STYLES = {41028: ('--', 'D'), 41029: ('-', 'o')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--resource', type=Path, required=True)
    parser.add_argument('--host-summary', type=Path, action='append', default=[])
    parser.add_argument('--counter-label', default='10:51 resource counter')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    nodes, curves, sources = {}, {}, []
    for directory in sorted(args.reports.glob('crossseed_dev*K_4102?_20261120*')):
        match = re.fullmatch(r'crossseed_dev(\d+)K_(4102[89])_20261120(?:_\w+)?', directory.name)
        if not match:
            continue
        step, seed = int(match[1])*1000, int(match[2])
        receipt = json.loads((directory/'receipt.json').read_text())
        if (receipt['status'] != 'partial_crossseed_development_snapshot'
                or receipt['training_seed'] != seed or receipt['fixed_new_steps'] != step
                or receipt['independent_test_used'] is not False
                or receipt['winner_confirmation'] is not False):
            raise ValueError('Only the exact archived fixed-node development snapshots')
        for row in read(directory/'development_nodes.csv'):
            key = (seed, row['arm'], row['deployment_mode'], row['condition'], int(row['new_steps']))
            if (key in nodes and nodes[key] != row) or int(row['episodes']) != 100:
                raise ValueError('Repeated DEV evidence must agree; no mixed modes or reduced episodes')
            nodes[key] = row
        for arm in receipt['available_arms']:
            rows = [r for r in read(directory/'training.csv') if r['arm'] == arm]
            if [int(r['new_steps']) for r in rows] != list(range(1000, step+1, 1000)):
                raise ValueError('Whole native1K prefix required')
            key = (seed, arm)
            if key not in curves or step > int(curves[key][-1]['new_steps']):
                curves[key] = rows  # Longest observed prefix, never best-performing node.
        sources.append(dict(directory=directory.name,
            receipt_sha256=sha256(directory/'receipt.json'),
            nodes_sha256=sha256(directory/'development_nodes.csv'),
            training_sha256=sha256(directory/'training.csv')))
    for directory in args.host_summary:
        receipt = json.loads((directory/'receipt.json').read_text())
        seed = receipt['training_seed']
        if (receipt['status'] != 'complete_crossseed_one_host_development_summary'
                or seed not in STYLES or receipt['completed_runs'] != 3
                or receipt['development_nodes'] != 96
                or receipt['independent_test_used'] is not False
                or receipt['winner_confirmation'] is not False):
            raise ValueError('Require the complete native host development summary')
        for row in read(directory/'development_nodes.csv'):
            key = (seed, row['arm'], row['deployment_mode'], row['condition'], int(row['new_steps']))
            if (int(row['training_seed']) != seed or int(row['episodes']) != 100
                    or (key in nodes and nodes[key] != row)):
                raise ValueError('Complete and partial archived evidence must agree')
            nodes[key] = row
        training = read(directory/'training.csv')
        for arm in ARMS:
            rows = [r for r in training if r['arm'] == arm]
            if [int(r['new_steps']) for r in rows] != list(range(1000, 150001, 1000)):
                raise ValueError('Require the whole completed150K native prefix')
            old = curves.get((seed, arm), [])
            if old and old != rows[:len(old)]:
                raise ValueError('Completed training must preserve previously archived prefixes')
            curves[(seed, arm)] = rows
        sources.append(dict(directory=directory.name,
            receipt_sha256=sha256(directory/'receipt.json'),
            nodes_sha256=sha256(directory/'development_nodes.csv'),
            training_sha256=sha256(directory/'training.csv')))
    resource, resource_sources = {}, []
    for path in sorted(args.resource.glob('*.json')):
        value = json.loads(path.read_text())
        for job in value['jobs']:
            match = re.fullmatch(r'Reach-(4102[89])-(.+)', job['name'])
            if match and job['progress']:
                resource[(int(match[1]), match[2])] = job['progress']
        resource_sources.append(dict(file=path.name, sha256=sha256(path), unix_time=value['unix_time']))
    if not nodes or not curves:
        raise ValueError('Need actual archived results, not inferred or fabricated values')
    max_step = max([int(rows[-1]['new_steps']) for rows in curves.values()] +
                   [int(row['new_steps']) for row in resource.values()])
    xmax = max(20, math.ceil(max_step/20000)*20)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'figure.dpi': 150,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    fig.subplots_adjust(left=.055, right=.985, top=.89, bottom=.22, hspace=.42, wspace=.30)
    for ax, condition, title in zip(axes[0],
            ('nominal', 'radius_0575', 'radius_060', 'execution_scale_085'),
            ('Nominal', 'Radius 57.5 mm', 'Radius 60 mm', 'Execution scale 0.85')):
        for seed, arm in sorted(curves):
            label, color = ARMS[arm]; style, marker = STYLES[seed]
            rows = sorted((r for key, r in nodes.items() if key[:4] == (seed, arm, 'sampled', condition)),
                          key=lambda r: int(r['new_steps']))
            ax.plot([int(r['new_steps'])/1000 for r in rows],
                    [100*int(r['safe_successes'])/int(r['episodes']) for r in rows],
                    color=color, linestyle=style, marker=marker, markersize=4)
        if condition in ('radius_0575', 'radius_060'):
            target = 93 if condition == 'radius_0575' else 90
            ax.axhline(target, color='#999999', linestyle=':', linewidth=1)
            ax.text(xmax*.98, target-4, f'Target {target}%', ha='right', fontsize=8, color='#777777')
        ax.set_title(title); ax.set_ylabel('DEV safe success (%)'); ax.set_ylim(0, 104)
    counter_plotted = False
    for ax, metric, title in zip(axes[1],
            ('violations', 'training_cumulative_return', 'safe_timeouts', 'training_stall_episodes'),
            ('Cumulative training violations', 'Cumulative sampled training return',
             'Cumulative training timeouts', 'Stalled training episodes')):
        for (seed, arm), rows in sorted(curves.items()):
            label, color = ARMS[arm]; style, marker = STYLES[seed]
            ax.plot([int(r['new_steps'])/1000 for r in rows], [float(r[metric]) for r in rows],
                    color=color, linestyle=style, linewidth=1.7)
            point = resource.get((seed, arm))
            if point and int(point['new_steps']) > int(rows[-1]['new_steps']):
                ax.scatter([int(point['new_steps'])/1000], [float(point[metric])], color=color, marker='x', s=35)
                counter_plotted = True
        ax.set_title(title); ax.set_ylim(bottom=0)
    for ax in axes.flat:
        ax.set_xlim(0, xmax); ax.grid(alpha=.2)
    handles = [Line2D([], [], color=ARMS[arm][1], linestyle=STYLES[seed][0],
               marker=STYLES[seed][1], label=f'{ARMS[arm][0]} | seed {seed}') for seed, arm in sorted(curves)]
    if counter_plotted:
        handles.append(Line2D([], [], color='#555555', marker='x', linestyle='', label=args.counter_label))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=3, fontsize=9)
    fig.suptitle('Reach development: two-seed campaign | sampled deployment | unequal observed progress', fontsize=13)
    fig.text(.5, .16, 'New training transitions after each 400K parent (thousands)', ha='center', fontsize=10)
    footer = 'All archived fixed DEV nodes and full 1K prefixes. No independent test or checkpoint selection.'
    if counter_plotted:
        footer += ' x markers: isolated later resource counters.'
    fig.text(.5, .025, footer, ha='center', fontsize=9)
    args.output.mkdir(parents=True, exist_ok=False)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'reach_crossseed_progress.{suffix}')
    plt.close(fig)
    (args.output/'provenance.json').write_text(json.dumps(dict(
        status='partial_existing_data_learning_curves', sources=sources, resource_sources=resource_sources,
        archived_dev_nodes=len(nodes), plotted_training_prefixes={f'{s}/{a}':int(r[-1]['new_steps'])
            for (s,a),r in curves.items()}, training_seed_count=len({s for s,a in curves}),
        deployment_mode='sampled', metric_pending=any((s, 'Metric-only') not in curves for s in STYLES),
        complete_campaign=len(args.host_summary) == 2,
        new_training_steps=0, new_simulation_controls=0, independent_test_used=False,
        best_checkpoint_selected=False, physical_robot_interactions=0,
        connected_nodes_do_not_prove_continuous_dominance=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
