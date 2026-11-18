"""All completed Piper DEV endpoints: task gains versus actual training risk."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def endpoint(directory, run, condition):
    rows = read(directory/'dev_nodes.csv')
    value = next(r for r in rows if r['run'] == run and
                 r['new_steps'] == '100000' and r['condition'] == condition)
    receipt = json.loads((directory/'receipt.json').read_text())
    training = next(r for r in receipt['completed_native_receipts'] if r['run'] == run)
    if training['new_training_steps'] != 100000:
        raise ValueError('Use the complete registered endpoint, not the best intermediate node')
    return training['segment_counts']['violations'], 100*int(value['safe_successes'])/int(value['episodes'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'figure.dpi': 160,
                         'axes.spines.top': False, 'axes.spines.right': False})
    cases = [
        ('round1_analysis', 'R0-fixed', 'Fixed P5', '#009E73'),
        ('round1_analysis', 'R1-ramp50k', '0 to1 /50K', '#CC79A7'),
        ('round1_analysis', 'R1-ramp75k', '0 to1 /75K', '#E69F00'),
        ('round2_analysis', 'R1-quarter75k', '.25 to1 /75K', '#56B4E9'),
        ('round2_analysis', 'R2-activation2mm', '+2mm, fixed', '#D55E00'),
        ('round3_analysis', 'R3-activation2mm-quarter75k', '+2mm, .25 to1 /75K', '#882255'),
        ('round3_analysis', 'Metric-only', 'Metric-only', '#333333')]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), constrained_layout=True)
    points = []
    for ax, condition, target, title in zip(axes, ('radius_0575', 'radius_060'),
            (93, 90), ('Reach57.5mm', 'Reach60mm')):
        for directory, arm, label, color in cases:
            risk, success = endpoint(args.reports/directory, 'piper_reach-'+arm, condition)
            ax.scatter(risk, success, color=color, s=65, edgecolor='white', linewidth=.6, label=label, zorder=3)
            # Same risk/success values can overlap; all are preserved in the table.
            points.append(dict(task='Reach', arm=arm, condition=condition,
                               training_violations=risk, dev_safe_success_percent=success))
        ax.axvline(58*1.1, color='#777777', linestyle='--', linewidth=1)
        ax.axhline(target, color='#777777', linestyle=':', linewidth=1)
        ax.axvspan(0, 58*1.1, ymin=(target-65)/35, ymax=1, color='#009E73', alpha=.09)
        ax.set(xlim=(0, 185), ylim=(65, 100), title=title,
               xlabel='Training violations (new100K)', ylabel='DEV safe success (%)')
        ax.grid(alpha=.15)
    axes[0].legend(loc='lower left', fontsize=8)
    axes[1].text(4, 98, 'Joint target region\n(risk <=1.10 x fixed P5)', fontsize=8, va='top')
    fig.suptitle('Piper Reach:all seven completed DEV endpoints;one common training parent')
    for suffix in ('png', 'pdf'): fig.savefig(args.output/f'reach_task_risk_pareto.{suffix}')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    for arm, label, color in [('Payload-P5-fixed', 'Fixed P5', '#009E73'),
                              ('Payload-activation2mm', '+2mm activation', '#D55E00'),
                              ('Payload-Metric-only', 'Metric-only', '#333333')]:
        for condition, marker in [('radius_055', 'o'), ('radius_0575', '^')]:
            risk, success = endpoint(args.reports/'payload_round1_analysis', 'piper_payload-'+arm, condition)
            radius = {'radius_055': '55mm', 'radius_0575': '57.5mm'}[condition]
            ax.scatter(risk, success, color=color, marker=marker, s=70, label=label+' /'+radius)
            points.append(dict(task='Payload', arm=arm, condition=condition,
                               training_violations=risk, dev_safe_success_percent=success))
    ax.axvline(136*1.1, color='#777777', linestyle='--', linewidth=1)
    ax.set(xlim=(80, 215), ylim=(55, 100), xlabel='Training violations (new100K)',
           ylabel='DEV safe success (%)', title='Payload:three completed DEV arms;no independent test')
    ax.grid(alpha=.15); ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('Task/risk gains do not establish timeout/stall nonregression')
    for suffix in ('png', 'pdf'): fig.savefig(args.output/f'payload_task_risk_pareto.{suffix}')
    plt.close(fig)
    with (args.output/'all_endpoints.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0])); writer.writeheader(); writer.writerows(points)
    (args.output/'provenance.json').write_text(json.dumps(dict(
        endpoint='all completed new100K,no intermediate selection',
        native_training_seed_count=1, episodes_per_condition=100,
        uncertainty='descriptive only;no training-seed uncertainty or statistical significance claim',
        points=len(points), independent_test_used=False, new_training_steps=0, new_simulation_controls=0), indent=2)+'\n')


if __name__ == '__main__': main()
