"""Compact the completed Payload closure, retaining frozen identities and costs.

No model loading, training, simulation, candidate reselection or physical I/O.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: json.loads((args.input / name).read_text())
    receipt, selection, plan = (read(name + '.json') for name in ('receipt', 'selection', 'plan'))
    digest = hashlib.sha256((args.input / 'selection.json').read_bytes()).hexdigest()
    if not (receipt['status'] == 'completed' and receipt['selection_sha256'] == digest
            and receipt['selection_frozen_before_test'] is True
            and receipt['common_validation_episodes'] == 1500
            and receipt['independent_test_episodes'] == 600
            and receipt['training_transitions'] == receipt['physical_robot_interactions'] == 0
            and selection['independent_test_used_for_selection'] is False):
        raise ValueError('Require the genuine completed frozen Payload closure')
    if plan['checkpoint_steps'] != [60000, 120000, 180000, 240000, 300000]:
        raise ValueError('Preserve the exact five registered candidate opportunities')
    paths = [args.input / f'validation-{m.lower()}-{s}.json'
             for m in ('Iso', 'EA') for s in plan['checkpoint_steps']]
    paths += [args.input / f'independent-{m.lower()}.json' for m in ('Iso', 'EA')]
    summaries, episodes = [], []
    controls = substeps = queries = validation_episodes = 0
    for path in paths:
        result = json.loads(path.read_text())
        independent = result['split'] == 'independent_test'
        count, seed = (100, 71027) if independent else (50, 61027)
        if independent and (result['selection_sha256'] != digest or result['checkpoint_steps'] !=
                selection['methods'][result['method']]['candidate']['steps']):
            raise ValueError('Independent labels do not bind the pre-test identity')
        if len(result['conditions']) != 3 or result['new_training_transitions'] != 0:
            raise ValueError('Native three-condition frozen evaluation required')
        for condition in result['conditions']:
            records = condition['records']
            if len(records) != count or {r['seed'] for r in records} != set(range(seed, seed + count)):
                raise ValueError('Incomplete or replaced common layouts')
            if (sum(r['safe_success'] for r in records) != condition['safe_success_count']
                    or sum(r['violation'] for r in records) != condition['violation_count']
                    or sum(r['length'] for r in records) != condition['new_simulation_control_intervals']):
                raise ValueError('Episode evidence does not match the aggregate')
            if independent:
                summaries.append(dict(method=result['method'], selected_checkpoint_steps=result['checkpoint_steps'],
                    **{k: v for k, v in condition.items() if k != 'records'}))
                for record in records:
                    episodes.append(dict(method=result['method'], selected_checkpoint_steps=result['checkpoint_steps'],
                        condition=condition['condition'], **{k: v for k, v in record.items() if k != 'trajectory'}))
            else:
                validation_episodes += len(records)
        controls += result['new_simulation_control_intervals']
        substeps += result['new_physics_substeps']; queries += result['distance_queries']
    if (len(episodes), validation_episodes, controls, substeps, queries) != (
            600, 1500, receipt['native_simulation_controls'], receipt['native_physics_substeps'], receipt['distance_queries']):
        raise ValueError('Actual sample/query accounting mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('plan.json', 'selection.json', 'receipt.json', 'results.csv'):
        shutil.copy2(args.input / name, args.output / name)
    write_csv(args.output / 'independent_summary.csv', summaries)
    write_csv(args.output / 'independent_episodes.csv', episodes)
    for method, value in selection['methods'].items():
        record = dict(robot_family='Piper', task='Payload', method=method,
            selected_checkpoint_steps=value['candidate']['steps'], selection_split='validation',
            independent_test_used_for_selection=False, candidate_count=5,
            risk_qualified=value['candidate']['score']['risk_qualified'], status=value['selection_status'],
            score=value['candidate']['score'], original_selection_sha256=digest,
            actual_Payload_training_consumed_each_branch=300000, shared_Reach_parent_training_steps=300000,
            physical_deployment_eligible=False, hardware_calibration_pending=True)
        (args.output / f'selection-{method.lower()}.json').write_text(json.dumps(record, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    conditions = ('nominal', 'new_geometry', 'execution_scale_085')
    for method, color, offset in (('Iso', '#287ea6', -.18), ('EA', '#df8a37', .18)):
        rows = [next(r for r in summaries if r['method'] == method and r['condition'] == c) for c in conditions]
        steps = rows[0]['selected_checkpoint_steps'] // 1000
        for ax, key, title in zip(axes, ('safe_success', 'violation'), ('Safe success (%)', 'Safety-distance violation (%)')):
            bars = ax.bar([i + offset for i in range(3)], [r[key] * 100 for r in rows], .36,
                          color=color, label=f'{method} selected{steps}K')
            ax.bar_label(bars, fmt='%.0f', fontsize=9)
            ax.set_xticks(range(3), ('Nominal', 'Geometry', 'Scale0.85'))
            ax.set(ylabel=title, ylim=(0, 118)); ax.grid(axis='y', alpha=.2)
    figure.suptitle('Frozen-selected Piper Payload:100 independent layouts/condition, one training seed', fontsize=12)
    figure.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .91), ncol=2)
    figure.text(.5, .025, 'Both branches actually consumed300K Payload transitions + the same Reach parent. Not physical safety.',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .07, 1, .82))
    for suffix in ('png', 'pdf'):
        figure.savefig(args.output / f'independent_payload.{suffix}', dpi=170)
    plt.close(figure)
    (args.output / 'pack_receipt.json').write_text(json.dumps(dict(
        status='completed_postprocessing_only', original_selection_sha256=digest,
        independent_episode_rows=600, common_validation_episodes=validation_episodes,
        verified_original_native_controls=controls, verified_original_native_substeps=substeps,
        verified_original_distance_queries=queries, new_simulation_controls=0, new_training_transitions=0,
        physical_robot_interactions=0), indent=2) + '\n')
    print(json.dumps(dict(independent_episodes=600, original_native_controls=controls, new_training=0)))


if __name__ == '__main__':
    main()
