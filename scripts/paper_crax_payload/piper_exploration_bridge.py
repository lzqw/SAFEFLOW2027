"""Descriptive E05 bridge from existing logs/trajectories; no simulation or RL."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


COUNTERS = ('training_safe_successes','training_violation_count','training_safe_timeouts','episodes')


def load_curve(paths):
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if [r['steps'] for r in rows] != list(range(1000,300001,1000)):
        raise ValueError('Require the complete matched300K logs, no missing/duplicate grid points')
    previous = dict.fromkeys(COUNTERS,0)
    for row in rows:
        if any(row[k] < previous[k] for k in COUNTERS):
            raise ValueError('A cumulative counter reset would invalidate this bridge')
        if sum(row[k] for k in COUNTERS[:3]) != row['episodes']:
            raise ValueError('Training terminal classes must partition completed episodes')
        previous = row
    return [dict(steps=0, **dict.fromkeys(COUNTERS,0))] + rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def sample_polyline(points, spacing=.005):
    """Fixed arclength samples rather than giving slower control more samples."""
    points = np.asarray(points,dtype=float)
    lengths = np.linalg.norm(np.diff(points,axis=0),axis=1)
    cumulative = np.concatenate([[0.],np.cumsum(lengths)])
    unique, indices = np.unique(cumulative,return_index=True)
    if len(unique) < 2:
        return points[:1],0.
    positions = np.arange(0.,unique[-1],spacing)
    positions = np.append(positions,unique[-1])
    return np.column_stack([np.interp(positions,unique,points[indices,j]) for j in range(3)]),float(unique[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',type=Path,required=True)
    parser.add_argument('--closure',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve every descriptive analysis and its declared choices')
    args.output.mkdir(parents=True)
    curves, sources = {}, {}
    for method in ('iso','ea'):
        paths = [args.logs/f'{method}-initial100k.jsonl',args.logs/f'{method}-continuation300k.jsonl']
        curves[method] = load_curve(paths)
        sources.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    learning, fixed, budgets = [], [], []
    for method, curve in curves.items():
        learning.extend(dict(method=method,**{k:r[k] for k in ('steps',*COUNTERS)}) for r in curve)
        for target in (50000,100000,150000,200000,250000,300000):
            row = next(r for r in curve if r['steps']==target)
            fixed.append(dict(method=method,**{k:row[k] for k in ('steps',*COUNTERS)}))
        for cap in (100,200,400,600):
            row = max((r for r in curve if r['training_violation_count']<=cap),key=lambda r:r['steps'])
            budgets.append(dict(method=method,violation_budget=cap,
                last_logged_steps_within_budget=row['steps'],actual_logged_violations=row['training_violation_count'],
                cumulative_safe_successes=row['training_safe_successes'],log_grid_resolution_steps=1000,
                interpolation_used=False))
    write_csv(args.output/'learning_counters.csv',learning)
    write_csv(args.output/'fixed_transitions.csv',fixed)
    write_csv(args.output/'within_violation_budget.csv',budgets)

    # Existing independent trajectories are deployment-policy samples, NOT
    # historical training paths or evidence of newly discovered safe routes.
    evaluations = {}
    for method in curves:
        path = args.closure/f'independent-{method}.json'
        evaluations[method] = json.loads(path.read_text())
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    coverage, cells = [], []
    for ci, condition in enumerate(evaluations['iso']['conditions']):
        by_method = {}
        for method in curves:
            other = evaluations[method]['conditions'][ci]
            if other['condition'] != condition['condition']:
                raise ValueError('Native conditions must match without posthoc filtering')
            by_method[method] = {r['seed']:r for r in other['records'] if r['safe_success']}
        common = sorted(set(by_method['iso']) & set(by_method['ea']))
        for method in curves:
            occupied = set()
            controls, length, samples = 0,0.,0
            for seed in common:
                row = by_method[method][seed]
                points, distance = sample_polyline(row['trajectory'])
                occupied.update(tuple(int(v) for v in cell) for cell in np.floor(points/.02).astype(int))
                controls += len(row['trajectory']);length += distance;samples += len(points)
            coverage.append(dict(condition=condition['condition'],method=method,
                independent_safe_success_count=len(by_method[method]),paired_successful_layouts=len(common),
                grid_spacing_m=.02,arclength_spacing_m=.005,occupied_world_cells=len(occupied),
                total_logged_control_points=controls,total_arclength_samples=samples,
                total_logged_polyline_length_m=length,initial_to_first_step_missing_in_legacy_data=True,
                evidence_kind='descriptive_frozen_successful_policy_occupancy_not_training_exploration',
                homotopy_route_count_supported=False))
            cells.extend(dict(condition=condition['condition'],method=method,x=x,y=y,z=z) for x,y,z in sorted(occupied))
    write_csv(args.output/'paired_success_occupancy.csv',coverage)
    write_csv(args.output/'occupied_cells.csv',cells)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,3.5))
    for method,curve in curves.items():
        y=[r['training_safe_successes'] for r in curve]
        axes[0].plot([r['steps']/1000 for r in curve],y,label=method.upper())
        axes[1].plot([r['training_violation_count'] for r in curve],y,label=method.upper())
    axes[0].set_xlabel('New Payload transitions (thousands)')
    axes[1].set_xlabel('Cumulative training violations')
    for axis in axes:
        axis.set_ylabel('Cumulative safe task successes');axis.legend();axis.grid(alpha=.2)
    fig.suptitle('One matched training seed; common Reach actor parent, no new RL')
    fig.tight_layout()
    fig.savefig(args.output/'training_task_cost_bridge.png',dpi=180)
    fig.savefig(args.output/'training_task_cost_bridge.pdf')
    plt.close(fig)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='completed_descriptive_bridge',
        training_seed=41027,matched_new_payload_transitions_each=300000,shared_reach_parent_steps=300000,
        new_training_transitions=0,new_simulation_controls=0,physical_robot_interactions=0,
        source_sha256=sources,fixed_transition_targets=[50000,100000,150000,200000,250000,300000],
        violation_budgets=[100,200,400,600],training_trajectory_coverage_available=False,
        selected_policy_occupancy_is_not_training_coverage=True,route_diversity_claim_supported=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
