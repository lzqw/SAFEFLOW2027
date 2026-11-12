"""Plot actual development checkpoints, never train, evaluate or select a policy.

Each CSV is captured once as bytes, hashed and parsed from that exact snapshot.
Different trials/seeds/configurations remain separate; no smoothing, best-point
selection, missing-value zero filling, synthetic start points or seed averaging.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
COLORS = {'stable_iso': '#2563eb', 's_iso': '#2563eb', 'ea': '#16a34a',
          's_ea': '#16a34a', 'sac_lag': '#d97706', 'sac_lag_native': '#d97706',
          'local': '#7c3aed', 's_local': '#7c3aed'}
CONDITIONS = {'nominal': 'Nominal', 'radius_070': 'Radius 7 cm',
              'payload_wide': 'Wider payload', 'execution_scale_085': 'Action scale 0.85',
              'alternative_route': 'Offset obstacle'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(path, metrics):
    for _ in range(3):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            break
    else:
        raise RuntimeError(f'CSV changed during all bounded reads: {path}')
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    required = {'trial_id', 'method', 'training_seed', 'steps', 'phase', *metrics}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f'Missing plot columns in {path}')
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'Incomplete CSV snapshot: {path}; retry after collection completes')
    for row in rows:
        if row['phase'] not in ('development', 'reference'):
            raise ValueError('Do not relabel formal/independent tests as development')
        row['plot_steps_k'] = float(row['steps'])/1000
        for key in metrics:
            value = None if row[key] == '' else float(row[key])
            if value is not None and not math.isfinite(value):
                raise ValueError(f'Nonfinite measured value {key} in {path}')
            row[key] = value
    receipt = dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                   snapshot_sha256=digest(data), snapshot_bytes=len(data), rows=len(rows),
                   captured_at_utc=datetime.now(timezone.utc).isoformat(),
                   phase_counts=dict(Counter(row['phase'] for row in rows)),
                   trial_ids=sorted({row['trial_id'] for row in rows}),
                   training_seeds=sorted({row['training_seed'] for row in rows}),
                   missing_metrics={key:sum(row[key] is None for row in rows) for key in metrics})
    kept = ['trial_id', 'method', 'training_seed', 'steps', 'phase', 'condition',
            'evaluation_seed', 'episodes', *metrics]
    receipt['plotted_snapshot_records'] = [{key:row[key] for key in kept if key in row}
                                           for row in rows]
    return rows, receipt


def series(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row['trial_id'], row['method'], row['training_seed'], row.get('evaluation_seed', ''))
        grouped[key].append(row)
    seeds = sorted({key[2] for key in grouped})
    styles, count = {}, Counter()
    markers, lines = ['o', '^', 's', 'D', 'v', 'P', 'X'], ['-', '--', ':', '-.']
    for key in sorted(grouped):
        name, method, seed, evaluation = key
        variant = count[(method, seed)]
        count[(method, seed)] += 1
        styles[key] = dict(color=COLORS.get(method, '#64748b'),
            linestyle=lines[(seeds.index(seed)+variant) % len(lines)],
            marker=markers[(seeds.index(seed)+variant) % len(markers)],
            markersize=4, markerfacecolor='none', linewidth=1.35, alpha=.88)
    return grouped, styles


def draw(ax, grouped, styles, metric, *, condition=None, percent=False):
    handles = []
    for key in sorted(grouped):
        points = sorted((row for row in grouped[key]
                         if row[metric] is not None and
                         (condition is None or row.get('condition') == condition)),
                        key=lambda row:row['plot_steps_k'])
        if not points:
            continue
        label = f'{key[0]} | {key[1]} | seed {key[2]} | eval {key[3]}'
        handle, = ax.plot([p['plot_steps_k'] for p in points],
                          [p[metric]*(100 if percent else 1) for p in points],
                          label=label, **styles[key])
        handles.append(handle)
    ax.grid(alpha=.2)
    ax.set_xlabel('Actual training transitions (thousands)')
    ax.set_xlim(left=0)
    if percent:
        ax.set_ylim(-2, 102)
    return handles


def pointgoal_figure(rows, source, output, budget):
    grouped, styles = series(rows)
    legend_rows = math.ceil(max(1, len(grouped))/2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5+.17*legend_rows))
    handles = draw(axes[0], grouped, styles, 'goals_per_episode')
    draw(axes[1], grouped, styles, 'episode_cost')
    axes[0].set(title='Goals per native episode', ylabel='Goals / episode')
    axes[1].set(title='Native episode cost', ylabel='Mean raw episode cost')
    axes[0].set_ylim(bottom=-.025)
    axes[1].set_ylim(bottom=-2)
    axes[1].axhline(budget, color='#dc2626', linestyle='--', linewidth=1)
    handles.append(Line2D([], [], color='#dc2626', linestyle='--', label=f'Common cost budget {budget:g}'))
    fig.suptitle('PointGoal development validation — all measured trials/checkpoints', fontsize=13)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=2,
               fontsize=7, frameon=False)
    fig.text(.5, .014, 'Not independent tests; training-step budgets/configurations differ. No smoothing/selection/jitter; coincident zeros may overlap.\n'
             f'CSV SHA256 {source["snapshot_sha256"][:16]} · single measured endpoints stay single points.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .13+.025*legend_rows, 1, .94))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def payload_figure(rows, source, output):
    present = {row['condition'] for row in rows}
    conditions = [key for key in CONDITIONS if key in present] + sorted(present-set(CONDITIONS))
    if not conditions:
        raise ValueError('No measured Payload conditions')
    grouped, styles = series(rows)
    fig, axes = plt.subplots(2, len(conditions), figsize=(max(8, 3.35*len(conditions)), 8.3), squeeze=False)
    all_handles = {}
    for col, condition in enumerate(conditions):
        for row, metric in enumerate(['safe_success', 'violation']):
            ax = axes[row, col]
            for handle in draw(ax, grouped, styles, metric, condition=condition, percent=True):
                all_handles[handle.get_label()] = handle
            target = (90 if condition == 'nominal' else 80) if row == 0 else (5 if condition == 'nominal' else 10)
            ax.axhline(target, color='#94a3b8', linestyle=':', linewidth=1)
            if row == 0:
                ax.set_title(CONDITIONS.get(condition, condition.replace('_', ' ')), fontsize=10)
            if col == 0:
                ax.set_ylabel('Safe success (%)' if row == 0 else 'Violation (%)')
    fig.suptitle('Payload development validation — each seed and stress condition retained', fontsize=13)
    handles = list(all_handles.values())
    handles.append(Line2D([], [], color='#94a3b8', linestyle=':', label='Effort targets only; not filters'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .05),
               ncol=min(3, max(1, len(handles))), fontsize=7, frameon=False)
    fig.text(.5, .012, 'Not independent tests. No seed/condition averaging or best-checkpoint selection; safe timeouts are not counted as violations.\n'
             f'CSV SHA256 {source["snapshot_sha256"][:16]} · missing values are not filled; segments join observed checkpoints only.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, .94))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointgoal-csv', type=Path, default=ROOT/'reports/paper_crax_payload/CRAX_POINTGOAL_RESULTS.csv')
    parser.add_argument('--payload-csv', type=Path, default=ROOT/'reports/paper_crax_payload/PAYLOAD_RESULTS.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports/paper_crax_payload')
    parser.add_argument('--cost-budget', type=float, default=25.)
    parser.add_argument('--replace', action='store_true', help='Explicitly refresh only these generated artifacts')
    args = parser.parse_args()
    names = ['learning-curves-pointgoal-development.png', 'learning-curves-payload-development.png',
             'learning-curves-development-receipt.json']
    outputs = [args.output_dir/name for name in names]
    if not args.replace and any(path.exists() for path in outputs):
        parser.error('Preserve existing artifacts or explicitly use --replace')
    point, point_source = snapshot(args.pointgoal_csv, ['goals_per_episode', 'episode_cost'])
    payload, payload_source = snapshot(args.payload_csv, ['safe_success', 'violation'])
    if not point or not payload:
        raise ValueError('Actual nonempty result snapshots required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pointgoal_figure(point, point_source, outputs[0], args.cost_budget)
    conditions = payload_figure(payload, payload_source, outputs[1])
    sources = [(args.pointgoal_csv, point_source), (args.payload_csv, payload_source)]
    for path, source in sources:
        source['current_sha256_at_finish'] = digest(path.read_bytes())
        source['changed_after_snapshot'] = source['snapshot_sha256'] != source['current_sha256_at_finish']
    receipt = dict(status='completed_development_snapshot_plots',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(Path(__file__).read_bytes()), matplotlib_version=matplotlib.__version__,
        inputs=[source for _, source in sources],
        inputs_changed_after_capture=any(source['changed_after_snapshot'] for _, source in sources),
        outputs=[dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                      sha256=digest(path.read_bytes())) for path in outputs[:2]],
        payload_conditions=conditions,
        payload_condition_display_labels={key:CONDITIONS.get(key, key) for key in conditions},
        learned_usual_route_obstruction_verified=False,
        pointgoal_cost_budget=args.cost_budget,
        no_methods_seeds_or_weak_checkpoints_filtered=True, smoothing=False, seed_averaging=False,
        independent_test_claim=False, new_training_transitions=0,
        new_simulation_controls=0, new_evaluation_episodes=0, real_interactions=0)
    outputs[2].write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:receipt[key] for key in ['status', 'inputs_changed_after_capture',
                                                'new_training_transitions', 'new_evaluation_episodes']}))


if __name__ == '__main__':
    main()
