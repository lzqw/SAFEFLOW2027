"""Plot already mirrored EA-STAB evidence; no training, rollout or selection.

Keep individual trials/segments visible. Missing evaluations are absent, not
zeros; elapsed times include evaluation/checkpoint overhead within the native
runner, not just simulator time. Shared hosts confound cross-run speed ratios.
"""
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPORT = Path(__file__).resolve().parents[3] / 'reports/star/arm_ea_stability'
COLORS = {'s_iso': 'C0', 's_local': 'C2', 's_ea': 'C1'}
NAMES = {'s_iso': 'S-Iso', 's_local': 'S-Local', 's_ea': 'S-EA'}
CONDITIONS = [('nominal', 'Nominal'), ('radius_070', 'Radius 7cm'),
              ('radius_080', 'Radius 8cm'), ('lateral_030', 'Lateral +/-3cm'),
              ('lateral_060', 'Lateral +/-6cm')]


def load_trials():
    trials = []
    for path in sorted((REPORT / 'native_runs').glob('*/run_config.json')):
        config = json.loads(path.read_text())
        if config['stage'] != 'ea_stability_v2_development':
            continue
        with (path.parent / 'progress.csv').open(newline='') as stream:
            rows = [{key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(stream)]
        if not rows:
            continue
        receipt_path = path.parent / 'receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        label = f"{NAMES[config['variant']]} {config['training_seed']}"
        if config['initial_steps']:
            label += f" from {config['initial_steps']//1000}k"
        incident_path=path.parent/'engineering_interruption.json'
        trials.append(dict(name=path.parent.name, config=config, rows=rows,
            incident=json.loads(incident_path.read_text()) if incident_path.exists() else {},
            receipt=receipt, label=label, color=COLORS[config['variant']],
            style='-' if config['training_seed'] == 2027 else '--',
            evaluations=sorted([json.loads(p.read_text()) for p in path.parent.glob('evaluation_*.json')],
                               key=lambda value: value['milestone_steps'])))
    return trials


def finish(fig, axes, trials, title, destination):
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.set_xlabel('Train transitions (k)')
    handles = [plt.Line2D([], [], color=t['color'], linestyle=t['style'], label=t['label']) for t in trials]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle(title + '\nPer trial; no smoothing, extrapolation or best-checkpoint selection', fontsize=11)
    fig.tight_layout(rect=(0, .11, 1, .93))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def progress_figure(trials, panels, title, destination):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), squeeze=False)
    for axis, (key, heading, multiplier) in zip(axes.flat, panels):
        axis.set_title(heading, fontsize=10)
        for trial in trials:
            # Exclude the non-learning/random prefix from numeric learner plots
            # and throughput; keep it in cumulative safety/episode curves.
            rows = trial['rows']
            if key not in ('recent_success', 'training_violation_count', 'recent_episode_length',
                           'recent_violation_rate', 'recent_return'):
                rows = [row for row in rows if row['actor_updates'] > 0 and row['steps'] >= 10000]
            axis.plot([r['steps']/1000 for r in rows], [r[key]*multiplier for r in rows],
                      color=trial['color'], linestyle=trial['style'])
            if key == 'q_replay_mean':
                axis.plot([r['steps']/1000 for r in rows], [r['q_policy_mean'] for r in rows],
                          color=trial['color'], linestyle=trial['style'], alpha=.4)
        if key in ('recent_success', 'recent_violation_rate', 'clipping_rate'):
            axis.set_ylim(-2, 102)
    finish(fig, axes, trials, title, destination)


def throughput(trial):
    config, rows, receipt = trial['config'], trial['rows'], trial['receipt']
    last = rows[-1]
    first = next((r for r in rows if r['steps'] >= max(config['initial_steps']+10000, last['steps']-20000)), None)
    complete = bool(receipt and receipt.get('status') == 'completed')
    # The final progress row is emitted BEFORE the milestone evaluation. Use
    # the receipt for completed wall time, or label a running row as partial.
    elapsed = receipt['segment_elapsed_seconds'] if complete else last['segment_elapsed_seconds']
    steps = receipt['steps'] if complete else int(last['steps'])
    window_seconds = last['segment_elapsed_seconds']-first['segment_elapsed_seconds'] if first else 0
    incident=trial['incident']
    lifetime=(datetime.fromisoformat(incident['native_terminal_observed_utc'])-
              datetime.fromisoformat(incident['native_started_utc'])).total_seconds() if (
                  incident.get('native_terminal_observed_utc') and incident.get('native_started_utc')) else None
    return dict(trial=trial['name'], variant=config['variant'], training_seed=config['training_seed'],
        host=config['host'], gpu=config['gpu'], initial_steps=config['initial_steps'], steps=steps,
        new_transitions=steps-config['initial_steps'], segment_elapsed_seconds=elapsed,
        end_to_end_transitions_per_second=(steps-config['initial_steps'])/elapsed,
        recent_window_start_steps=int(first['steps']) if window_seconds > 0 else '',
        recent_window_end_steps=int(last['steps']) if window_seconds > 0 else '',
        recent_window_transitions_per_second=(last['steps']-first['steps'])/window_seconds if window_seconds > 0 else '',
        observed_failed_process_lifetime_seconds=lifetime if lifetime is not None else '',
        logged_transitions_per_failed_process_lifetime=(steps-config['initial_steps'])/lifetime if lifetime else '',
        scope='completed_including_final_evaluation_and_save' if complete else
            'interrupted_logged_prefix;process_lifetime_separately_includes_postlog_CUDA_wait' if incident else
            'running_logged_prefix_not_final_milestone_overhead',
        source_commit=config['source_commit'], evidence=f"native_runs/{trial['name']}/progress.csv")


def main():
    trials = load_trials()
    if not trials:
        raise SystemExit('No mirrored native development progress; nothing to plot')
    figures = REPORT / 'figures'
    figures.mkdir(exist_ok=True)
    progress_figure(trials, [
        ('recent_success', 'Training recent safe success (%)', 100),
        ('training_violation_count', 'Native lineage cumulative training violations', 1),
        ('recent_episode_length', 'Recent completed episode length (steps)', 1),
        ('segment_train_fps', 'Native segment end-to-end train transitions/s', 1),
        ('recent_violation_rate', 'Training recent violation rate (%)', 100),
        ('recent_return', 'Recent training episode return', 1)],
        'EA-STAB v2: training behavior and actual elapsed throughput (not held-out performance)',
        figures / 'development-training.png')
    progress_figure(trials, [
        ('q_replay_mean', 'Signed Q mean: replay solid opacity / policy faint', 1),
        ('reward_td', 'Reward TD diagnostic', 1),
        ('alpha', 'Energy coefficient alpha', 1),
        ('energy', 'Path energy', 1),
        ('safety_multiplier', 'Safety multiplier', 1),
        ('clipping_rate', 'Action clipping (%)', 100)],
        'EA-STAB v2: shared learner stability diagnostics', figures / 'development-learner.png')
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), squeeze=False)
    for column, (condition, title) in enumerate(CONDITIONS):
        for row, metric in enumerate(('safe_success', 'violation')):
            axis = axes[row, column]
            axis.set_title(title + (' safe success' if row == 0 else ' violation'), fontsize=10)
            axis.set_ylim(-3, 103)
            axis.set_ylabel('%')
            for trial in trials:
                points = [(e['milestone_steps']/1000, next(c[metric] for c in e['conditions']
                           if c['condition']['name'] == condition)*100) for e in trial['evaluations']]
                if points:
                    axis.plot(*zip(*points), marker='o', color=trial['color'], linestyle=trial['style'])
    finish(fig, axes, trials, 'EA-STAB v2: fixed development evaluations; nominal n=30, stress n=20 per checkpoint\n'
           'Fixed 1M uses n=100/condition and independent layouts; absent checkpoints are not zero',
           figures / 'development-evaluation.png')
    speeds = [throughput(trial) for trial in trials]
    with (REPORT / 'TRAINING_THROUGHPUT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(speeds[0]))
        writer.writeheader()
        writer.writerows(speeds)
    manifest = dict(kind='mirrored_native_development_curves_not_final_decision',
        trials=[dict(trial=t['name'], source_commit=t['config']['source_commit'],
                     latest_logged_steps=int(t['rows'][-1]['steps']),
                     evaluation_milestones=[e['milestone_steps'] for e in t['evaluations']]) for t in trials],
        evaluation_sampling='individual training seeds; episodes are not independent training seeds',
        throughput='Measured native wall time includes learner, physics, completed evaluations and saves within segment; running prefixes exclude final milestone overhead. No causal cross-host speed ratio.',
        missing='No imputation or incomplete-run removal; future revisions remain separately labeled',
        training_transitions_added=0, physical_evaluation_queries_added=0)
    (figures / 'development-curves.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
