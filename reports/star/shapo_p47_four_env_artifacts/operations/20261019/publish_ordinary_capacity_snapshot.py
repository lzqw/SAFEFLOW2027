#!/usr/bin/env python3
"""Publish an already-collected ordinary frame, without SSH or scientific tests."""
import argparse
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stamp')
    parser.add_argument('previous')
    args = parser.parse_args()
    assert all(re.fullmatch(r'\d{8}_\d{4}', s) for s in (args.stamp, args.previous))
    raw = read(BASE / ('capacity_progress_observation_' + args.stamp + '.json'))
    previous_name = 'training_status_' + args.previous + '.json'
    current = read(OUT / previous_name)
    old = {w['job']: w for w in current['active_training']}
    workers, placement = [], {}
    for host, info in raw['hosts'].items():
        assert info['status'] == 'observed'
        placement[host] = {'gpu': 0, 'cpu': 0}
        for native in info['training']:
            w = dict(native, host=host, observed_at_unix=info['observed_at_unix'])
            w['total_budget_steps'] = 10000000 if w['job'].startswith('point-') else 5000000
            w['estimated_training_only_days_at_last_epoch_fps'] = (w['total_budget_steps'] - w['steps']) / w['last_epoch_fps'] / 86400
            workers.append(w)
            placement[host]['cpu' if w['device'] == 'cpu' else 'gpu'] += 1
    # This renderer is only for the current18-cell steady runtime. A transition,
    # terminal job, duplicate source/target overlap, or backward counter needs
    # separate evidence-aware treatment, never silently masked here.
    assert len(workers) == len({w['job'] for w in workers}) == 18
    assert set(old) == {w['job'] for w in workers}
    assert all(w['steps'] > old[w['job']]['steps'] for w in workers)
    for key in list(current):
        if key.startswith('ordinary_progress_since_'):
            del current[key]
    current['ordinary_progress_since_' + args.previous] = {'all_18_existing_cells_logged_more_steps': True}
    observed = datetime.fromtimestamp(raw['observed_at_unix'], ZoneInfo('Asia/Shanghai'))
    current.update(observed_at_unix=raw['observed_at_unix'], observed_at_local=observed.isoformat(),
                   previous_status=previous_name, native_observation_file='ordinary_observation_' + args.stamp + '.json',
                   next_ordinary_observation_local=(observed + timedelta(seconds=1800)).isoformat(),
                   active_training=workers, placement=placement,
                   A100_capacity=raw['hosts']['tjucta-a100']['gpus'],
                   storage={h: v['storage'] for h, v in raw['hosts'].items()})
    total = 10000000 + sum(w['steps'] for w in workers)
    current['aggregate_logged_budget_progress'].update(
        accepted_logged_steps_including_two_budget_complete_cells=total, fraction=total / 150000000)
    goal = next(w for w in workers if w['job'] == 'point-goal-2027')
    walker = next(w for w in workers if w['job'] == 'walker2d-2031')
    button = next(w for w in workers if w['job'] == 'point-button-2027')
    current['performance_engineering'].update(
        current_Goal2027_steps=goal['steps'], current_Goal2027_fps=goal['last_epoch_fps'],
        A100_Walker2031_current_logged_steps=walker['steps'],
        A100_Walker2031_current_last_epoch_FPS=walker['last_epoch_fps'],
        whole_5080_Point_last_epoch_FPS_snapshot_sum=sum(w['last_epoch_fps'] for w in workers if w['host'] == '5080' and w['device'] != 'cpu'))
    names = ('Button2027_5080_regular_capacity_dispatch_20261020_state.json',
             'A100_unstarted_transfers_post_local_reboot_20261020_state.json')
    snapshots = []
    for name in names:
        snapshot = 'snapshot_' + args.stamp + '_' + name
        shutil.copyfile(BASE / name, OUT / snapshot)
        snapshots.append(snapshot)
    current['capacity_dispatch_state_snapshots'] = snapshots
    current['data_notice'] = (
        f'{observed.isoformat()} ordinary native frame:all18 active existing cells advanced;'
        '11GPU/7CPU;2/20 budgets complete. 9 verifiedPointMPS clients on5080;'
        f'Button2027 remains on A100 at{button["steps"]}steps,awaiting stable target capacity '
        'and genuine fresh50k full input boundary;no source pause or fresh replacement. '
        'Capacity schedulers use existing30min frames,not additional broad SSH polling. '
        'Source/host/separate4T originals preserved;recovery floors1137276steps/10344cost '
        'plus unknown historical/unlogged tails. Early2027 parent gaps still unresolved;'
        'strict final gates not waived. No scientific parameter,budget,seed,extra evaluation '
        'or foreign-project change. Per-job time extrapolations are training-only,not guarantees.')
    assert current['known_recovery_accounting']['extra_environment_steps_lower_bound'] == 1137276
    assert current['known_recovery_accounting']['extra_training_safety_cost_lower_bound'] == 10344
    name = 'training_status_' + args.stamp + '.json'
    write(OUT / name, current)
    shutil.copyfile(BASE / ('capacity_progress_observation_' + args.stamp + '.json'), OUT / current['native_observation_file'])
    root = read(OUT / 'training_status.json')
    root.update(superseded_by=name, snapshot_notice='Historical17:27 payload below. Latest:' + name + '. ' + current['data_notice'])
    write(OUT / 'training_status.json', root)
    print(json.dumps({'status': name, 'active': len(workers), 'retained_logged_steps': total,
                      'placement': placement, 'next_regular_frame_local': current['next_ordinary_observation_local']}))


if __name__ == '__main__':
    main()
