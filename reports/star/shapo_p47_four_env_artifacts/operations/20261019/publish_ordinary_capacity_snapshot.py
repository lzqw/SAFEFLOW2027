#!/usr/bin/env python3
"""Publish an already-collected ordinary frame, without SSH or scientific tests."""
import argparse
import hashlib
import json
import math
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


def new_steady_runtime_recoveries():
    records = []
    specs = [('point-button-2027_balanced_MPS_5080_20261020_state.json',
              'handoff_completed_first_target_epoch_verified', 'handoff_Button2027_A100_to_5080_MPS_20261020.py')]
    specs += [(job + '_cpu_5080_to_4080_20261020_state.json',
               'handoff_completed_first_CPU_epoch_verified', 'handoff_Ant_CPU_5080_to_4080_20261020.py')
              for job in ('ant-2028', 'ant-2029')]
    specs += [('ant-2030_cpu_to_mps_a100_gpu0_20261020_state.json',
               'completed_first_MPS_epoch_source_retired', 'handoff_ant2030_A100_GPU0_MPS_20261020.py'),
              ('walker2d-2030_cpu_to_mps_a100_gpu1_20261020_state.json',
               'completed_first_MPS_epoch_source_retired', 'handoff_walkers_A100_GPU1_MPS_20261020.py')]
    for name, phase, controller in specs:
        path = BASE / name
        if not path.exists():
            continue
        s = read(path)
        if s.get('phase') != phase:
            raise ValueError('Use transition-specific evidence, not a steady renderer: ' + name)
        assert s['source_retired'] is True and s['source_paused'] is False
        assert hashlib.sha256((BASE / controller).read_bytes()).hexdigest() == s['controller_sha256']
        steps = s['last_logged_source_steps'] - s['resume_steps']
        last_cost = s.get('last_logged_source_cumulative_training_cost', s.get('last_logged_source_cost'))
        cost = last_cost - s['resume_cumulative_training_cost']
        known_steps = s.get('known_repeated_environment_steps_lower_bound', s.get('known_repeated_steps_lower_bound'))
        known_cost = s.get('known_additional_training_cost_lower_bound', s.get('known_extra_training_cost_lower_bound'))
        assert 0 <= steps <= 5000 and steps == known_steps
        assert math.isfinite(cost) and cost >= 0 and cost == known_cost
        records.append({'job': s['job'], 'state_file': name, 'extra_steps_lower_bound': steps,
                        'extra_safety_cost_lower_bound': cost, 'unlogged_tail_unknown_not_zero': True})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stamp')
    parser.add_argument('previous')
    args = parser.parse_args()
    assert all(re.fullmatch(r'\d{8}_\d{4}', s) for s in (args.stamp, args.previous))
    raw = read(BASE / ('capacity_progress_observation_' + args.stamp + '.json'))
    previous_name = 'training_status_' + args.previous + '.json'
    current = read(OUT / previous_name)
    root_status = read(OUT / 'training_status.json')
    latest_science = root_status.get('newest_intermediate_scientific_result')
    if latest_science:
        summary_path = (OUT / latest_science).resolve()
        summary_path.relative_to((OUT.parent.parent / 'partial_milestones').resolve())
        detail = read(summary_path)
        aggregation = detail.get('descriptive_aggregation', {})
        metrics = aggregation.get('metrics', {})
        current['latest_published_native_science_summary_reference'] = latest_science
        if {'mean_return', 'mean_cost', 'cost_cvar_20', 'failure_rate'} <= set(metrics):
            old_science = current.get('latest_native_science', {})
            if old_science.get('summary') != latest_science:
                current['previous_native_science_summary'] = old_science
            current['latest_native_science'] = {
                'summary': latest_science, 'environment': detail['environment'],
                'milestone_steps': detail['milestone_steps'],
                'final_budget_steps': detail['total_training_budget_steps'],
                'available_native_training_seeds': detail['available_training_seeds'],
                'descriptive_training_seed_unit_n': aggregation['n'],
                'mean_return': metrics['mean_return']['mean_across_five_training_seeds'],
                'SD_return_across_training_seeds': metrics['mean_return']['sd_across_five_training_seeds'],
                'mean_cost': metrics['mean_cost']['mean_across_five_training_seeds'],
                'mean_of_seed_specific_CVaR20': metrics['cost_cvar_20']['mean_across_five_training_seeds'],
                'mean_failure_rate': metrics['failure_rate']['mean_across_five_training_seeds'],
                'not_final_5M_or_strict_provenance_validated_cohort': True,
                'no_matched_C0_or_causal_improvement_claim': True,
                'known_original_full_parent_gap_seeds': detail.get('known_original_full_parent_gap_seeds', [2027]),
                'assembled_from_existing_native_results_only': True,
                'extra_evaluation_rollouts': detail['additional_evaluation_rollouts']}
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
    rtx_points = [w for w in workers if w['host'] == '5080' and w['device'] != 'cpu']
    gpu_count = sum(p['gpu'] for p in placement.values())
    cpu_count = sum(p['cpu'] for p in placement.values())
    current['performance_engineering'].update(
        current_Goal2027_steps=goal['steps'], current_Goal2027_fps=goal['last_epoch_fps'],
        A100_Walker2031_current_logged_steps=walker['steps'],
        A100_Walker2031_current_last_epoch_FPS=walker['last_epoch_fps'],
        whole_5080_Point_last_epoch_FPS_snapshot_sum=sum(w['last_epoch_fps'] for w in rtx_points),
        current_5080_Point_MPS_actor_count=len(rtx_points),
        verified_5080_native_MPS_client_pids=[w['pid'] for w in rtx_points],
        remaining_two_A100_CPU_to_GPU_transfers_not_yet_launched=not any(
            (BASE / n).exists() for n in ('ant-2030_cpu_to_mps_a100_gpu0_20261020_state.json',
                                         'walker2d-2030_cpu_to_mps_a100_gpu1_20261020_state.json')))
    current['performance_engineering'].pop('nine_existing_5080_Point_MPS_actors', None)
    names = ['Button2027_5080_regular_capacity_dispatch_20261020_state.json',
             'A100_unstarted_transfers_post_local_reboot_20261020_state.json']
    if (BASE / 'Ant_CPU_5080_to_4080_regular_dispatch_20261020_state.json').exists():
        names.append('Ant_CPU_5080_to_4080_regular_dispatch_20261020_state.json')
    snapshots = []
    for name in names:
        snapshot = 'snapshot_' + args.stamp + '_' + name
        shutil.copyfile(BASE / name, OUT / snapshot)
        snapshots.append(snapshot)
    current['capacity_dispatch_state_snapshots'] = snapshots
    new_recoveries = new_steady_runtime_recoveries()
    recovery_steps = 1137276 + sum(r['extra_steps_lower_bound'] for r in new_recoveries)
    recovery_cost = 10344 + sum(r['extra_safety_cost_lower_bound'] for r in new_recoveries)
    current['known_recovery_accounting'].update(
        extra_environment_steps_lower_bound=recovery_steps,
        extra_training_safety_cost_lower_bound=recovery_cost,
        subsequent_runtime_placement_recoveries=new_recoveries)
    current['data_notice'] = (
        f'{observed.isoformat()} ordinary native frame:all18 active existing cells advanced;'
        f'{gpu_count}GPU/{cpu_count}CPU;2/20 budgets complete. {len(rtx_points)}PointGPU clients on5080;'
        f'Button2027 currently on{button["host"]} at{button["steps"]}steps. '
        'Completed placements require genuine full inputs and target epoch proof before source retirement;'
        'partial transactions need separate evidence and cannot pass this steady-state renderer. '
        'Capacity schedulers use existing30min frames,not additional broad SSH polling. '
        f'Source/host/separate4T originals preserved;recovery floors{recovery_steps}steps/{recovery_cost:g}cost '
        'plus unknown historical/unlogged tails. Early2027 parent gaps still unresolved;'
        'strict final gates not waived. No scientific parameter,budget,seed,extra evaluation '
        'or foreign-project change. Per-job time extrapolations are training-only,not guarantees.')
    name = 'training_status_' + args.stamp + '.json'
    write(OUT / name, current)
    shutil.copyfile(BASE / ('capacity_progress_observation_' + args.stamp + '.json'), OUT / current['native_observation_file'])
    root = read(OUT / 'training_status.json')
    maintenance = root.get('newer_storage_maintenance_after_snapshot')
    if maintenance:
        path = (OUT / maintenance).resolve()
        path.relative_to(OUT.resolve())
        if read(path).get('finished_at_unix', float('inf')) <= raw['observed_at_unix']:
            root['recent_storage_maintenance'] = root.pop('newer_storage_maintenance_after_snapshot')
    if root.get('newer_scientific_summary_after_snapshot') == current.get('latest_native_science', {}).get('summary'):
        root.pop('newer_scientific_summary_after_snapshot', None)
    root.update(superseded_by=name, snapshot_notice='Historical17:27 payload below. Latest:' + name + '. ' + current['data_notice'])
    write(OUT / 'training_status.json', root)
    print(json.dumps({'status': name, 'active': len(workers), 'retained_logged_steps': total,
                      'placement': placement, 'next_regular_frame_local': current['next_ordinary_observation_local']}))


if __name__ == '__main__':
    main()
