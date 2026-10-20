#!/usr/bin/env python3
"""Keep the pre-transition native frame separate from a later paused-source state."""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')
PREFIX = 'walker2d-2030_cpu_to_mps_a100_gpu1_20261020'


def read(path):
    return json.loads(path.read_text())


def immutable(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    s = read(BASE / (PREFIX + '_state.json'))
    assert s['phase'] in ('source_paused_full_input_preserved', 'MPS_target_starting_source_paused')
    assert s['source_paused'] is True and s['source_retired'] is False
    raw = read(BASE / 'capacity_progress_observation_20261021_0321.json')
    old = read(OUT / 'training_status_20261021_0251.json')
    workers = [dict(w, host=h) for h, v in raw['hosts'].items() for w in v.get('training', [])]
    previous = {w['job']: w for w in old['active_training']}
    assert len(workers) == len({w['job'] for w in workers}) == 18
    assert all(v['status'] == 'observed' for v in raw['hosts'].values())
    assert set(previous) == {w['job'] for w in workers}
    assert all(w['steps'] > previous[w['job']]['steps'] for w in workers)
    assert raw['observed_at_unix'] < s['updated_at_unix']
    native_name = 'ordinary_observation_20261021_0321.json'
    state_name = PREFIX + '_inflight_state_20261021_0322.json'
    queue_name = 'snapshot_20261021_0322_A100_unstarted_transfers_post_local_reboot_20261020_state.json'
    immutable(OUT / native_name, raw)
    immutable(OUT / state_name, s)
    immutable(OUT / queue_name, read(BASE / 'A100_unstarted_transfers_post_local_reboot_20261020_state.json'))
    total = 10000000 + sum(w['steps'] for w in workers)
    gain = sum(w['steps'] - previous[w['job']]['steps'] for w in workers)
    observed = datetime.fromtimestamp(raw['observed_at_unix'], ZoneInfo('Asia/Shanghai'))
    report = {
        'kind': 'pre_handoff_native_progress_with_separate_later_Walker2030_MPS_transaction',
        'engineering_observed_at_unix': time.time(),
        'last_steady_training_status': 'training_status_20261021_0251.json',
        'native_frame_before_this_transaction': native_name,
        'native_observed_at_local': observed.isoformat(),
        'native_all_18_cells_advanced': True,
        'native_logged_step_gain_since_0251': gain,
        'native_accepted_logged_steps_including_two_budget_complete_cells': total,
        'native_training_budget_fraction': total / 150000000,
        'completed_training_budgets': 2,
        'native_worker_rows_are_not_a_current_post_pause_worker_census': True,
        'job': s['job'], 'inflight_phase': s['phase'],
        'inflight_state_snapshot': state_name, 'dispatcher_snapshot': queue_name,
        'source_identity': s['source_identity'],
        'source_held_paused_until_verified_first_target_epoch': True,
        'target_launch_requested': s['target_launch_requested'],
        'first_target_epoch_and_source_retirement_not_yet_proven': True,
        'genuine_resume_steps': s['resume_steps'],
        'genuine_input_sha256': s['checkpoint_sha256'],
        'genuine_input_has_full_replay_rng_optimizers_and_finite_model': s['full_replay_rng_optimizers_finite_model'],
        'physical_target': 'A100 GPU1 existing private MPS server',
        'capacity_before_pause': s['capacity_before_pause'],
        'conditional_extra_work_if_completed': {
            'steps': s['known_repeated_steps_lower_bound'],
            'safety_cost': s['known_extra_training_cost_lower_bound'],
            'not_added_to_completed_recovery_totals_yet': True,
            'unknown_unlogged_source_tail_not_zero': True},
        'last_completed_recovery_accounting': old['known_recovery_accounting'],
        'no_new_experiment_cells': True, 'no_extra_scientific_evaluations': True,
        'full_training_budget_steps': 5000000,
        'scientific_parameters_changed': False, 'foreign_projects_modified': False,
        'no_speed_gain_guaranteed_before_actual_target_epochs': True,
        'full_final_scientific_audit_executed': False,
        'final_scientific_report': False,
        'legacy_provenance': old['legacy_provenance'],
        'final_scientific_requirements_unchanged': old['final_scientific_requirements_unchanged'],
        'next_regular_resource_check_local': (observed + timedelta(seconds=1800)).isoformat(),
        'data_notice': 'The03:21 native cohort frame preceded the Walker2030 CPU-to-GPU1 pause. All18 advanced in that frame; do not infer all18 are currently running while this later handoff is partial. Source remains available in memory; completion requires a real new5k target epoch, verified MPS GPU identity and source retirement. Resume is the genuine4.35M full checkpoint, not a candidate policy choice. All5M budgets, source62ca739 and strict final gates remain unchanged. Early2027 original parent gaps remain unresolved.'}
    name = 'training_status_20261021_0322_Walker2030_A100_inflight.json'
    immutable(OUT / name, report)
    root = read(OUT / 'training_status.json')
    root.update(superseded_by=name, latest_inflight_engineering_transition=state_name,
                snapshot_notice='Historical17:27 payload below. Latest engineering report:' + name + '. ' + report['data_notice'])
    temporary = OUT / 'training_status.Walker2030_inflight.tmp'
    with temporary.open('x') as stream:
        stream.write(json.dumps(root, indent=2) + '\n')
    temporary.replace(OUT / 'training_status.json')
    print(json.dumps({'status': name, 'phase': s['phase'], 'native_logged_steps': total,
                      'native_step_gain': gain, 'first_target_epoch_proven': False}))


if __name__ == '__main__':
    main()
