#!/usr/bin/env python3
"""One engineering group check AFTER all3 moves; no scientific final audit."""
import json
import subprocess
import time
from pathlib import Path

from point_5080_to_4080_MPS_accounting_20261021 import verify_Point_5080_to_4080_MPS

BASE = Path(__file__).resolve().parent
UNIT = 'safeflow-points-idle4080-capacity.service'
ACCOUNT = BASE / 'Point_4080_GPU_completed_group_accounting_20261021.json'
QUEUE = BASE / 'Point_4080_GPU_completed_queue_snapshot_20261021.json'
RETIRE = BASE / 'Point_4080_GPU_completed_coordinator_retirement_20261021.json'


def immutable(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def unit_state():
    result = subprocess.run(['systemctl', '--user', 'show', UNIT,
                             '-p', 'MainPID', '-p', 'Result', '-p', 'ExecMainStatus',
                             '-p', 'ActiveState', '-p', 'SubState', '-p', 'UnitFileState'],
                            check=True, capture_output=True, text=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines())


def main():
    assert all(not p.exists() for p in (ACCOUNT, QUEUE, RETIRE)), 'Never replay a completed group check/retirement'
    queue = json.loads((BASE / 'Point_5080_to_4080_MPS_regular_dispatch_20261021_state.json').read_text())
    assert queue['phase'] == 'all_requested_placements_completed'
    assert queue['maximum_existing_cells_to_move'] == len(queue['completed']) == 3
    expected = {'point-button-2028', 'point-button-2029', 'point-button-2030'}
    assert {x['job'] for x in queue['completed']} == expected
    before = unit_state()
    assert before['MainPID'] == '0' and before['ActiveState'] == 'inactive'
    assert before['Result'] == 'success' and before['ExecMainStatus'] == '0'
    records = verify_Point_5080_to_4080_MPS(BASE)
    assert len(records) == 3 and {r['job'] for r in records} == expected
    additions = [{k: r[k] for k in ('job', 'known_additional_physical_steps',
                                   'known_additional_physical_training_cost', 'unknown_unlogged_tail_not_zero')}
                 for r in records]
    immutable(ACCOUNT, {
        'kind': 'three_completed_existing_Point_MPS_placements_engineering_accounting_not_final_scientific_audit',
        'checked_at_unix': time.time(), 'verified_completed_existing_handoffs': len(records),
        'additions': additions, 'first_epoch_native_MPS_and_full_input_identities_checked': True,
        'completed_recovery_floors_including_prior_history': {
            'environment_steps': 1162276 + sum(r['known_additional_physical_steps'] for r in records),
            'safety_cost': 10386 + sum(r['known_additional_physical_training_cost'] for r in records),
            'historical_and_unlogged_unknown_work_not_zero': True},
        'new_scientific_evaluations_or_training_cells': 0, 'scientific_parameters_or_budgets_changed': False,
        'full_final_20_cell_scientific_audit_executed': False, 'legacy2027_parent_gaps_not_repaired': True})
    immutable(QUEUE, queue)
    command = ['systemctl', '--user', 'disable', UNIT]
    subprocess.run(command, check=True, capture_output=True, text=True)
    after = unit_state()
    assert after['UnitFileState'] == 'disabled' and after['MainPID'] == '0'
    immutable(RETIRE, {
        'kind': 'completed_local_one_shot_dispatcher_boot_trigger_retirement_not_training_stop',
        'completed_at_unix': time.time(), 'unit': UNIT, 'command': command,
        'before': before, 'after': after, 'no_stop_or_now_flag_used': True,
        'successful_exit_proven_before_disable': True, 'queue_snapshot': QUEUE.name,
        'group_accounting_receipt': ACCOUNT.name, 'remote_learner_signals_or_restarts': 0,
        'other_projects_or_scientific_parameters_or_budgets_changed': False,
        'regular1800_second_monitor_unchanged': True,
        'expected_lifecycle': 'All3 existing-cell placements finished. Inactive local dispatcher is expected; independent remote tmux learners continue. Do not restart historical coordinator73574 or its retired predecessor72138.'})
    print(json.dumps({'verified_new_completed_group': len(records), 'retired_only_finished_boot_trigger': UNIT}))


if __name__ == '__main__':
    main()
