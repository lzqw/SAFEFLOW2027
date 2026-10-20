#!/usr/bin/env python3
"""Record the newly completed two-cell CPU move and retire only its boot trigger."""
import json
import subprocess
import time
from pathlib import Path

from walker_cpu_5080_to_4080_recovery_accounting_20261021 import verify_Walker_CPU_5080_to_4080

BASE = Path(__file__).resolve().parent
UNIT = 'safeflow-walker-cpu-5080-to-4080-capacity.service'
ACCOUNT = BASE / 'Walker_CPU_completed_group_accounting_check_20261021_0536.json'
RETIRE = BASE / 'Walker_CPU_completed_coordinator_retirement_20261021.json'
QUEUE = BASE / 'Walker_CPU_completed_queue_snapshot_20261021.json'


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
    assert all(not p.exists() for p in (ACCOUNT, RETIRE, QUEUE)), 'Never replay completed reporting/retirement'
    queue = json.loads((BASE / 'Walker_CPU_5080_to_4080_regular_dispatch_20261021_state.json').read_text())
    assert queue['phase'] == 'all_requested_handoffs_completed' and queue['pending'] == []
    records = verify_Walker_CPU_5080_to_4080(BASE)
    assert {r['job'] for r in records} == {'walker2d-2028', 'walker2d-2029'}
    before = unit_state()
    assert before['MainPID'] == '0' and before['ActiveState'] == 'inactive'
    assert before['Result'] == 'success' and before['ExecMainStatus'] == '0'
    immutable(ACCOUNT, {
        'kind': 'completed_existing_Walker_CPU_group_engineering_accounting_not_final_scientific_audit',
        'observed_at_unix': time.time(), 'verified_completed_existing_CPU_handoffs': len(records),
        'jobs': [{k: r[k] for k in ('job', 'known_additional_physical_steps',
                                  'known_additional_physical_training_cost', 'unknown_unlogged_tail_not_zero')}
                 for r in records],
        'all_original_inputs_and_first_epoch_identities_checked': True,
        'full_final_scientific_audit_executed': False,
        'additional_scientific_evaluation_rollouts': 0, 'new_training_cells': 0,
        'scientific_parameters_changed': False, 'budgets_reduced': False,
        'other_projects_paused': False, 'original_2027_full_parent_gaps_not_repaired': True})
    immutable(QUEUE, queue)
    command = ['systemctl', '--user', 'disable', UNIT]
    subprocess.run(command, check=True, capture_output=True, text=True)
    after = unit_state()
    assert after['UnitFileState'] == 'disabled' and after['MainPID'] == '0'
    immutable(RETIRE, {
        'kind': 'completed_local_one_shot_coordinator_boot_trigger_retirement_not_training_stop',
        'completed_at_unix': time.time(), 'unit': UNIT, 'command': command,
        'queue_snapshot': QUEUE.name, 'accounting_receipt': ACCOUNT.name,
        'before': before, 'after': after, 'no_stop_or_now_flag_used': True,
        'successful_exit_proven_before_disable': True,
        'remote_learner_signals': 0, 'remote_learners_restarted': 0,
        'regular_monitor986_not_changed': True, 'scientific_parameters_or_budgets_changed': False,
        'expected_lifecycle': 'Both requested Walker CPU placements completed. Inactive coordinator is expected. Remote learners and1800-second capacity monitor continue; never restart historical coordinator32455.'})
    print(json.dumps({'verified_new_completed_group': len(records), 'unit': UNIT, 'after': after}))


if __name__ == '__main__':
    main()
