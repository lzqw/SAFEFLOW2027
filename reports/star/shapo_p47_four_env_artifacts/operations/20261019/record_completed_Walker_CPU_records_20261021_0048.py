#!/usr/bin/env python3
"""Record the one completed Walker placement while the second cell stays put."""
import json
import time
from pathlib import Path
from walker_cpu_5080_to_4080_recovery_accounting_20261021 import verify_Walker_CPU_5080_to_4080

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')
NAME = 'Walker_CPU_completed_records_accounting_check_20261021_0048.json'


def main():
    assert not (BASE / NAME).exists() and not (OUT / NAME).exists()
    records = verify_Walker_CPU_5080_to_4080(BASE)
    assert {r['job'] for r in records} == {'walker2d-2028'}
    assert not (BASE / 'walker2d-2029_walker_cpu_5080_to_4080_20261021_state.json').exists()
    r = records[0]
    s = r['receipt']
    receipt = {'kind': 'completed_existing_Walker_CPU_records_engineering_accounting_not_final_scientific_audit',
               'observed_at_unix': time.time(), 'verified_completed_existing_CPU_handoffs': 1,
               'job': r['job'], 'target_identity': s['target_identity'],
               'input_checkpoint_sha256': s['checkpoint_sha256'], 'resume_steps': s['resume_steps'],
               'first_target_steps': s['first_verified_target_steps'], 'first_target_FPS': s['first_verified_target_fps'],
               'source_retired_after_first_epoch_originals_preserved': True,
               'known_additional_physical_steps_this_placement': r['known_additional_physical_steps'],
               'known_additional_physical_safety_cost_this_placement': r['known_additional_physical_training_cost'],
               'unknown_unlogged_source_work_not_zero': True,
               'second_Walker_placement_not_started': True,
               'second_Walker_training_continues_at_original_location': True,
               'additional_scientific_evaluation_rollouts': 0, 'new_training_cells': 0,
               'full_final_scientific_audit_executed': False, 'budgets_reduced': False,
               'other_projects_paused': False, 'scientific_parameters_changed': False,
               'original_2027_full_parent_gaps_not_repaired': True}
    encoded = json.dumps(receipt, indent=2) + '\n'
    (BASE / NAME).write_text(encoded)
    (OUT / NAME).write_text(encoded)
    print(json.dumps({'receipt': NAME, 'verified_completed_existing_CPU_handoffs': 1,
                      'additional_steps_this_placement': r['known_additional_physical_steps'],
                      'additional_safety_cost_this_placement': r['known_additional_physical_training_cost']}))


if __name__ == '__main__':
    main()
