#!/usr/bin/env python3
"""One engineering accounting check of two completed existing Ant CPU moves."""
import json
import time
from pathlib import Path
from cpu_5080_to_4080_recovery_accounting_20261020 import verify_Ant_CPU_5080_to_4080

BASE = Path(__file__).resolve().parent
NAME = 'Ant_CPU_completed_group_accounting_check_20261020_2323.json'
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')


def main():
    assert not (BASE / NAME).exists() and not (OUT / NAME).exists()
    records = verify_Ant_CPU_5080_to_4080(BASE)
    assert {r['job'] for r in records} == {'ant-2028', 'ant-2029'}
    rows = []
    for r in records:
        s = r['receipt']
        rows.append({'job': r['job'], 'resume_steps': s['resume_steps'],
                     'first_target_steps': s['first_verified_target_steps'],
                     'first_target_FPS': s['first_verified_target_fps'],
                     'target_identity': s['target_identity'],
                     'input_checkpoint_sha256': s['checkpoint_sha256'],
                     'known_additional_physical_steps': r['known_additional_physical_steps'],
                     'known_additional_physical_training_cost': r['known_additional_physical_training_cost'],
                     'source_retired_after_first_epoch_and_original_evidence_preserved': True,
                     'unknown_unlogged_source_tail_not_zero': True})
    receipt = {'kind': 'completed_two_cell_engineering_accounting_not_final_scientific_audit',
               'observed_at_unix': time.time(), 'verified_completed_existing_CPU_handoffs': len(records),
               'records': rows, 'additional_environment_steps_lower_bound_this_group': sum(r['known_additional_physical_steps'] for r in records),
               'additional_safety_cost_lower_bound_this_group': sum(r['known_additional_physical_training_cost'] for r in records),
               'physical_recovery_unknown_tails_not_assumed_zero': True,
               'additional_scientific_evaluation_rollouts': 0,
               'full_final_scientific_audit_executed': False,
               'scientific_parameters_changed': False, 'budgets_reduced': False,
               'other_projects_paused': False, 'new_training_cells': 0,
               'original_2027_full_parent_gaps_not_repaired': True}
    encoded = json.dumps(receipt, indent=2) + '\n'
    (BASE / NAME).write_text(encoded)
    (OUT / NAME).write_text(encoded)
    print(json.dumps({'receipt': NAME, 'verified_completed_existing_CPU_handoffs': len(records),
                      'additional_steps_lower_bound_this_group': receipt['additional_environment_steps_lower_bound_this_group']}))


if __name__ == '__main__':
    main()
