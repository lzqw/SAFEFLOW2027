#!/usr/bin/env python3
"""Attach a previously published local1M reference; no new archive read or rollout."""
import json
import os
import time
from pathlib import Path

from assemble_Point_4M_available7_20261021 import BASE, CACHE, OUT, JOB, read, digest, metrics, protocol


def main():
    receipt_path = BASE / 'Goal2028_local_1M_context_attachment_20261021_receipt.json'
    assert not receipt_path.exists(), 'Inspect original context attachment; never replay'
    before_path = BASE / 'Point_4M_available7_summary_before_local_1M_context_20261021.json'
    before_sha = digest(before_path)
    assert digest(CACHE / 'summary.json') == digest(OUT / 'summary.json') == before_sha
    s = read(OUT / 'summary.json')
    row = next(r for r in s['records'] if r['job'] == JOB)
    assert 'same_seed_native_1M' not in row
    relative = Path('../first_milestone_20_cells_20261020/native_evaluations') / JOB / 'evaluation_step_1000000.json'
    path = (OUT / relative).resolve()
    one = read(path)
    protocol(one, 1000000)
    four = row['native_4M']['metrics']
    row['same_seed_native_1M'] = {'previously_published_relative_path': relative.as_posix(),
                                 'sha256': digest(path), 'metrics': metrics(one)}
    row['differences_4M_minus_1M'] = {
        'mean_return': four['mean_return'] - one['mean_return'],
        'success_percentage_points': round(100 * (four['success_rate'] - one['success_rate']), 10),
        'mean_cost': four['mean_cost'] - one['mean_cost'],
        'cost_cvar_20': four['cost_cvar_20'] - one['cost_cvar_20']}
    data = json.dumps(s, indent=2) + '\n'
    for root in (CACHE, OUT):
        temporary = root / 'summary.local_1M_context.tmp'
        with temporary.open('x') as stream:
            stream.write(data)
        os.replace(temporary, root / 'summary.json')
    receipt = {'kind': 'local_report_baseline_path_context_fix_not_new_evaluation',
        'completed_at_unix': time.time(), 'job': JOB,
        'reason': 'Resolve OUT/../published1M before testing existence; the new output directory did not yet exist on initial assembly.',
        'previous_generated_summary_preserved_at': str(before_path), 'previous_summary_sha256': before_sha,
        'updated_summary_sha256': digest(OUT / 'summary.json'),
        'existing_published_1M_SHA': digest(path),
        'unchanged_4M_SHA': row['native_4M']['sha256'],
        'unchanged_4M_metrics': four, 'added_same_seed_differences': row['differences_4M_minus_1M'],
        'other_six_records_and_4M_CSV_not_modified': True,
        'additional_archive_reads_or_network_calls': 0, 'new_model_loads_or_evaluation_rollouts': 0,
        'full_checkpoint_or_final_scientific_audit_executed': False}
    with receipt_path.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'job': JOB, 'same_seed_1M_metrics': metrics(one),
                      'existing_4M_minus_1M': row['differences_4M_minus_1M'], 'new_rollouts': 0}))


if __name__ == '__main__':
    main()
