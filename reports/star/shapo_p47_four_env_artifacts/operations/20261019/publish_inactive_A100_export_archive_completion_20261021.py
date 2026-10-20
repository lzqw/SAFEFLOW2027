#!/usr/bin/env python3
"""Publish the completed storage-only release, without running ML or SSH."""
import json
import shutil
from pathlib import Path

from inactive_A100_export_storage_accounting_20261021 import verify_inactive_A100_export_relocation

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')
NAME = 'A100_inactive_export_archive_20261021_completion_receipt.json'


def main():
    assert not (BASE / NAME).exists() and not (OUT / NAME).exists(), 'Never replay completed storage publication'
    records = verify_inactive_A100_export_relocation(BASE)
    assert len(records) == 1
    record = records[0]
    s = json.loads((BASE / record['release_state_file']).read_text())
    r, a = s['source_release'], s['target_recheck']
    receipt = {
        'kind': 'completed_storage_only_inactive_A100_export_archival',
        'phase': s['phase'], 'finished_at_unix': r['finished_at_unix'],
        'source_manifest': record['manifest_file'], 'source_manifest_sha256': s['manifest_sha256'],
        'copy_state_snapshot': record['copy_state_file'], 'release_state_snapshot': record['release_state_file'],
        'source_root': r['source_root'], 'target_host': '4080', 'target_root': r['target_root'],
        'closed_export_job_directories': 15, 'source_copy_files_released': 89,
        'source_logical_copy_bytes_released': r['source_logical_bytes_released'],
        'filesystem_free_bytes_before': r['filesystem_free_bytes_before'],
        'filesystem_free_bytes_after': r['filesystem_free_bytes_after'],
        'observed_free_delta_also_reflects_shared_filesystem_activity': True,
        'separate_4T_bytes_and_directory_entries_verified_and_fsynced': a['archive_bytes_and_directory_entries_fsynced'],
        'distinct_original_bytes_removed_from_all_locations': 0,
        'canonical_run_files_modified': 0, 'active_export_job_files_modified': 0,
        'learner_signals_or_restarts': 0, 'scientific_parameters_or_budgets_changed': False,
        'new_model_or_scientific_evaluation_runs': 0, 'full_final_scientific_audit_executed': False,
        'existing_2027_missing_parent_identities_not_repaired': True,
        'restore_note': 'Each original export directory retains cold_export_before_final_import_20261021.json. All exact old files are at the recorded4T directory and covered by ARCHIVE_MANIFEST.json. Restore into a separate directory first and verify SHA/mode/mtime. Never overwrite any later fresh export or running job.'}
    with (BASE / NAME).open('x') as stream:json.dump(receipt, stream, indent=2);stream.write('\n')
    for filename in (NAME, record['copy_state_file'], record['release_state_file']):
        assert not (OUT / filename).exists()
        shutil.copyfile(BASE / filename, OUT / filename)
    root = json.loads((OUT / 'training_status.json').read_text())
    root['newer_storage_maintenance_after_snapshot'] = NAME
    root['inactive_A100_export_archive_completed'] = NAME
    (OUT / 'training_status.json').write_text(json.dumps(root, indent=2) + '\n')
    print(json.dumps({'completed_storage_release': NAME,
                      'free_GiB_before': r['filesystem_free_bytes_before'] / 1024**3,
                      'free_GiB_after': r['filesystem_free_bytes_after'] / 1024**3,
                      'released_source_copy_files': 89, 'unique_originals_removed': 0}))


if __name__ == '__main__':
    main()
