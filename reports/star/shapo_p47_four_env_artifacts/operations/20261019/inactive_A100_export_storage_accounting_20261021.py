"""Disclose inactive backup-export archival without altering scientific identities."""
import hashlib
import json
from pathlib import Path


def verify_inactive_A100_export_relocation(base):
    base = Path(base)
    name = 'A100_inactive_export_archive_20261021_release_state.json'
    if not (base / name).exists():
        return []  # A copy-only operation has not removed any source data.
    s = json.loads((base / name).read_text())
    if s['phase'] != 'closed_export_copies_released_after_verified_4T':
        raise ValueError('Inactive-export source release requires inspection')
    assert s['controller_sha256'] == hashlib.sha256((base / 'release_inactive_A100_export_copies_20261021.py').read_bytes()).hexdigest()
    blob = (base / s['manifest_file']).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == s['manifest_sha256']
    m = json.loads(blob);a = s['target_recheck'];r = s['source_release']
    assert m['source_root'] == r['source_root'] == '/home/lzqw/SAFEFLOW2027_A100/recovery_backups/export'
    assert a['manifest_sha256'] == r['manifest_sha256'] == s['manifest_sha256']
    assert a['target_root'] == r['target_root'] == '/home/carla/4T/SAFEFLOW2027_4080/retired_export_snapshots_from_a100/20261021/inactive_export_snapshots'
    assert a['separate_4T_filesystem_verified'] is True and r['original_bytes_remain_on_verified_separate_4T'] is True
    assert a['archive_bytes_and_directory_entries_fsynced'] is True
    assert a['verified_regular_files'] == s['source_files_removed'] == r['source_files_removed'] == m['file_count'] == len(m['files']) == 89
    assert len(m['scope_jobs']) == len(r['pointer_directories']) == 15
    assert a['verified_logical_bytes'] == r['source_logical_bytes_released'] == m['total_logical_bytes']
    assert r['canonical_run_files_modified'] == r['active_export_job_files_modified'] == r['learner_signals_or_restarts'] == 0
    assert r['distinct_original_bytes_removed_from_all_locations'] == 0
    assert r['other_projects_modified'] is False
    return [{'format': 'verified-inactive-export-storage-relocation-v1', 'storage_kind': 'inactive_A100_export_snapshots',
             'source_root': m['source_root'], 'target_host': '4080', 'target_root': r['target_root'],
             'verified_original_regular_files': 89, 'verified_original_logical_bytes': m['total_logical_bytes'],
             'closed_export_job_directories': 15, 'source_copies_removed_after_verified_archive': 89,
             'manifest_file': s['manifest_file'], 'release_state_file': name, 'copy_state_file': s['copy_state_file'],
             'original_checkpoint_byte_identities_not_changed_or_aliased': True,
             'early2027_missing_full_parents_not_repaired': True, 'additional_training_steps_or_cost_not_claimed': True}]
