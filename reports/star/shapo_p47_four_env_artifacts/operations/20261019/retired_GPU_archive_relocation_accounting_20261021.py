"""Disclose exact source-to4T archival relocation, never substitute checkpoint identities."""
import hashlib
import json
from pathlib import Path

NAME = 'A100_retired_GPU_archive_20261021_source_release_state.json'
MANIFEST = 'A100_retired_GPU_archive_20261021_v2_source_manifest.json'
SOURCE_ROOT = '/home/lzqw/SAFEFLOW2027_A100/recovery_backups/retired_gpu_workers'
TARGET_ROOT = '/home/carla/4T/SAFEFLOW2027_4080/retired_originals_from_a100/20261021/retired_gpu_workers'


def verify_retired_GPU_archive_relocation(base):
    base = Path(base)
    if not (base / NAME).exists():
        return []
    s = json.loads((base / NAME).read_text())
    if s.get('phase') != 'source_copies_released_after_4T_verification':
        raise ValueError('Retired-original archival source release needs inspection')
    assert s['controller_sha256'] == hashlib.sha256((base / 'release_verified_A100_retired_GPU_sources_20261021.py').read_bytes()).hexdigest()
    payload = (base / MANIFEST).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == s['source_manifest_sha256']
    m = json.loads(payload)
    a, r = s['target_recheck'], s['source_release']
    assert a['phase'] == 'archive_reverified_immediately_before_source_release'
    assert a['source_manifest_sha256'] == r['source_manifest_sha256'] == s['source_manifest_sha256']
    assert a['target_root'] == r['target_root'] == TARGET_ROOT and r['source_root'] == m['source_root'] == SOURCE_ROOT
    assert a['separate_4T_filesystem_verified'] is True and r['original_bytes_remain_on_verified_separate_4T'] is True
    assert r['phase'] == 'source_copies_released_to_verified_4T_archive'
    assert s['source_files_removed'] == r['source_files_removed'] == a['verified_regular_files'] == m['file_count'] == len(m['files']) == 598
    assert r['source_logical_bytes_released'] == a['verified_data_bytes'] == m['total_logical_bytes'] == 8013297651
    assert r['distinct_original_bytes_removed_from_all_locations'] == 0
    assert r['training_processes_signalled'] == 0 and r['other_projects_modified'] is False
    assert len(r['pointer_directories']) == len(m['scope_directories']) == 13
    rows = {v['relative_path']: v for v in m['files']}
    for job in ('point-goal-2027', 'point-button-2027'):
        old = json.loads((base / (job + '_balanced_MPS_5080_20261020_state.json')).read_text())
        relative = str(Path(old['retired_source_full_checkpoint']).relative_to(SOURCE_ROOT))
        assert rows[relative]['sha256'] == old['checkpoint_sha256']
    return [{'format': 'retired-originals-verified-storage-relocation-v1',
             'source_root': SOURCE_ROOT, 'target_host': '4080', 'target_root': TARGET_ROOT,
             'verified_original_regular_files': 598, 'verified_original_logical_bytes': 8013297651,
             'source_copies_removed_after_verified_archive': 598,
             'source_directories_retain_location_and_restore_pointers': 13,
             'source_manifest_sha256': s['source_manifest_sha256'],
             'manifest_file': MANIFEST, 'release_state_file': NAME, 'copy_state_file': s['copy_state_file'],
             'original_checkpoint_byte_identities_not_changed_or_aliased': True,
             'early2027_missing_full_parents_not_repaired': True,
             'additional_training_steps_or_cost_not_claimed': True}]
