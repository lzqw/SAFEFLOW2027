#!/usr/bin/env python3
"""Free only three NEW redundant host inputs; durable4T and originals remain."""
import json
import os
import shutil
import time
from pathlib import Path

from host_full_input_cache_20261020 import MANIFEST, digest, install_link, manifest, verify_durable
from extend_host_input_cache_Ant_Walker_20261021 import atomic_json

BASE = Path(__file__).resolve().parent
NAME = 'host_input_cache_completed_three_extension_20261021'
SPECS = [
    ('walker2d-2030', 'cpu_to_mps_a100_gpu1_20261020', 'b38d2afa46729d59523734cf758fb7ce24997a2f1cc6ab9ac285156a281e2f03'),
    ('ant-2030', 'cpu_to_mps_a100_gpu0_20261020', '94d74c222744cde1eeb95b6746f7f237a28f362301ea3b6781f1e4b2251ff96c'),
    ('walker2d-2029', 'walker_cpu_5080_to_4080_20261021', '56247ec76542793320795b65d0eec8b772d62a3f28f12987a1de3d7114692ac1')]


def main():
    state_path, receipt, before = (BASE / (NAME + suffix) for suffix in ('_state.json', '_receipt.json', '_previous_manifest.json'))
    assert all(not p.exists() for p in (state_path, receipt, before)), 'Never replay a completed extension'
    original = MANIFEST.read_bytes()
    assert digest(MANIFEST) == '8d3a1e919231a346e6f97503b7afa76826b85e3a73c4fcfdd490209429bc04e9'
    m = manifest(BASE)
    assert len(m['entries']) == 11
    rows = []
    total = 0
    for job, side, sha in SPECS:
        s = json.loads((BASE / (job + '_' + side + '_state.json')).read_text())
        assert s['phase'] in ('completed_first_MPS_epoch_source_retired', 'handoff_completed_first_CPU_epoch_verified')
        assert s['source_retired'] is True and s['source_paused'] is False
        p = s['input_preservation']
        assert p['host_input_sha256'] == p['archive_input_sha256'] == s['checkpoint_sha256'] == sha
        assert p['archive_separate_4T_filesystem_verified'] is True
        relative = Path(job) / side / 'input/latest_full.pt'
        host, cache = Path(m['host_root']) / relative, Path(m['cache_root']) / relative
        archive = Path('/home/carla/4T/SAFEFLOW2027_4080/mps_runtime_transitions') / relative
        assert p['host_input_path'] == str(host) and p['archive_input_path'] == str(archive)
        assert host.is_file() and not host.is_symlink() and host.stat().st_nlink == 1
        assert digest(host) == sha and not cache.exists() and not cache.is_symlink()
        assert not any(r['host_path'] == str(host) for r in m['entries'])
        total += host.stat().st_size
        rows.append({'job': job, 'host_path': str(host), 'cache_path': str(cache), 'archive_path': str(archive), 'sha256': sha})
    assert shutil.disk_usage(m['cache_root']).free >= total + 512*1024**2
    free_before = shutil.disk_usage(m['host_root']).free
    state = {'phase': 'verifying_three_new_durable_inputs', 'new_entries': rows, 'started_at_unix': time.time()}
    atomic_json(state_path, state)
    proof = verify_durable(dict(m, entries=rows))
    for row in rows:
        host, cache = Path(row['host_path']), Path(row['cache_path'])
        cache.parent.mkdir(parents=True, exist_ok=True)
        part = cache.with_name(cache.name + '.completed_three.partial')
        with host.open('rb') as source, part.open('xb') as target:
            shutil.copyfileobj(source, target); target.flush(); os.fsync(target.fileno())
        assert digest(part) == row['sha256']
        part.rename(cache)
    assert MANIFEST.read_bytes() == original
    with before.open('xb') as stream:
        stream.write(original); stream.flush(); os.fsync(stream.fileno())
    m['entries'].extend(rows)
    atomic_json(MANIFEST, m)  # Persistent reboot restoration metadata precedes link replacement.
    assert len(manifest(BASE)['entries']) == 14
    for row in rows:
        install_link(row)
        assert Path(row['host_path']).is_symlink() and digest(Path(row['host_path'])) == row['sha256']
    r = {'kind': 'three_NEW_completed_host_inputs_added_to_existing_restorable_RAM_cache',
         'phase': 'completed_same_SHA_links_and_durable_originals_preserved', 'completed_at_unix': time.time(),
         'new_entries': rows, 'new_durable_copy_proof': proof, 'previous_manifest': before.name,
         'previous_manifest_sha256': '8d3a1e919231a346e6f97503b7afa76826b85e3a73c4fcfdd490209429bc04e9',
         'updated_manifest_sha256': digest(MANIFEST), 'restore_manifest_entries': 14,
         'duplicate_host_regular_bytes_released': total, 'host_free_bytes_before': free_before,
         'host_free_bytes_after': shutil.disk_usage(m['host_root']).free,
         'existing_ensure_cache_restores_all_fourteen_after_reboot': True,
         'source_originals_and_4T_originals_removed': 0, 'learner_signals_or_restarts': 0,
         'new_scientific_rollouts_or_audits': 0, 'scientific_parameters_or_budgets_changed': False}
    atomic_json(receipt, r)
    state.update(phase=r['phase'], receipt=receipt.name, completed_at_unix=time.time())
    atomic_json(state_path, state)
    print(json.dumps({k: r[k] for k in ('phase', 'duplicate_host_regular_bytes_released', 'host_free_bytes_before', 'host_free_bytes_after', 'updated_manifest_sha256')}))


if __name__ == '__main__':
    main()
