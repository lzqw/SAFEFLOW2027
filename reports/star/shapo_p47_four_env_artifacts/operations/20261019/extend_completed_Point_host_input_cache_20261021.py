#!/usr/bin/env python3
"""Cache ONE newly completed Point placement input; keep durable originals."""
import argparse
import fcntl
import json
import os
import re
import shutil
import time
from pathlib import Path

from host_full_input_cache_20261020 import MANIFEST, digest, install_link, manifest, verify_durable
from extend_host_input_cache_Ant_Walker_20261021 import atomic_json

BASE = Path(__file__).resolve().parent
SIDE = 'point_5080_to_4080_MPS_20261021'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('stamp')
    parser.add_argument('--cache-completed-host-duplicate', action='store_true', required=True)
    args = parser.parse_args()
    assert args.job in {slug + '-' + str(seed) for slug in ('point-goal', 'point-button') for seed in range(2027, 2032)}
    assert re.fullmatch(r'2026102[1-9]_\d{4}', args.stamp)
    lock = (BASE / 'host_full_input_cache_Point_extension_20261021.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prefix = 'host_input_cache_' + args.job + '_' + args.stamp
    state_path, receipt, previous = (BASE / (prefix + suffix) for suffix in ('_state.json', '_receipt.json', '_previous_manifest.json'))
    assert all(not p.exists() for p in (state_path, receipt, previous)), 'Inspect original extension; never replay'
    original = MANIFEST.read_bytes()
    old_sha = digest(MANIFEST)
    m = manifest(BASE)
    assert 14 <= len(m['entries']) <= 16
    s = json.loads((BASE / (args.job + '_' + SIDE + '_state.json')).read_text())
    assert s['phase'] == 'completed_first_MPS_epoch_source_retired'
    assert s['source_retired'] is True and s['source_paused'] is False
    assert s['retired_source_full_checkpoint_SHA_verified'] is True
    assert s['controller_sha256'] == digest(BASE / 'handoff_Point_5080_to_4080_MPS_20261021.py')
    p = s['input_preservation'];sha = s['checkpoint_sha256']
    assert p['host_input_sha256'] == p['archive_input_sha256'] == sha
    assert p['archive_separate_4T_filesystem_verified'] is True
    relative = Path(args.job) / SIDE / 'input/latest_full.pt'
    host, cache = Path(m['host_root']) / relative, Path(m['cache_root']) / relative
    archive = Path('/home/carla/4T/SAFEFLOW2027_4080/mps_runtime_transitions') / relative
    assert p['host_input_path'] == str(host) and p['archive_input_path'] == str(archive)
    assert host.is_file() and not host.is_symlink() and host.stat().st_nlink == 1 and digest(host) == sha
    assert not cache.exists() and not cache.is_symlink()
    assert not any(r['host_path'] == str(host) for r in m['entries'])
    size = host.stat().st_size
    assert shutil.disk_usage(m['cache_root']).free >= size + 512*1024**2
    free_before = shutil.disk_usage(m['host_root']).free
    row = {'job': args.job, 'host_path': str(host), 'cache_path': str(cache), 'archive_path': str(archive), 'sha256': sha}
    state = {'phase': 'verifying_new_completed_Point_durable_input', 'new_entry': row,
             'created_at_unix': time.time(), 'previous_manifest_sha256': old_sha}
    atomic_json(state_path, state)
    try:
        proof = verify_durable(dict(m, entries=[row]))
        cache.parent.mkdir(parents=True, exist_ok=True)
        part = cache.with_name(cache.name + '.completed_Point.partial')
        with host.open('rb') as source, part.open('xb') as target:
            shutil.copyfileobj(source, target);target.flush();os.fsync(target.fileno())
        assert digest(part) == sha
        part.rename(cache)
        assert MANIFEST.read_bytes() == original
        with previous.open('xb') as stream:
            stream.write(original);stream.flush();os.fsync(stream.fileno())
        m['entries'].append(row)
        atomic_json(MANIFEST, m)  # Durable restoration metadata is committed FIRST.
        install_link(row)
        assert host.is_symlink() and digest(host) == sha
        r = {'kind': 'one_NEW_completed_Point_host_duplicate_added_to_restorable_RAM_cache',
             'phase': 'completed_same_SHA_link_and_durable_originals_preserved',
             'job': args.job, 'completed_at_unix': time.time(), 'new_entry': row,
             'new_durable_copy_proof': proof, 'previous_manifest': previous.name,
             'previous_manifest_sha256': old_sha, 'updated_manifest_sha256': digest(MANIFEST),
             'restore_manifest_entries': len(m['entries']), 'duplicate_host_regular_bytes_released': size,
             'host_free_bytes_before': free_before, 'host_free_bytes_after': shutil.disk_usage(m['host_root']).free,
             'unchanged_ensure_cache_restores_all_manifest_entries_after_reboot': True,
             'source_originals_and_4T_originals_removed': 0, 'learner_signals_or_restarts': 0,
             'scientific_parameters_or_budgets_changed': False, 'new_scientific_rollouts_or_audits': 0}
        atomic_json(receipt, r)
        state.update(phase=r['phase'], receipt=receipt.name, completed_at_unix=time.time())
        atomic_json(state_path, state)
        print(json.dumps({k: r[k] for k in ('job', 'phase', 'restore_manifest_entries', 'duplicate_host_regular_bytes_released', 'host_free_bytes_after', 'updated_manifest_sha256')}))
    except Exception as error:
        state.update(phase='needs_inspection_no_automatic_replay', error=str(error))
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
