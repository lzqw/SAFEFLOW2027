#!/usr/bin/env python3
"""Release only three proven redundant host copies; keep durable4T recovery."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from host_full_input_cache_20261020 import MANIFEST, digest, install_link, manifest, verify_durable

BASE = Path(__file__).resolve().parent
PREFIX = 'host_input_cache_Ant_Walker_extension_20261021'
STATE = BASE / (PREFIX + '_state.json')
RECEIPT = BASE / (PREFIX + '_receipt.json')
BEFORE = BASE / 'host_full_input_cache_manifest_eight_before_Ant_Walker_extension_20261021.json'
OLD_MANIFEST_SHA = '6c49f8985483a05cac6ea3d5473f739a5c4d43fbddc55b9aead8be2ad1b55e2b'
HELPER_SHA = 'b5bcb294c6d0fb8fd412584d3fbfa2006bb78071040cb75c7830ff6d180949da'
SPECS = (
    ('ant-2028', 'cpu_5080_to_4080_20261020', 146463011,
     'ece3b5647da88e05990f06acddf968bed7bc7b01607acefaabde4666c5889f71'),
    ('ant-2029', 'cpu_5080_to_4080_20261020', 146464803,
     '5e20506d499075c5128b3f165dcb28f433caa2b06ad7f5fccc8efeb1d6eea3cd'),
    ('walker2d-2028', 'walker_cpu_5080_to_4080_20261021', 110902627,
     '6047fcba32353f3c0b036ad008292a9321477222b7d36474c6a817310f7d5b43'),
)


def atomic_json(path, value):
    temporary = path.with_suffix('.extension.tmp')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value, indent=2) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-three-verified-host-duplicates', action='store_true', required=True)
    parser.parse_args()
    lock = (BASE / (PREFIX + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not any(p.exists() for p in (STATE, RECEIPT, BEFORE)), 'Inspect original attempt; never replay'
    original = MANIFEST.read_bytes()
    assert hashlib.sha256(original).hexdigest() == OLD_MANIFEST_SHA
    assert digest(BASE / 'host_full_input_cache_20261020.py') == HELPER_SHA
    m = manifest(BASE)
    assert len(m['entries']) == 8
    rows = []
    for job, side, size, sha in SPECS:
        s = json.loads((BASE / (job + '_' + side + '_state.json')).read_text())
        assert s['phase'] == 'handoff_completed_first_CPU_epoch_verified'
        assert s['source_retired'] is True and s['source_paused'] is False
        preserved = s['input_preservation']
        assert preserved['host_input_sha256'] == preserved['archive_input_sha256'] == s['checkpoint_sha256'] == sha
        assert preserved['archive_separate_4T_filesystem_verified'] is True
        relative = Path(job) / side / 'input/latest_full.pt'
        host = Path(m['host_root']) / relative
        cache = Path(m['cache_root']) / relative
        archive = Path('/home/carla/4T/SAFEFLOW2027_4080/mps_runtime_transitions') / relative
        assert preserved['host_input_path'] == str(host) and preserved['archive_input_path'] == str(archive)
        assert host.resolve() == host and cache.resolve() == cache
        assert host.is_file() and not host.is_symlink() and host.stat().st_size == size
        assert host.stat().st_nlink == 1 and digest(host) == sha
        assert not cache.exists() and not cache.is_symlink()
        assert not any(r['host_path'] == str(host) for r in m['entries'])
        rows.append({'job': job, 'host_path': str(host), 'cache_path': str(cache),
                     'archive_path': str(archive), 'sha256': sha})
    total = sum(spec[2] for spec in SPECS)
    assert shutil.disk_usage(m['cache_root']).free >= total + 512 * 1024**2
    free_before = shutil.disk_usage(m['host_root']).free
    state = {'phase': 'verifying_only_three_new_durable_copies', 'new_entries': rows,
             'old_manifest_SHA': OLD_MANIFEST_SHA, 'created_at_unix': time.time(),
             'learner_signals_or_new_rollouts': 0, 'source_or_4T_files_removed': 0}
    atomic_json(STATE, state)
    try:
        proof = verify_durable(dict(m, entries=rows))
        for row in rows:
            host, cache = Path(row['host_path']), Path(row['cache_path'])
            cache.parent.mkdir(parents=True, exist_ok=True)
            part = cache.with_name(cache.name + '.Ant_Walker_extension.partial')
            with host.open('rb') as source, part.open('xb') as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            assert digest(part) == row['sha256']
            part.rename(cache)
        # Commit persistent restore metadata before replacing any regular file.
        assert MANIFEST.read_bytes() == original
        with BEFORE.open('xb') as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        m['entries'].extend(rows)
        atomic_json(MANIFEST, m)
        assert len(manifest(BASE)['entries']) == 11
        state.update(phase='eleven_entry_restore_manifest_committed_before_links', durable_proof=proof)
        atomic_json(STATE, state)
        for row in rows:
            install_link(row)
            assert Path(row['host_path']).is_symlink()
            assert os.readlink(row['host_path']) == row['cache_path']
        receipt = {
            'kind': 'three_verified_redundant_host_inputs_added_to_restorable_RAM_cache',
            'phase': 'completed_same_SHA_links_and_durable_originals_preserved',
            'completed_at_unix': time.time(), 'new_entries': rows,
            'previous_manifest_preserved_at': str(BEFORE),
            'previous_manifest_sha256': OLD_MANIFEST_SHA,
            'updated_manifest_sha256': digest(MANIFEST),
            'unchanged_restore_helper_sha256': HELPER_SHA,
            'new_durable_copy_proof': proof, 'restore_manifest_entries': 11,
            'duplicate_host_regular_bytes_released': total,
            'host_free_bytes_before': free_before,
            'host_free_bytes_after': shutil.disk_usage(m['host_root']).free,
            'canonical_paths_preserved_as_same_SHA_links': True,
            'existing_ensure_cache_restores_all_eleven_after_reboot': True,
            'initial_and_prior_extension_receipts_not_overwritten': True,
            'source_originals_and_4T_originals_removed': 0,
            'new_scientific_rollouts_or_audits': 0,
            'training_budget_or_parameters_changed': False,
            'learner_processes_signaled_or_restarted': 0,
            'scientific_byte_checks_or_gates_waived': False}
        atomic_json(RECEIPT, receipt)
        state.update(phase=receipt['phase'], receipt=str(RECEIPT), completed_at_unix=time.time())
        atomic_json(STATE, state)
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        state.update(phase='needs_inspection_no_automatic_replay', error=str(error))
        atomic_json(STATE, state)
        raise


if __name__ == '__main__':
    main()
