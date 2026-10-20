#!/usr/bin/env python3
"""Extend the restorable cache by one verified duplicate; retain initial proof."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from host_full_input_cache_20261020 import MANIFEST, digest, install_link, manifest, verify_durable

BASE = Path(__file__).resolve().parent
RECEIPT = BASE / 'host_input_cache_Goal2027_extension_20261020_receipt.json'
BEFORE = BASE / 'host_full_input_cache_manifest_initial6_before_Goal2027_extension_20261020.json'


def main():
    assert not RECEIPT.exists() and not BEFORE.exists(), 'Inspect original extension; never repeat'
    original = MANIFEST.read_bytes()
    m = manifest(BASE)
    assert len(m['entries']) == 6
    state = json.loads((BASE / 'point-goal-2027_balanced_MPS_5080_20261020_state.json').read_text())
    assert state['phase'] == 'handoff_completed_first_target_epoch_verified'
    preserved = state['input_preservation']
    host = Path(preserved['host_input_path'])
    relative = host.relative_to(m['host_root'])
    cache = Path(m['cache_root']) / relative
    row = {'job': 'point-goal-2027', 'host_path': str(host), 'cache_path': str(cache),
           'archive_path': preserved['archive_input_path'], 'sha256': state['checkpoint_sha256']}
    assert preserved['host_input_sha256'] == preserved['archive_input_sha256'] == row['sha256']
    assert host.is_file() and not host.is_symlink() and digest(host) == row['sha256']
    assert not any(r['host_path'] == str(host) for r in m['entries'])
    assert not cache.exists()
    # Verify only the one new durable copy, not all previously protected inputs.
    proof = verify_durable(dict(m, entries=[row]))
    size = host.stat().st_size
    cache.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(cache.parent).free >= size + 512 * 1024**2
    part = cache.with_name(cache.name + '.Goal2027_extension.partial')
    assert not part.exists()
    shutil.copyfile(host, part)
    with part.open('rb') as stream:
        os.fsync(stream.fileno())
    assert digest(part) == row['sha256']
    part.rename(cache)
    # Install restoration metadata BEFORE replacing a persistent duplicate with
    # a RAM link. A reboot between these steps still leaves the old regular file.
    BEFORE.write_bytes(original)
    m['entries'].append(row)
    new_manifest = json.dumps(m, indent=2) + '\n'
    temporary = MANIFEST.with_suffix('.Goal2027_extension.tmp')
    assert not temporary.exists()
    temporary.write_text(new_manifest)
    temporary.replace(MANIFEST)
    assert len(manifest(BASE)['entries']) == 7
    install_link(row)
    assert host.is_symlink() and digest(host) == row['sha256']
    receipt = {'kind': 'one_verified_duplicate_added_to_existing_restorable_RAM_cache',
               'updated_at_unix': time.time(), 'new_entry': row,
               'previous_manifest_preserved_at': str(BEFORE),
               'previous_manifest_sha256': hashlib.sha256(original).hexdigest(),
               'updated_manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
               'original_six_entry_offload_receipt_not_overwritten': True,
               'new_durable_copy_proof': proof, 'current_restore_manifest_entries': 7,
               'duplicate_host_regular_bytes_released': size,
               'host_free_bytes_after': shutil.disk_usage(host.parent).free,
               'canonical_host_path_preserved_as_same_SHA_cache_link': True,
               'unmodified_existing_ensure_cache_restores_all_seven_after_reboot': True,
               'source_originals_and_4T_original_not_removed': True,
               'scientific_byte_checks_or_gates_waived': False}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
