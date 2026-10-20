#!/usr/bin/env python3
"""Hard-link byte-identical backup copies; never remove distinct checkpoints."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path('/home/carla/SAFEFLOW2027_4080/recovery_backups/from_a100')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--root', default=str(ROOT), choices=(str(ROOT),
        '/home/lzqw/SAFEFLOW2027_A100/recovery_backups/from_4080'))
    args = parser.parse_args()
    root = Path(args.root)
    assert root.resolve() == root and root.is_dir()
    allowed = {f'{slug}-{seed}' for slug in ('point-goal', 'point-button', 'ant', 'walker2d')
               for seed in range(2027, 2032)}
    rows = []
    for directory in sorted(root.iterdir()):
        if directory.name not in allowed or not directory.is_dir() or directory.is_symlink():
            continue
        previous = directory / 'previous_full.pt'
        verified = directory / 'last_verified_full.pt'
        if not previous.is_file() or not verified.is_file() or previous.is_symlink() or verified.is_symlink():
            continue
        first, second = previous.stat(), verified.stat()
        if first.st_ino == second.st_ino or first.st_size != second.st_size or first.st_nlink != 1:
            continue
        first_hash, second_hash = digest(previous), digest(verified)
        if first_hash != second_hash:
            continue
        assert (previous.stat().st_ino, previous.stat().st_mtime_ns) == (first.st_ino, first.st_mtime_ns)
        assert (verified.stat().st_ino, verified.stat().st_mtime_ns) == (second.st_ino, second.st_mtime_ns)
        row = {'job': directory.name, 'bytes': first.st_size, 'sha256': first_hash,
               'previous_inode': first.st_ino, 'verified_inode': second.st_ino,
               'previous_link_count': first.st_nlink,
               'replaced_duplicate_allocated_bytes': first.st_blocks * 512,
               'mode': 'applied' if args.apply else 'inspection_only'}
        if args.apply:
            temporary = directory / '.identical_previous_full.pt'
            assert not temporary.exists()
            os.link(verified, temporary)
            os.replace(temporary, previous)
            assert previous.stat().st_ino == verified.stat().st_ino
        rows.append(row)
        print(json.dumps(row), flush=True)
    payload = {'observed_at_unix': time.time(), 'apply': args.apply, 'root': str(root),
               'only_byte_identical_copies_deduplicated': True,
               'distinct_checkpoints_removed': 0, 'rows': rows}
    if args.apply:
        receipt = root.parent / f'backup_deduplication_{time.time_ns()}.json'
        receipt.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'deduplicated_copies': len(rows),
                      'identical_copy_bytes': sum(row['bytes'] for row in rows)}), flush=True)


if __name__ == '__main__':
    main()
