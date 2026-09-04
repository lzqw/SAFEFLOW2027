#!/usr/bin/env python3
"""Restore and verify one privately retained SHAPO final-full checkpoint patch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.star.backup_shapo_p47_retained_checkpoints_to_host import (  # noqa: E402
    fsync_directory,
    sha256_file,
    stable_local_identity,
)


def load_receipt(path: Path) -> dict[str, Any]:
    """Load one strict patch receipt mapping."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('final-full patch receipt is not a mapping')
    return payload


def restore_final_checkpoint(
    *,
    base: Path,
    patch: Path,
    receipt_path: Path,
    output: Path,
    zstd_bin: Path,
) -> dict[str, Any]:
    """Decode one patch only after proving base, patch, and destination identities."""
    receipt = load_receipt(receipt_path)
    required = {
        'backup_version': 'shapo-p47-retained-final-patch-v1',
        'status': 'passed',
        'label': 'final_full',
        'storage_representation': 'zstd_patch_from_validated_latest_full',
        'reconstruction_verified': True,
        'scientific_result': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'used_for_preregistered_checkpoint_retention': True,
        'scientific_metric_values_used_for_decision': False,
        'scientific_metric_values_reported': False,
        'checkpoint_counters_validated': True,
        'training_state_changed': False,
        'scientific_protocol_changed': False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f'final-full patch receipt field {key!r} changed')
    for path, label in ((base, 'base'), (patch, 'patch'), (receipt_path, 'receipt')):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'final-full {label} is not a regular file')
    if not zstd_bin.is_file() or not os.access(zstd_bin, os.X_OK):
        raise ValueError('zstd binary is unavailable or not executable')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'refusing to overwrite final checkpoint: {output}')
    base_identity = stable_local_identity(base)
    patch_identity = stable_local_identity(patch)
    if (
        int(receipt.get('base_checkpoint_bytes', -1)) != base_identity[0]
        or receipt.get('base_checkpoint_sha256') != base_identity[1]
    ):
        raise ValueError('final-full patch base identity mismatch')
    if (
        int(receipt.get('patch_bytes', -1)) != patch_identity[0]
        or receipt.get('patch_sha256') != patch_identity[1]
    ):
        raise ValueError('final-full patch payload identity mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp-{os.getpid()}')
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f'refusing to overwrite temporary restore path: {temporary}')
    try:
        completed = subprocess.run(
            [
                str(zstd_bin),
                '-q',
                '-d',
                f'--patch-from={base}',
                str(patch),
                '-o',
                str(temporary),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f'zstd final-full patch restore failed: {completed.stderr.strip()}')
        expected_bytes = int(receipt.get('final_checkpoint_bytes', -1))
        expected_sha256 = str(receipt.get('final_checkpoint_sha256', ''))
        if temporary.stat().st_size != expected_bytes or sha256_file(temporary) != expected_sha256:
            raise ValueError('restored final-full checkpoint identity mismatch')
        if (
            stable_local_identity(base) != base_identity
            or stable_local_identity(patch) != patch_identity
        ):
            raise ValueError('final-full patch inputs changed during reconstruction')
        os.chmod(temporary, 0o600)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        fsync_directory(output.parent)
    except Exception:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return {
        'status': 'passed',
        'job': receipt['job'],
        'environment': receipt['environment'],
        'training_seed': receipt['training_seed'],
        'checkpoint_env_steps': receipt['checkpoint_env_steps'],
        'output': str(output),
        'bytes': output.stat().st_size,
        'sha256': sha256_file(output),
        'scientific_result': False,
        'used_for_checkpoint_or_configuration_selection': False,
    }


def main() -> None:
    """Restore one exact final-full checkpoint to a caller-selected new path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--patch', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zstd-bin', type=Path, required=True)
    args = parser.parse_args()
    result = restore_final_checkpoint(
        base=args.base.resolve(),
        patch=args.patch.resolve(),
        receipt_path=args.receipt.resolve(),
        output=args.output.resolve(),
        zstd_bin=args.zstd_bin.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
