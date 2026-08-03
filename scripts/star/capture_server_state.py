#!/usr/bin/env python3
"""Capture the reproducible STAR server manifest and Python environment lock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


OMNISAFE_BASELINE = '15603dd7a654a991d0a4648216b69d60b81a6366'
FLAC_COMMIT = '61a25eef42935fc395b1e04071e2779311fcb705'
SHAPO_COMMIT = '1bb770437bbbe40a89938661343744243e164bf9'


def command(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode, result.stdout.strip()


def read_cgroup(name: str) -> str | None:
    path = Path('/sys/fs/cgroup') / name
    return path.read_text(encoding='utf-8').strip() if path.exists() else None


def parse_cpu_quota(cpu_max: str | None) -> float | None:
    if cpu_max is None:
        return None
    quota, period = cpu_max.split()
    return None if quota == 'max' else int(quota) / int(period)


def parse_byte_limit(value: str | None) -> int | None:
    return None if value in (None, 'max') else int(value)


def disk_record(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    code, mount = command(['findmnt', '-J', '-T', str(path), '-o', 'TARGET,SOURCE,FSTYPE'])
    mount_data: dict[str, Any] | None = None
    if code == 0:
        parsed = json.loads(mount)
        filesystems = parsed.get('filesystems', [])
        mount_data = filesystems[0] if filesystems else None
    return {
        'path': str(path.resolve()),
        'mount': mount_data,
        'total_bytes': usage.total,
        'used_bytes': usage.used,
        'free_bytes': usage.free,
        'free_gib': round(usage.free / 2**30, 3),
    }


def git_record(path: Path, expected: str | None = None) -> dict[str, Any]:
    sha_code, sha = command(['git', 'rev-parse', 'HEAD'], cwd=path)
    branch_code, branch = command(['git', 'branch', '--show-current'], cwd=path)
    status_code, status = command(['git', 'status', '--porcelain'], cwd=path)
    return {
        'path': str(path.resolve()),
        'commit': sha if sha_code == 0 else None,
        'expected_commit': expected,
        'matches_expected': expected is None or (sha_code == 0 and sha == expected),
        'branch': branch if branch_code == 0 and branch else None,
        'detached': branch_code == 0 and not branch,
        'clean': status_code == 0 and not status,
        'status': status.splitlines() if status else [],
    }


def package_versions() -> dict[str, str | None]:
    names = (
        'torch',
        'numpy',
        'scipy',
        'pandas',
        'gymnasium',
        'gymnasium-robotics',
        'safety-gymnasium',
        'mujoco',
        'pygame',
        'omnisafe',
        'pytorch-lightning',
        'cvxpy',
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_records() -> list[dict[str, Any]]:
    fields = (
        'index,name,uuid,driver_version,memory.total,compute_cap,pstate,'
        'temperature.gpu,power.limit'
    )
    code, output = command(
        ['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
    )
    if code != 0:
        return [{'error': output}]
    keys = fields.split(',')
    return [
        {key: value.strip() for key, value in zip(keys, line.split(','), strict=True)}
        for line in output.splitlines()
        if line.strip()
    ]


def port_is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--safe-root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--manifest-output', type=Path, required=True)
    parser.add_argument('--lock-output', type=Path, required=True)
    args = parser.parse_args()

    safe_root = args.safe_root.resolve()
    repo = args.repo.resolve()
    constraints = repo / 'scripts/star/constraints-py311.txt'
    cpu_max = read_cgroup('cpu.max')
    memory_max = read_cgroup('memory.max')
    pip_check_code, pip_check = command([sys.executable, '-m', 'pip', 'check'])

    try:
        import torch

        torch_record: dict[str, Any] = {
            'version': torch.__version__,
            'cuda_build': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'cudnn_version': torch.backends.cudnn.version(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError as error:
        torch_record = {'error': str(error)}

    manifest = {
        'schema_version': 1,
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'kernel': platform.release(),
        'architecture': platform.machine(),
        'python': {
            'version': platform.python_version(),
            'executable': sys.executable,
        },
        'cpu': {
            'logical_visible': psutil.cpu_count(logical=True),
            'physical_visible': psutil.cpu_count(logical=False),
            'affinity': sorted(os.sched_getaffinity(0)),
            'cgroup_cpu_max': cpu_max,
            'cgroup_quota_cpus': parse_cpu_quota(cpu_max),
            'cgroup_cpuset_effective': read_cgroup('cpuset.cpus.effective'),
        },
        'memory': {
            'cgroup_memory_max': memory_max,
            'cgroup_memory_max_bytes': parse_byte_limit(memory_max),
            'system_total_bytes': psutil.virtual_memory().total,
        },
        'gpu': gpu_records(),
        'torch': torch_record,
        'disks': {
            'safe_data_disk': disk_record(safe_root),
            'system_root': disk_record(Path('/')),
        },
        'repositories': {
            'SAFEFLOW2027': git_record(repo),
            'omnisafe_required_baseline': OMNISAFE_BASELINE,
            'FLAC': git_record(safe_root / 'code/FLAC', FLAC_COMMIT),
            'SHAPO': git_record(safe_root / 'code/SHAPO', SHAPO_COMMIT),
        },
        'packages': package_versions(),
        'constraints': {
            'path': str(constraints),
            'sha256': file_sha256(constraints),
        },
        'pip_check': {'returncode': pip_check_code, 'output': pip_check},
        'ports': {
            '6006_tensorboard_listening': port_is_listening(6006),
            '6008_read_only_monitor_listening': port_is_listening(6008),
        },
        'runtime_layout': {
            name: str(safe_root / relative)
            for name, relative in {
                'code': 'code',
                'environment': 'envs/star-p0-p1',
                'pip_cache': 'cache/pip',
                'xdg_cache': 'cache/xdg',
                'huggingface_cache': 'cache/huggingface',
                'torch_cache': 'cache/torch',
                'torchinductor_cache': 'cache/torchinductor',
                'temporary_files': 'cache/tmp',
                'active_runs': 'runs/active',
                'completed_runs': 'runs/completed',
                'artifacts': 'artifacts',
            }.items()
        },
        'billing': {
            'currency': 'CNY',
            'hourly_rate': 1.48,
            'cost_per_mstep_formula': '1.48 * 1e6 / (3600 * aggregate_env_fps)',
        },
    }

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        f'{json.dumps(manifest, indent=2, sort_keys=True)}\n',
        encoding='utf-8',
    )

    freeze_code, freeze = command([sys.executable, '-m', 'pip', 'freeze', '--all'])
    if freeze_code != 0:
        raise RuntimeError(f'pip freeze failed: {freeze}')
    lock_header = [
        '# STAR P0/P1 server environment lock',
        f'# captured_at_utc={manifest["captured_at_utc"]}',
        f'# python={platform.python_version()}',
        f'# executable={sys.executable}',
        f'# omnisafe_required_baseline={OMNISAFE_BASELINE}',
        f'# flac_commit={FLAC_COMMIT}',
        f'# shapo_commit={SHAPO_COMMIT}',
        f'# constraints_sha256={manifest["constraints"]["sha256"]}',
    ]
    args.lock_output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_output.write_text(
        '\n'.join([*lock_header, freeze, '']),
        encoding='utf-8',
    )
    print(json.dumps({'manifest': str(args.manifest_output), 'lock': str(args.lock_output)}))


if __name__ == '__main__':
    main()
