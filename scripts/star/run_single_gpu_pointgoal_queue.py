#!/usr/bin/env python3
"""Run the preregistered PointGoal prior-method queue on one RTX 3090."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')
MATRIX_RELATIVE = Path('configs/star/single_gpu_paper_core.yaml')
EXPECTED_PROTOCOL_VERSION = 'star-single-gpu-paper-core-v1'
EXPECTED_SHAPO_COMMIT = '1bb770437bbbe40a89938661343744243e164bf9'
METHODS = ('SACLag', 'SHAPO-PIDLag')
SEEDS = (2027, 2028, 2029)
PROBE_STEPS = 20_000
FORMAL_STEPS = 200_000
DUAL_RATIO_THRESHOLD = 1.20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def git_output(source: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_sources(source: Path, shapo_source: Path) -> dict[str, str]:
    protocol_path = source / PROTOCOL_RELATIVE
    matrix_path = source / MATRIX_RELATIVE
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('protocol_version') != EXPECTED_PROTOCOL_VERSION:
        raise ValueError('unexpected single-GPU protocol version')
    if protocol.get('status') != 'locked_before_any_new_single_gpu_scientific_training':
        raise ValueError('single-GPU protocol is not locked')
    source_commit = git_output(source, 'rev-parse', 'HEAD')
    if git_output(source, 'status', '--porcelain'):
        raise ValueError('SAFEFLOW2027 source must be clean before launch')
    shapo_commit = git_output(shapo_source, 'rev-parse', 'HEAD')
    if shapo_commit != EXPECTED_SHAPO_COMMIT:
        raise ValueError(f'expected SHAPO {EXPECTED_SHAPO_COMMIT}, observed {shapo_commit}')
    if git_output(shapo_source, 'status', '--porcelain'):
        raise ValueError('official SHAPO source must be clean before launch')
    return {
        'matrix_sha256': sha256_file(matrix_path),
        'protocol_sha256': sha256_file(protocol_path),
        'shapo_commit': shapo_commit,
        'source_commit': source_commit,
    }


def receipt_complete(path: Path, method: str, seed: int, steps: int, protocol_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get('status') == 'complete'
        and payload.get('algorithm') == method
        and payload.get('seed') == seed
        and payload.get('steps') == steps
        and payload.get('protocol_sha256') == protocol_sha
    )


def run_directory(root: Path, phase: str, method: str, seed: int) -> Path:
    return root / phase / method.lower().replace('-', '_') / f'seed-{seed}'


def command_for(
    args: argparse.Namespace,
    identities: dict[str, str],
    phase: str,
    method: str,
    seed: int,
    steps: int,
) -> tuple[list[str], Path, Path]:
    target = run_directory(args.run_root, phase, method, seed)
    result = target / 'receipt.json'
    log = target / 'training.log'
    common = [
        '--result',
        str(result),
        '--phase',
        'throughput' if phase.startswith('throughput') else 'formal',
        '--source-commit',
        identities['source_commit'],
        '--protocol-sha256',
        identities['protocol_sha256'],
    ]
    if method == 'SACLag':
        command = [
            str(args.python_bin),
            str(args.source / 'scripts/star/train_pointgoal_saclag_baseline.py'),
            '--seed',
            str(seed),
            '--steps',
            str(steps),
            '--steps-per-epoch',
            '2000',
            '--replay-size',
            '400000',
            '--device',
            'cuda:0',
            '--log-dir',
            str(target / 'training'),
            *common,
        ]
    elif method == 'SHAPO-PIDLag':
        command = [
            str(args.python_bin),
            str(args.source / 'scripts/star/train_shapo_pidlag_official.py'),
            '--official-source',
            str(args.official_shapo_source),
            *common,
            '--seed',
            str(seed),
            '--task',
            'SafetyPointGoal1-v0',
            '--num-envs',
            '1',
            '--total-steps',
            str(steps),
            '--steps-per-epoch',
            '20000',
            '--device',
            'cuda',
            '--device-id',
            '0',
            '--use-eval',
            'False',
            '--use-shapo-actor',
            'True',
            '--perturbation-target-kl',
            '0.00001',
            '--cost-limit',
            '10.0',
            '--experiment',
            phase,
            '--log-dir',
            str(target / 'training'),
        ]
    else:  # pragma: no cover - guarded by frozen call sites
        raise ValueError(f'unsupported method {method}')
    return command, result, log


def start_process(command: list[str], log_path: Path) -> tuple[subprocess.Popen[str], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open('a', encoding='utf-8')
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        cwd=Path(command[1]).resolve().parents[2],
        env={
            **os.environ,
            'MKL_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'WANDB_MODE': 'offline',
            'WANDB_SILENT': 'true',
        },
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, stream


def run_one(
    args: argparse.Namespace,
    identities: dict[str, str],
    phase: str,
    method: str,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    command, receipt, log = command_for(args, identities, phase, method, seed, steps)
    if receipt_complete(receipt, method, seed, steps, identities['protocol_sha256']):
        return {'method': method, 'receipt': str(receipt), 'state': 'already_complete'}
    started = time.monotonic()
    process, stream = start_process(command, log)
    try:
        return_code = process.wait()
    finally:
        stream.close()
    if return_code:
        raise RuntimeError(f'{method} seed {seed} failed with exit {return_code}; see {log}')
    if not receipt_complete(receipt, method, seed, steps, identities['protocol_sha256']):
        raise RuntimeError(f'{method} seed {seed} exited without a valid receipt')
    return {
        'controller_elapsed_seconds': time.monotonic() - started,
        'method': method,
        'receipt': str(receipt),
        'state': 'completed',
    }


def run_pair(
    args: argparse.Namespace,
    identities: dict[str, str],
    phase: str,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    pending: list[tuple[str, list[str], Path, Path]] = []
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        command, receipt, log = command_for(args, identities, phase, method, seed, steps)
        if receipt_complete(receipt, method, seed, steps, identities['protocol_sha256']):
            rows.append({'method': method, 'receipt': str(receipt), 'state': 'already_complete'})
        else:
            pending.append((method, command, receipt, log))
    started_at = utc_now()
    started = time.monotonic()
    running: list[tuple[str, subprocess.Popen[str], Any, Path, Path]] = []
    for method, command, receipt, log in pending:
        process, stream = start_process(command, log)
        running.append((method, process, stream, receipt, log))
    failures: list[str] = []
    for method, process, stream, receipt, log in running:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append(f'{method} exit={return_code} log={log}')
        elif not receipt_complete(
            receipt,
            method,
            seed,
            steps,
            identities['protocol_sha256'],
        ):
            failures.append(f'{method} missing valid receipt {receipt}')
        else:
            rows.append({'method': method, 'receipt': str(receipt), 'state': 'completed'})
    elapsed = time.monotonic() - started
    timing = {
        'controller_elapsed_seconds': elapsed,
        'finished_at_utc': utc_now(),
        'newly_launched_methods': [method for method, *_ in pending],
        'phase': phase,
        'protocol_sha256': identities['protocol_sha256'],
        'rows': sorted(rows, key=lambda row: row['method']),
        'seed': seed,
        'started_at_utc': started_at,
        'steps_per_method': steps,
    }
    timing_path = args.run_root / phase / f'pair-seed-{seed}.json'
    atomic_json(timing_path, timing)
    if failures:
        raise RuntimeError('; '.join(failures))
    return timing


def concurrency_decision(args: argparse.Namespace, identities: dict[str, str]) -> dict[str, Any]:
    single_receipt = run_directory(
        args.run_root,
        'throughput_single',
        'SACLag',
        2027,
    ) / 'receipt.json'
    pair_timing_path = args.run_root / 'throughput_dual' / 'pair-seed-2027.json'
    single = json.loads(single_receipt.read_text(encoding='utf-8'))
    pair = json.loads(pair_timing_path.read_text(encoding='utf-8'))
    if set(pair['newly_launched_methods']) != set(METHODS):
        raise ValueError('dual throughput decision requires one fresh paired launch')
    single_fps = PROBE_STEPS / float(single['elapsed_seconds'])
    dual_fps = (2 * PROBE_STEPS) / float(pair['controller_elapsed_seconds'])
    ratio = dual_fps / single_fps
    decision = {
        'decision': 'two_learners' if ratio >= DUAL_RATIO_THRESHOLD else 'one_learner',
        'dual_aggregate_fps': dual_fps,
        'dual_over_single_ratio': ratio,
        'engineering_only': True,
        'observed_at_utc': utc_now(),
        'protocol_sha256': identities['protocol_sha256'],
        'scientific_metrics_inspected': False,
        'single_saclag_fps': single_fps,
        'threshold': DUAL_RATIO_THRESHOLD,
    }
    atomic_json(args.run_root / 'throughput_decision.json', decision)
    return decision


def run_formal(args: argparse.Namespace, identities: dict[str, str]) -> None:
    decision_path = args.run_root / 'throughput_decision.json'
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    if decision['protocol_sha256'] != identities['protocol_sha256']:
        raise ValueError('throughput decision belongs to another protocol')
    for seed in SEEDS:
        if decision['decision'] == 'two_learners':
            run_pair(args, identities, 'formal_pointgoal', seed, FORMAL_STEPS)
        else:
            for method in METHODS:
                run_one(args, identities, 'formal_pointgoal', method, seed, FORMAL_STEPS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--official-shapo-source', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--python-bin', type=Path, required=True)
    parser.add_argument(
        '--mode',
        choices=('probe-single', 'probe-dual', 'decide', 'formal', 'all'),
        default='all',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source = args.source.resolve()
    args.official_shapo_source = args.official_shapo_source.resolve()
    args.run_root = args.run_root.resolve()
    args.python_bin = args.python_bin.resolve()
    identities = validate_sources(args.source, args.official_shapo_source)
    args.run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_root / 'queue_identity.json',
        {**identities, 'protocol_version': EXPECTED_PROTOCOL_VERSION},
    )
    if args.mode in ('probe-single', 'all'):
        run_one(args, identities, 'throughput_single', 'SACLag', 2027, PROBE_STEPS)
    if args.mode in ('probe-dual', 'all'):
        run_pair(args, identities, 'throughput_dual', 2027, PROBE_STEPS)
    if args.mode in ('decide', 'all'):
        concurrency_decision(args, identities)
    if args.mode in ('formal', 'all'):
        run_formal(args, identities)
    atomic_json(
        args.run_root / 'queue_state.json',
        {
            'finished_at_utc': utc_now(),
            'mode': args.mode,
            'protocol_sha256': identities['protocol_sha256'],
            'state': 'complete',
        },
    )


if __name__ == '__main__':
    main()
