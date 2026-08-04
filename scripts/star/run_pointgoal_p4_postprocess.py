#!/usr/bin/env python3
"""Wait for three PointGoal runs and execute the reproducible STAR P4 evidence pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MILESTONES = (0, 50_000, 100_000, 200_000, 300_000)
EXPECTED_SEEDS = (2027, 2028, 2029)
EXPECTED_EVAL_SEEDS = tuple(range(12_027, 12_047))


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'expected a JSON object in {path}')
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def assert_finite(value: Any, location: str = 'payload') -> None:
    """Reject non-finite numerics recursively while allowing JSON metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f'{location}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f'{location}[{index}]')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'non-finite value at {location}: {value!r}')


def process_alive(pid: int) -> bool:
    """Return whether a local process identifier still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid_files(pid_files: list[Path], poll_seconds: float) -> None:
    """Wait until every recorded training process exits."""
    if poll_seconds <= 0:
        raise ValueError('poll seconds must be positive')
    pids = [int(path.read_text(encoding='utf-8').strip()) for path in pid_files]
    if any(pid <= 0 for pid in pids):
        raise ValueError('PID files must contain positive integers')
    while True:
        active = [pid for pid in pids if process_alive(pid)]
        print(
            json.dumps({'waiting_for_training_pids': active}, sort_keys=True),
            flush=True,
        )
        if not active:
            return
        time.sleep(poll_seconds)


def discover_run(parent: Path, receipt_path: Path) -> dict[str, Any]:
    """Validate one completed 300k run and return all downstream paths."""
    summaries = list((parent / 'runs').glob('STAR-*/seed-*/artifacts/training_summary.json'))
    if len(summaries) != 1:
        raise ValueError(f'expected one training summary below {parent}, found {len(summaries)}')
    run_dir = summaries[0].parent.parent
    summary = read_json(summaries[0])
    manifest = read_json(run_dir / 'artifacts/run_manifest.json')
    receipt = read_json(receipt_path)
    for name, payload in (('summary', summary), ('manifest', manifest), ('receipt', receipt)):
        assert_finite(payload, f'{parent.name}.{name}')
    seed = int(summary['seed'])
    if int(manifest['seed']) != seed or int(receipt['seed']) != seed:
        raise ValueError(f'seed mismatch below {parent}')
    if summary.get('status') != 'completed' or receipt.get('status') != 'passed':
        raise ValueError(f'run {seed} did not complete successfully')
    if int(summary['env_steps']) != 300_000 or int(receipt['steps']) != 300_000:
        raise ValueError(f'run {seed} did not reach exactly 300000 environment steps')
    if int(summary['updates']) != 298_000 or int(receipt['updates']) != 298_000:
        raise ValueError(f'run {seed} did not execute exactly 298000 updates')
    if tuple(int(value) for value in manifest['eval_milestones']) != MILESTONES:
        raise ValueError(f'run {seed} does not contain the preregistered milestones')
    checkpoints = [run_dir / f'checkpoints/milestone_{milestone}.pt' for milestone in MILESTONES]
    full_checkpoint = run_dir / 'checkpoints/final_full.pt'
    missing = [
        str(path)
        for path in [*checkpoints, full_checkpoint]
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(f'run {seed} is missing checkpoints: {missing}')
    return {
        'seed': seed,
        'parent': parent,
        'run_dir': run_dir,
        'receipt_path': receipt_path,
        'checkpoints': checkpoints,
        'full_checkpoint': full_checkpoint,
        'summary': summary,
        'receipt': receipt,
    }


def command(
    arguments: list[str],
    source_dir: Path,
    environment: dict[str, str],
    records: list[dict[str, Any]],
) -> None:
    """Run one evidence command and record its exact arguments and wall time."""
    started = time.monotonic()
    subprocess.run(arguments, cwd=source_dir, env=environment, check=True)
    records.append({'arguments': arguments, 'elapsed_seconds': time.monotonic() - started})


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:  # pylint: disable=too-many-locals
    """Validate inputs and run evaluation, audits, visualizations, and strict summary."""
    if len(args.run_parents) != 3 or len(args.receipts) != 3:
        raise ValueError('exactly three run parents and three receipts are required')
    if args.pid_files and len(args.pid_files) != 3:
        raise ValueError('provide either zero or exactly three PID files')
    eval_seeds = tuple(int(value.strip()) for value in args.eval_seeds.split(',') if value.strip())
    if eval_seeds != EXPECTED_EVAL_SEEDS:
        raise ValueError(f'expected fixed evaluation seeds {EXPECTED_EVAL_SEEDS}')
    wait_for_pid_files(args.pid_files, args.poll_seconds)
    runs = [
        discover_run(parent, receipt) for parent, receipt in zip(args.run_parents, args.receipts)
    ]
    runs.sort(key=lambda item: item['seed'])
    if tuple(item['seed'] for item in runs) != EXPECTED_SEEDS:
        raise ValueError(f'expected training seeds {EXPECTED_SEEDS}')

    analysis_commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=args.source_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if args.analysis_source_commit and analysis_commit != args.analysis_source_commit:
        raise ValueError(
            f'analysis source is {analysis_commit}, expected {args.analysis_source_commit}',
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    old_pythonpath = environment.get('PYTHONPATH')
    environment['PYTHONPATH'] = (
        f'{args.source_dir}{os.pathsep}{old_pythonpath}' if old_pythonpath else str(args.source_dir)
    )
    python = str(args.python)
    records: list[dict[str, Any]] = []

    checkpoint_output = args.output_root / 'checkpoint-evaluation'
    checkpoint_work = args.work_root / 'checkpoint-evaluation'
    checkpoint_paths = [str(path) for item in runs for path in item['checkpoints']]
    command(
        [
            python,
            'scripts/star/evaluate_pointgoal_checkpoints.py',
            '--checkpoints',
            *checkpoint_paths,
            '--output-dir',
            str(checkpoint_output),
            '--work-dir',
            str(checkpoint_work),
            '--device',
            args.device,
            '--eval-seeds',
            args.eval_seeds,
        ],
        args.source_dir,
        environment,
        records,
    )

    audit_dirs: list[Path] = []
    for item in runs:
        seed = item['seed']
        audit_dir = args.output_root / f'audit/seed-{seed}'
        audit_dirs.append(audit_dir)
        command(
            [
                python,
                'scripts/star/analyze_pointgoal_replay.py',
                '--checkpoint',
                str(item['full_checkpoint']),
                '--output-dir',
                str(audit_dir),
                '--work-dir',
                str(args.work_root / f'audit/seed-{seed}'),
                '--device',
                args.device,
                '--max-samples',
                '500000',
            ],
            args.source_dir,
            environment,
            records,
        )
        command(
            [
                python,
                'scripts/star/visualize_pointgoal_mechanisms.py',
                '--checkpoint',
                str(item['full_checkpoint']),
                '--output-dir',
                str(args.output_root / f'mechanisms/seed-{seed}'),
                '--work-dir',
                str(args.work_root / f'mechanisms/seed-{seed}'),
                '--trajectory-artifact-dir',
                str(item['run_dir'] / 'artifacts'),
                '--device',
                args.device,
            ],
            args.source_dir,
            environment,
            records,
        )

    fixed_evaluations = checkpoint_output / 'pointgoal_checkpoint_evaluation.csv'
    run_dirs = [str(item['run_dir']) for item in runs]
    command(
        [
            python,
            'scripts/star/plot_star_diagnostics.py',
            '--run-dirs',
            *run_dirs,
            '--checkpoint-evaluation-csv',
            str(fixed_evaluations),
            '--output-dir',
            str(args.output_root / 'figures'),
        ],
        args.source_dir,
        environment,
        records,
    )
    command(
        [
            python,
            'scripts/star/summarize_pointgoal_p4.py',
            '--run-dirs',
            *run_dirs,
            '--audit-dirs',
            *(str(path) for path in audit_dirs),
            '--receipts',
            *(str(item['receipt_path']) for item in runs),
            '--checkpoint-evaluation-csv',
            str(fixed_evaluations),
            '--output-dir',
            str(args.output_root / 'summary'),
        ],
        args.source_dir,
        environment,
        records,
    )
    return {
        'status': 'completed',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'declared_training_source_commit': args.training_source_commit,
        'analysis_source_commit': analysis_commit,
        'device': args.device,
        'eval_seeds': list(eval_seeds),
        'training_seeds': [item['seed'] for item in runs],
        'run_dirs': run_dirs,
        'commands': records,
    }


def parser() -> argparse.ArgumentParser:
    """Build P4 postprocessing arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source-dir', type=Path, required=True)
    result.add_argument('--python', type=Path, default=Path(sys.executable))
    result.add_argument('--run-parents', nargs='+', type=Path, required=True)
    result.add_argument('--receipts', nargs='+', type=Path, required=True)
    result.add_argument('--pid-files', nargs='*', type=Path, default=[])
    result.add_argument('--output-root', type=Path, required=True)
    result.add_argument('--work-root', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument(
        '--eval-seeds',
        default=','.join(str(seed) for seed in EXPECTED_EVAL_SEEDS),
    )
    result.add_argument('--poll-seconds', type=float, default=30.0)
    result.add_argument('--training-source-commit', required=True)
    result.add_argument('--analysis-source-commit')
    return result


def main() -> None:
    """Run P4 postprocessing and always emit a machine-readable pipeline receipt."""
    args = parser().parse_args()
    receipt_path = args.output_root / 'postprocess_receipt.json'
    started = time.monotonic()
    try:
        receipt = run_pipeline(args)
    except Exception as error:  # pylint: disable=broad-exception-caught
        receipt = {
            'status': 'failed',
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - started,
            'error_type': type(error).__name__,
            'error': str(error),
        }
        atomic_json(receipt_path, receipt)
        raise
    receipt['elapsed_seconds'] = time.monotonic() - started
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
