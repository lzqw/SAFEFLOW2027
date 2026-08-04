#!/usr/bin/env python3
"""Re-evaluate STAR PointGoal checkpoints with fixed seeds and tie-correct statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

import omnisafe
from omnisafe.utils.star_validation import checkpoint_reconstruction_config


SUMMARY_FIELDS = (
    'episodes',
    'mean_return',
    'mean_cost',
    'cost_cvar_20',
    'violation_ratio',
    'calibration_error',
    'spearman',
    'predicted_risk_mean',
    'empirical_risk_mean',
)


def json_safe(value: Any) -> Any:
    """Replace undefined floating statistics with JSON null recursively."""
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON atomically."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one consistently headed CSV atomically."""
    if not rows:
        return
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def checkpoint_payload(path: Path) -> dict[str, Any]:
    """Load and validate one self-describing STAR checkpoint."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'star-checkpoint' or payload.get('version') != 1:
        raise ValueError(f'{path} is not a STAR checkpoint')
    extra = payload.get('extra', {})
    if extra.get('environment') != 'SafetyPointGoal1-v0':
        raise ValueError(f'{path} is not a SafetyPointGoal1-v0 checkpoint')
    return payload


def build_algorithm(  # pylint: disable=too-many-locals
    payload: dict[str, Any],
    device: str,
    eval_seeds: list[int],
    work_dir: Path,
) -> Any:
    """Reconstruct the exact actor, critic, adapter, and normalizer without resuming training."""
    step = int(payload['counters']['env_steps'])
    extra = payload['extra']
    custom = deepcopy(checkpoint_reconstruction_config(payload))
    custom['seed'] = int(extra['seed'])
    train_cfgs = custom.setdefault('train_cfgs', {})
    train_cfgs.update(
        {
            'device': device,
            'torch_threads': 1,
            'vector_env_nums': 1,
            'parallel': 1,
            'total_steps': max(1, step),
            'eval_episodes': len(eval_seeds),
        },
    )
    algo_cfgs = custom.setdefault('algo_cfgs', {})
    replay_size = max(2, int(algo_cfgs.get('size', 500_000)))
    algo_cfgs.update(
        {
            'steps_per_epoch': max(1, step),
            'size': replay_size,
            'batch_size': max(1, min(int(algo_cfgs.get('batch_size', 256)), replay_size - 1)),
            'eval_seeds': eval_seeds,
            'eval_milestones': [],
            'resume_from': '',
            'milestone_checkpoints': False,
            'checkpoint_replay': False,
        },
    )
    custom['logger_cfgs'] = {
        'log_dir': str(work_dir),
        'use_tensorboard': False,
        'use_wandb': False,
    }
    wrapper = omnisafe.Agent('STAR', 'SafetyPointGoal1-v0', custom_cfgs=custom)
    algorithm: Any = wrapper.agent
    algorithm._actor_critic.load_state_dict(payload['model'])  # pylint: disable=protected-access
    algorithm._obs_normalizer.load_state_dict(  # pylint: disable=protected-access
        payload['normalizer'],
    )
    return algorithm


def public_row(
    payload: dict[str, Any],
    checkpoint: Path,
    evaluation: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Return one compact public CSV row from a raw evaluation result."""
    row = {
        'training_seed': int(payload['extra']['seed']),
        'milestone': int(payload['counters']['env_steps']),
        'checkpoint': str(checkpoint),
        'evaluation_device': device,
        'spearman_method': 'average-rank tie-correct',
    }
    row.update(
        {
            field: (
                None
                if isinstance(evaluation[field], float) and not math.isfinite(evaluation[field])
                else evaluation[field]
            )
            for field in SUMMARY_FIELDS
        },
    )
    return row


def evaluate_checkpoint(  # pylint: disable=too-many-locals
    checkpoint: Path,
    output_dir: Path,
    work_dir: Path,
    device: str,
    eval_seeds: list[int],
) -> dict[str, Any]:
    """Evaluate one checkpoint and write its raw result and first-seed trajectory."""
    payload = checkpoint_payload(checkpoint)
    step = int(payload['counters']['env_steps'])
    seed = int(payload['extra']['seed'])
    algorithm = build_algorithm(
        payload,
        device,
        eval_seeds,
        work_dir / f'seed-{seed}-step-{step}',
    )
    try:
        evaluation = algorithm._evaluate(step)  # pylint: disable=protected-access
    finally:
        algorithm._logger.close()  # pylint: disable=protected-access
        algorithm._env.close()  # pylint: disable=protected-access

    trajectory = evaluation.pop('trajectory')
    result = {
        'training_seed': seed,
        'environment': 'SafetyPointGoal1-v0',
        'checkpoint': str(checkpoint),
        'evaluation_device': device,
        'spearman_method': 'average-rank tie-correct',
        **evaluation,
    }
    atomic_json(output_dir / f'evaluation_seed_{seed}_step_{step}.json', result)
    if trajectory:
        atomic_csv(output_dir / f'trajectory_seed_{seed}_step_{step}.csv', trajectory)
    return public_row(payload, checkpoint, evaluation, device)


def parser() -> argparse.ArgumentParser:
    """Build checkpoint-evaluation arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--eval-seeds', default='12027,12028,12029,12030,12031')
    return result


def main() -> None:
    """Evaluate all requested checkpoints and emit a stable combined CSV."""
    args = parser().parse_args()
    eval_seeds = [int(value) for value in args.eval_seeds.split(',') if value.strip()]
    if not eval_seeds:
        raise ValueError('at least one evaluation seed is required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        evaluate_checkpoint(
            checkpoint,
            args.output_dir,
            args.work_dir,
            args.device,
            eval_seeds,
        )
        for checkpoint in args.checkpoints
    ]
    rows.sort(key=lambda row: (row['milestone'], row['training_seed']))
    atomic_csv(args.output_dir / 'pointgoal_checkpoint_evaluation.csv', rows)
    print(
        json.dumps(
            {
                'status': 'passed',
                'checkpoints': len(rows),
                'evaluation_seeds': eval_seeds,
                'output_dir': str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == '__main__':
    main()
