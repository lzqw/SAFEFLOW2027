#!/usr/bin/env python3
"""Audit PointGoal replay risk calibration and conservative-boundary coverage."""

# This standalone checkpoint reader intentionally shares reconstruction logic with the
# mechanism renderer so either artifact command can be reproduced independently.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

import omnisafe
from omnisafe.common.buffer.star_replay_buffer import BOUNDARY


plt.switch_backend('Agg')


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an audit receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write one compact evidence table atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Read reconstruction fields and verify that replay is present."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'star-checkpoint' or payload.get('version') != 1:
        raise ValueError(f'{path} is not a STAR checkpoint')
    if payload.get('replay') is None:
        raise ValueError('replay calibration requires a full checkpoint with replay')
    return {
        'env_steps': int(payload['counters']['env_steps']),
        'environment': str(payload['extra']['environment']),
        'seed': int(payload['extra']['seed']),
        'replay_capacity': int(payload['replay']['capacity']),
        'replay_size': int(payload['replay']['size']),
    }


def build_algorithm(
    checkpoint: Path,
    metadata: dict[str, Any],
    device: str,
    work_dir: Path,
) -> Any:
    """Restore the trained model, normalizer, and compact resident replay."""
    batch_size = max(1, min(256, metadata['replay_capacity'] - 1))
    custom = {
        'seed': metadata['seed'],
        'train_cfgs': {
            'device': device,
            'torch_threads': 1,
            'vector_env_nums': 1,
            'parallel': 1,
            'total_steps': metadata['env_steps'],
            'eval_episodes': 0,
        },
        'algo_cfgs': {
            'steps_per_epoch': max(1, metadata['env_steps']),
            'size': metadata['replay_capacity'],
            'batch_size': batch_size,
            'eval_milestones': [],
            'resume_from': str(checkpoint),
            'milestone_checkpoints': False,
            'checkpoint_replay': False,
        },
        'logger_cfgs': {
            'log_dir': str(work_dir),
            'use_tensorboard': False,
            'use_wandb': False,
        },
    }
    wrapper = omnisafe.Agent('STAR', metadata['environment'], custom_cfgs=custom)
    return wrapper.agent


def deterministic_indices(size: int, maximum: int) -> torch.Tensor:
    """Select the full replay or a deterministic time-stratified subset."""
    count = min(size, maximum)
    if count <= 0:
        raise ValueError('resident replay and --max-samples must be non-empty')
    if count == size:
        return torch.arange(size, dtype=torch.long)
    indices = torch.linspace(0, size - 1, count, dtype=torch.float64).round().long()
    if indices.unique().numel() != count:
        raise RuntimeError('deterministic replay subsampling produced duplicate indices')
    return indices


@torch.no_grad()
def infer_replay(  # pylint: disable=too-many-locals
    algorithm: Any,
    indices: torch.Tensor,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    """Infer critic risk and expected cost on an unbiased resident replay subset."""
    buffer = algorithm._buf  # pylint: disable=protected-access
    device = algorithm._device  # pylint: disable=protected-access
    algo_cfgs = algorithm._cfgs.algo_cfgs  # pylint: disable=protected-access
    outputs: dict[str, list[torch.Tensor]] = {'predicted_risk': [], 'risk_ucb': [], 'cost': []}
    for start in range(0, indices.numel(), chunk_size):
        selected = indices[start : start + chunk_size]
        raw_obs = buffer.data['obs'][selected].to(device)
        obs = algorithm._normalize_obs(raw_obs)  # pylint: disable=protected-access
        context = buffer.data['context'][selected].to(device).reshape(-1, 1)
        augmented = torch.cat((obs, context), dim=-1)
        action = buffer.data['action'][selected].to(device)
        threshold = buffer.data['risk_threshold'][selected].to(device)
        quantiles = algorithm._actor_critic.cost_critic(  # pylint: disable=protected-access
            augmented,
            action,
        )
        heads = torch.sigmoid(
            (quantiles - threshold.reshape(1, -1, 1)) / float(algo_cfgs.risk_temperature),
        ).mean(dim=-1)
        mean_risk = heads.mean(dim=0)
        outputs['predicted_risk'].append(mean_risk.cpu())
        outputs['risk_ucb'].append(
            (mean_risk + float(algo_cfgs.risk_ucb_beta) * heads.std(dim=0, unbiased=False)).cpu(),
        )
        outputs['cost'].append(quantiles.mean(dim=-1).mean(dim=0).cpu())
    result = {key: torch.cat(value) for key, value in outputs.items()}
    for key in (
        'empirical_cost_to_go',
        'risk_threshold',
        'category',
        'source_random',
        'is_failure',
    ):
        result[key] = buffer.data[key][indices].clone()
    return result


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return average ranks with exact tie handling."""
    order = torch.argsort(values)
    sorted_values = values[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    ends = counts.cumsum(dim=0).to(torch.float64)
    starts = ends - counts
    group_ranks = 0.5 * (starts + ends - 1.0)
    ranks = torch.empty(values.shape[0], dtype=torch.float64)
    ranks[order] = group_ranks[inverse]
    return ranks


def spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return tie-correct rank correlation or NaN for a degenerate sample."""
    if first.numel() < 2:
        return float('nan')
    first_rank = average_ranks(first.to(torch.float64))
    second_rank = average_ranks(second.to(torch.float64))
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if float(denominator.item()) == 0.0:
        return float('nan')
    return float((first_centered * second_centered).sum().div(denominator).item())


def calibration_rows(
    predicted: torch.Tensor,
    empirical: torch.Tensor,
    bins: int,
) -> list[dict[str, Any]]:
    """Aggregate reliability values in equal-width predicted-risk bins."""
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    assignments = torch.bucketize(predicted, boundaries[1:-1])
    rows = []
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum().item())
        if count:
            predicted_mean = float(predicted[selected].mean().item())
            empirical_mean = float(empirical[selected].mean().item())
            absolute_gap = abs(predicted_mean - empirical_mean)
        else:
            predicted_mean = None
            empirical_mean = None
            absolute_gap = None
        rows.append(
            {
                'bin': index,
                'lower': float(boundaries[index].item()),
                'upper': float(boundaries[index + 1].item()),
                'count': count,
                'predicted_risk_mean': predicted_mean,
                'empirical_risk_mean': empirical_mean,
                'absolute_gap': absolute_gap,
            },
        )
    return rows


def threshold_rows(
    audit: dict[str, torch.Tensor],
    current_low: float,
    current_safe: float,
    boundary: float,
) -> list[dict[str, Any]]:
    """Measure conservative inner-boundary coverage under a small fixed threshold grid."""
    pairs = {
        (0.02, 0.30),
        (0.02, 0.40),
        (0.02, 0.45),
        (0.02, 0.49),
        (0.05, 0.40),
        (0.05, 0.45),
        (0.05, 0.49),
        (0.10, 0.40),
        (0.10, 0.45),
        (0.10, 0.49),
        (current_low, current_safe),
    }
    rows = []
    for low, safe in sorted(pairs):
        if not 0.0 <= low < safe < boundary:
            continue
        selected = torch.logical_and(audit['risk_ucb'] >= low, audit['risk_ucb'] <= safe)
        count = int(selected.sum().item())
        rows.append(
            {
                'boundary_low': low,
                'boundary_safe': safe,
                'is_current': low == current_low and safe == current_safe,
                'sample_count': count,
                'sample_fraction': count / audit['risk_ucb'].numel(),
                'source_random_fraction': (
                    float(audit['source_random'][selected].float().mean().item()) if count else 0.0
                ),
                'immediate_violation_fraction': (
                    float(audit['is_failure'][selected].float().mean().item()) if count else 0.0
                ),
            },
        )
    return rows


def plot_calibration(
    path: Path,
    rows: list[dict[str, Any]],
    predicted: torch.Tensor,
    empirical: torch.Tensor,
) -> None:
    """Render a reliability diagram and label-conditional risk histogram."""
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    valid_rows = [row for row in rows if row['count']]
    axes[0].plot([0, 1], [0, 1], color='black', linestyle='--', label='ideal')
    axes[0].plot(
        [row['predicted_risk_mean'] for row in valid_rows],
        [row['empirical_risk_mean'] for row in valid_rows],
        marker='o',
        label='resident replay',
    )
    axes[0].set(
        xlabel='mean predicted risk',
        ylabel='empirical exceedance frequency',
        xlim=(0, 1),
        ylim=(0, 1),
        title='Risk reliability',
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for label, color, name in ((0.0, 'tab:blue', 'empirical safe'), (1.0, 'tab:red', 'exceeded')):
        selected = empirical == label
        if bool(selected.any()):
            axes[1].hist(
                predicted[selected].numpy(),
                bins=20,
                range=(0, 1),
                alpha=0.55,
                density=True,
                color=color,
                label=name,
            )
    axes[1].set(
        xlabel='predicted risk',
        ylabel='density',
        xlim=(0, 1),
        title='Predicted-risk separation',
    )
    axes[1].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def finite_or_none(value: float) -> float | None:
    """Convert non-finite diagnostic values to strict-JSON null."""
    return value if math.isfinite(value) else None


def parser() -> argparse.ArgumentParser:
    """Build replay-audit arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--max-samples', type=int, default=50_000)
    result.add_argument('--chunk-size', type=int, default=8192)
    result.add_argument('--bins', type=int, default=10)
    return result


def main() -> None:  # pylint: disable=too-many-locals
    """Restore a full PointGoal checkpoint and emit unbiased replay-risk evidence."""
    args = parser().parse_args()
    if min(args.max_samples, args.chunk_size, args.bins) <= 1:
        raise ValueError('sample, chunk, and bin sizes must exceed one')
    metadata = checkpoint_metadata(args.checkpoint)
    if metadata['environment'] != 'SafetyPointGoal1-v0':
        raise ValueError('this replay audit is specific to SafetyPointGoal1-v0')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    algorithm = build_algorithm(args.checkpoint, metadata, args.device, args.work_dir)
    try:
        resident = algorithm._buf  # pylint: disable=protected-access
        indices = deterministic_indices(resident.size, args.max_samples)
        audit = infer_replay(algorithm, indices, args.chunk_size)
        labeled = torch.isfinite(audit['empirical_cost_to_go'])
        if int(labeled.sum().item()) < 2:
            raise ValueError('replay audit requires at least two finalized cost-to-go labels')
        predicted = audit['predicted_risk'][labeled]
        empirical_cost = audit['empirical_cost_to_go'][labeled]
        threshold = audit['risk_threshold'][labeled]
        empirical_risk = (empirical_cost > threshold).float()
        rows = calibration_rows(predicted, empirical_risk, args.bins)
        algo_cfgs = algorithm._cfgs.algo_cfgs  # pylint: disable=protected-access
        sensitivity = threshold_rows(
            audit,
            float(algo_cfgs.boundary_low),
            float(algo_cfgs.boundary_safe),
            float(algo_cfgs.risk_boundary),
        )
        atomic_csv(
            args.output_dir / 'replay_risk_calibration.csv',
            list(rows[0]),
            rows,
        )
        atomic_csv(
            args.output_dir / 'replay_boundary_thresholds.csv',
            list(sensitivity[0]),
            sensitivity,
        )
        plot_calibration(
            args.output_dir / 'replay_risk_calibration.png',
            rows,
            predicted,
            empirical_risk,
        )
        weighted_gap = sum(row['absolute_gap'] * row['count'] for row in rows if row['count'])
        weighted_gap /= max(int(labeled.sum().item()), 1)
        valid = resident.size
        resident_boundary = resident.data['category'][:valid] == BOUNDARY
        boundary_count = int(resident_boundary.sum().item())
        summary = {
            **metadata,
            'checkpoint': str(args.checkpoint),
            'sample_count': int(indices.numel()),
            'sample_is_full_replay': int(indices.numel()) == metadata['replay_size'],
            'labeled_count': int(labeled.sum().item()),
            'unlabeled_count': int((~labeled).sum().item()),
            'calibration_mae': float((predicted - empirical_risk).abs().mean().item()),
            'calibration_ece': weighted_gap,
            'predicted_cost_empirical_cost_spearman': finite_or_none(
                spearman(audit['cost'][labeled], empirical_cost),
            ),
            'predicted_risk_mean': float(predicted.mean().item()),
            'empirical_risk_mean': float(empirical_risk.mean().item()),
            'predicted_risk_p10': float(torch.quantile(predicted, 0.1).item()),
            'predicted_risk_p50': float(torch.quantile(predicted, 0.5).item()),
            'predicted_risk_p90': float(torch.quantile(predicted, 0.9).item()),
            'resident_pool_sizes': resident.pool_sizes,
            'resident_boundary_source_random_fraction': (
                float(
                    resident.data['source_random'][:valid][resident_boundary].float().mean().item(),
                )
                if boundary_count
                else 0.0
            ),
            'resident_boundary_immediate_violation_fraction': (
                float(
                    resident.data['is_failure'][:valid][resident_boundary].float().mean().item(),
                )
                if boundary_count
                else 0.0
            ),
            'current_boundary_low': float(algo_cfgs.boundary_low),
            'current_boundary_safe': float(algo_cfgs.boundary_safe),
            'risk_boundary': float(algo_cfgs.risk_boundary),
        }
        atomic_json(args.output_dir / 'replay_risk_summary.json', summary)
        print(json.dumps({'status': 'passed', **summary}, indent=2, sort_keys=True))
    finally:
        algorithm._logger.close()  # pylint: disable=protected-access
        algorithm._env.close()  # pylint: disable=protected-access


if __name__ == '__main__':
    main()
