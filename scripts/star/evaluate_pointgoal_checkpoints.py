#!/usr/bin/env python3
"""Re-evaluate legacy STAR PointGoal checkpoints with fixed stochastic flow samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import torch

import omnisafe
from omnisafe.adapter.star_adapter import STARContext
from omnisafe.models.critic.violation_probability_critic import (
    binary_auroc,
    binary_brier_score,
    binary_brier_skill,
)
from omnisafe.utils.star_validation import checkpoint_reconstruction_config


FlowMode = Literal['fixed-random-flow', 'center-flow-diagnostic']
DEVELOPMENT_EPISODES = 20
FORMAL_EPISODES = 50
DEFAULT_EVAL_SEED_START = 12_027
DEFAULT_FLOW_SEED_OFFSET = 1_000_003
NORMAL_95_Z = statistics.NormalDist().inv_cdf(0.975)
SUMMARY_FIELDS = (
    'episodes',
    'mean_return',
    'return_std',
    'return_ci95_low',
    'return_ci95_high',
    'mean_cost',
    'cost_std',
    'cost_ci95_low',
    'cost_ci95_high',
    'cost_cvar_20',
    'violation_ratio',
    'calibration_error',
    'brier_score',
    'brier_skill',
    'auroc',
    'spearman',
    'predicted_risk_mean',
    'empirical_risk_mean',
    'goal_met_total',
    'goals_per_episode_mean',
    'goal_met_episode_rate',
    'mean_start_end_displacement',
    'mean_action_norm',
    'mean_min_goal_distance',
    'stall_ratio',
    'step_stall_ratio',
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


def confidence_summary(values: list[float]) -> dict[str, float]:
    """Return mean, sample standard deviation, and a documented normal 95% CI."""
    if not values:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'ci95_low': float('nan'),
            'ci95_high': float('nan'),
        }
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = NORMAL_95_Z * standard_deviation / math.sqrt(len(values))
    return {
        'mean': mean,
        'std': standard_deviation,
        'ci95_low': mean - half_width,
        'ci95_high': mean + half_width,
    }


def resolve_eval_seeds(
    explicit: str | None,
    tier: Literal['development', 'formal'],
    start: int = DEFAULT_EVAL_SEED_START,
) -> list[int]:
    """Resolve explicit seeds or the fixed 20/50-seed development/formal protocol."""
    if explicit is not None:
        seeds = [int(value) for value in explicit.split(',') if value.strip()]
    else:
        count = DEVELOPMENT_EPISODES if tier == 'development' else FORMAL_EPISODES
        seeds = list(range(start, start + count))
    if not seeds:
        raise ValueError('at least one evaluation seed is required')
    if len(seeds) != len(set(seeds)):
        raise ValueError('evaluation seeds must be unique')
    return seeds


def sample_flow_base(
    action_dim: int,
    generator: torch.Generator,
    mode: FlowMode,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample a reproducible uniform flow base or the explicit center-flow diagnostic."""
    if mode == 'center-flow-diagnostic':
        return torch.zeros((1, action_dim), device=device, dtype=dtype)
    if mode != 'fixed-random-flow':
        raise ValueError(f'unknown flow evaluation mode {mode!r}')
    base = 2.0 * torch.rand((1, action_dim), generator=generator) - 1.0
    return base.to(device=device, dtype=dtype)


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
    """Reconstruct the exact legacy actor, critics, adapter, and normalizer."""
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
    has_probability_head = any(key.startswith('violation_critic.') for key in payload['model'])
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
            'legacy_checkpoint_compatibility': not has_probability_head,
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


def _truthy(value: Any) -> bool:
    """Convert scalar/tensor environment info to one boolean."""
    tensor = torch.as_tensor(value)
    return bool(tensor.bool().any().item()) if tensor.numel() else False


def _position(algorithm: Any, obs: torch.Tensor) -> tuple[tuple[float, float], str]:
    """Read simulator XY, falling back to the legacy observation projection."""
    position = algorithm._env.evaluation_position()  # pylint: disable=protected-access
    if position is not None:
        return position, 'simulator_xy'
    return (float(obs[0, 0].item()), float(obs[0, 1].item())), 'observation_projection'


def _mean_finite(values: list[float]) -> float:
    """Return the mean of finite values or NaN when the metric is unavailable."""
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float('nan')


def evaluate_rollouts(  # pylint: disable=too-many-locals,too-many-statements
    algorithm: Any,
    milestone: int,
    eval_seeds: list[int],
    mode: FlowMode,
    *,
    flow_seed_offset: int = DEFAULT_FLOW_SEED_OFFSET,
    stall_displacement_threshold: float = 0.25,
    step_stall_threshold: float = 1e-3,
) -> dict[str, Any]:
    """Evaluate one flow mode without mutating training state or global RNG state."""
    if stall_displacement_threshold < 0 or step_stall_threshold < 0:
        raise ValueError('stall thresholds must be non-negative')
    evaluation_env = algorithm._env.evaluation_env  # pylint: disable=protected-access
    algo_cfgs = algorithm._cfgs.algo_cfgs  # pylint: disable=protected-access
    device = algorithm._device  # pylint: disable=protected-access
    action_dim = algorithm._act_dim  # pylint: disable=protected-access
    returns: list[float] = []
    costs: list[float] = []
    lengths: list[int] = []
    goal_counts: list[int] = []
    displacements: list[float] = []
    episode_action_norms: list[float] = []
    minimum_goal_distances: list[float] = []
    stall_flags: list[bool] = []
    episode_step_stall_ratios: list[float] = []
    predicted_risks: list[float] = []
    predicted_costs: list[float] = []
    empirical_costs: list[float] = []
    thresholds: list[float] = []
    trajectory: list[dict[str, Any]] = []
    all_action_norms: list[float] = []
    total_stalled_movements = 0
    total_measured_movements = 0
    violations = 0
    transitions = 0
    actor_was_training = algorithm._actor_critic.training  # pylint: disable=protected-access
    algorithm._actor_critic.eval()  # pylint: disable=protected-access
    try:
        for episode_index, seed in enumerate(eval_seeds):
            obs, _ = evaluation_env.reset(seed=seed)
            context_state = STARContext(
                algorithm._env_id,  # pylint: disable=protected-access
                1,
                float(algorithm._cfgs.lagrange_cfgs.cost_limit),  # pylint: disable=protected-access
                int(algo_cfgs.max_episode_steps),
                device,
            )
            generator = torch.Generator(device='cpu')
            generator.manual_seed(flow_seed_offset + seed)
            episode_return = 0.0
            episode_cost = 0.0
            episode_goals = 0
            step_costs: list[float] = []
            step_predictions: list[float] = []
            step_predicted_costs: list[float] = []
            step_thresholds: list[float] = []
            action_norms: list[float] = []
            measured_movements = 0
            stalled_movements = 0
            start_position: tuple[float, float] | None = None
            last_position: tuple[float, float] | None = None
            previous_position: tuple[float, float] | None = None
            minimum_goal_distance = float('inf')
            for step in range(int(algo_cfgs.max_episode_steps)):
                position, position_source = _position(algorithm, obs)
                if start_position is None:
                    start_position = position
                last_position = position
                if previous_position is not None:
                    movement = math.dist(previous_position, position)
                    measured_movements += 1
                    stalled_movements += int(movement <= step_stall_threshold)
                previous_position = position
                layout = (  # pylint: disable=protected-access
                    algorithm._env.evaluation_layout() or {}
                )
                goal = layout.get('goal')
                if goal is not None:
                    minimum_goal_distance = min(minimum_goal_distance, math.dist(position, goal))

                context, threshold = context_state.values()
                augmented = algorithm._augment(obs, context)  # pylint: disable=protected-access
                base_action = sample_flow_base(
                    action_dim,
                    generator,
                    mode,
                    device=device,
                    dtype=augmented.dtype,
                )
                with torch.no_grad():
                    flow = algorithm._actor_critic.flow(  # pylint: disable=protected-access
                        augmented,
                        base_action=base_action,
                    )
                    action = flow.action
                    quantiles = (  # pylint: disable=protected-access
                        algorithm._actor_critic.cost_critic(augmented, action)
                    )
                    probability_critic = getattr(  # pylint: disable=protected-access
                        algorithm._actor_critic,
                        'violation_critic',
                        None,
                    )
                    if probability_critic is None:
                        risk = torch.sigmoid(
                            (quantiles - threshold.reshape(1, 1, 1))
                            / float(algo_cfgs.risk_temperature),
                        ).mean()
                    else:
                        risk = probability_critic.probabilities(augmented, action).mean()
                action_norm = float(torch.linalg.vector_norm(action, dim=-1).item())
                action_norms.append(action_norm)
                all_action_norms.append(action_norm)
                trajectory_row: dict[str, Any] | None = None
                if episode_index == 0:
                    trajectory_row = {
                        'milestone': milestone,
                        'evaluation_mode': mode,
                        'seed': seed,
                        'step': step,
                        'x': position[0],
                        'y': position[1],
                        'position_source': position_source,
                        'goal_x': goal[0] if goal is not None else None,
                        'goal_y': goal[1] if goal is not None else None,
                        'action_norm': action_norm,
                        'action_0': float(action[0, 0].item()),
                        'action_1': float(action[0, 1].item()),
                        'base_action_0': float(base_action[0, 0].item()),
                        'base_action_1': float(base_action[0, 1].item()),
                        'hazards_json': (
                            json.dumps(layout.get('hazards', []), separators=(',', ':'))
                            if step == 0
                            else ''
                        ),
                        'goal_radius': layout.get('goal_radius'),
                        'hazard_radius': layout.get('hazard_radius'),
                    }

                next_obs, reward, cost, terminated, truncated, info = evaluation_env.step(action)
                original_reward = float(
                    torch.as_tensor(info.get('original_reward', reward)).reshape(-1)[0].item(),
                )
                original_cost = float(
                    torch.as_tensor(info.get('original_cost', cost)).reshape(-1)[0].item(),
                )
                goal_met = _truthy(info.get('goal_met', False))
                episode_goals += int(goal_met)
                if trajectory_row is not None:
                    trajectory_row.update(
                        {
                            'reward': original_reward,
                            'cost': original_cost,
                            'goal_met': goal_met,
                        },
                    )
                    trajectory.append(trajectory_row)
                episode_return += original_reward
                episode_cost += original_cost
                step_costs.append(original_cost)
                step_predictions.append(float(risk.item()))
                step_predicted_costs.append(float(quantiles.mean().item()))
                step_thresholds.append(float(threshold.item()))
                violations += int(original_cost > 0)
                transitions += 1
                context_state.advance(
                    torch.tensor([original_cost], device=device),
                    terminated,
                    truncated,
                )
                obs = next_obs
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    break

            future_cost = 0.0
            episode_empirical = []
            for value in reversed(step_costs):
                future_cost += value
                episode_empirical.append(future_cost)
            episode_empirical.reverse()
            predicted_risks.extend(step_predictions)
            predicted_costs.extend(step_predicted_costs)
            empirical_costs.extend(episode_empirical)
            thresholds.extend(step_thresholds)
            returns.append(episode_return)
            costs.append(episode_cost)
            lengths.append(len(step_costs))
            goal_counts.append(episode_goals)
            assert start_position is not None and last_position is not None
            displacement = math.dist(start_position, last_position)
            displacements.append(displacement)
            episode_action_norms.append(statistics.fmean(action_norms))
            minimum_goal_distances.append(minimum_goal_distance)
            stall_flags.append(episode_goals == 0 and displacement <= stall_displacement_threshold)
            episode_step_stall_ratios.append(
                stalled_movements / max(measured_movements, 1),
            )
            total_stalled_movements += stalled_movements
            total_measured_movements += measured_movements
    finally:
        if actor_was_training:
            algorithm._actor_critic.train()  # pylint: disable=protected-access

    predicted_tensor = torch.tensor(predicted_risks)
    predicted_cost_tensor = torch.tensor(predicted_costs)
    empirical_tensor = torch.tensor(empirical_costs)
    threshold_tensor = torch.tensor(thresholds)
    empirical_risk = (empirical_tensor > threshold_tensor).float()
    return_stats = confidence_summary(returns)
    cost_stats = confidence_summary(costs)
    return {
        'milestone': milestone,
        'evaluation_mode': mode,
        'evaluation_role': 'primary' if mode == 'fixed-random-flow' else 'diagnostic',
        'base_action_policy': (
            'uniform[-1,1], local CPU generator seeded by flow_seed_offset + evaluation_seed'
            if mode == 'fixed-random-flow'
            else 'all-zero center base action'
        ),
        'flow_seed_offset': flow_seed_offset if mode == 'fixed-random-flow' else None,
        'episodes': len(returns),
        'seeds': eval_seeds,
        'returns': returns,
        'costs': costs,
        'lengths': lengths,
        'goal_met_counts': goal_counts,
        'start_end_displacements': displacements,
        'episode_mean_action_norms': episode_action_norms,
        'minimum_goal_distances': minimum_goal_distances,
        'stall_flags': stall_flags,
        'episode_step_stall_ratios': episode_step_stall_ratios,
        'mean_return': return_stats['mean'],
        'return_std': return_stats['std'],
        'return_ci95_low': return_stats['ci95_low'],
        'return_ci95_high': return_stats['ci95_high'],
        'mean_cost': cost_stats['mean'],
        'cost_std': cost_stats['std'],
        'cost_ci95_low': cost_stats['ci95_low'],
        'cost_ci95_high': cost_stats['ci95_high'],
        'ci95_method': 'normal approximation: mean +/- 1.959964 * sample_std / sqrt(n)',
        'cost_cvar_20': _tail_mean(costs),
        'violation_ratio': violations / max(transitions, 1),
        'calibration_error': float((predicted_tensor - empirical_risk).abs().mean().item()),
        'brier_score': binary_brier_score(predicted_tensor, empirical_risk),
        'brier_skill': binary_brier_skill(predicted_tensor, empirical_risk),
        'auroc': binary_auroc(predicted_tensor, empirical_risk),
        'spearman': _spearman(predicted_cost_tensor, empirical_tensor),
        'predicted_risk_mean': float(predicted_tensor.mean().item()),
        'empirical_risk_mean': float(empirical_risk.mean().item()),
        'goal_met_total': sum(goal_counts),
        'goals_per_episode_mean': statistics.fmean(goal_counts),
        'goal_met_episode_rate': sum(value > 0 for value in goal_counts) / len(goal_counts),
        'mean_start_end_displacement': statistics.fmean(displacements),
        'mean_action_norm': statistics.fmean(all_action_norms),
        'mean_min_goal_distance': _mean_finite(minimum_goal_distances),
        'stall_ratio': statistics.fmean(stall_flags),
        'stall_definition': (
            f'goal_met_count == 0 and start_end_displacement <= {stall_displacement_threshold}'
        ),
        'stall_displacement_threshold': stall_displacement_threshold,
        'step_stall_ratio': total_stalled_movements / max(total_measured_movements, 1),
        'step_stall_definition': f'consecutive simulator displacement <= {step_stall_threshold}',
        'step_stall_threshold': step_stall_threshold,
        'trajectory': trajectory,
    }


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    """Call the repository tie-correct implementation without relying on STAR internals."""
    from omnisafe.utils.star_validation import spearman  # pylint: disable=import-outside-toplevel

    return spearman(first, second)


def _tail_mean(values: list[float], fraction: float = 0.2) -> float:
    """Return the upper-tail mean used as empirical cost CVaR."""
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    return sum(sorted(values, reverse=True)[:count]) / count


def public_row(
    payload: dict[str, Any],
    checkpoint: Path,
    evaluation: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Return one compact public CSV row from one evaluation mode."""
    row = {
        'training_seed': int(payload['extra']['seed']),
        'milestone': int(payload['counters']['env_steps']),
        'checkpoint': str(checkpoint),
        'evaluation_device': device,
        'evaluation_mode': evaluation.get('evaluation_mode', 'legacy-unspecified'),
        'evaluation_role': evaluation.get('evaluation_role', 'legacy-unspecified'),
        'spearman_method': 'average-rank tie-correct',
        'ci95_method': evaluation.get('ci95_method'),
    }
    row.update(
        {
            field: (
                None
                if isinstance(evaluation.get(field), float) and not math.isfinite(evaluation[field])
                else evaluation.get(field)
            )
            for field in SUMMARY_FIELDS
        },
    )
    return row


def evaluate_checkpoint(  # pylint: disable=too-many-arguments,too-many-locals
    checkpoint: Path,
    output_dir: Path,
    work_dir: Path,
    device: str,
    eval_seeds: list[int],
    *,
    flow_seed_offset: int,
    stall_displacement_threshold: float,
    step_stall_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate fixed-random and center-flow modes and write their raw artifacts."""
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
        primary = evaluate_rollouts(
            algorithm,
            step,
            eval_seeds,
            'fixed-random-flow',
            flow_seed_offset=flow_seed_offset,
            stall_displacement_threshold=stall_displacement_threshold,
            step_stall_threshold=step_stall_threshold,
        )
        center = evaluate_rollouts(
            algorithm,
            step,
            eval_seeds,
            'center-flow-diagnostic',
            flow_seed_offset=flow_seed_offset,
            stall_displacement_threshold=stall_displacement_threshold,
            step_stall_threshold=step_stall_threshold,
        )
    finally:
        algorithm._logger.close()  # pylint: disable=protected-access
        algorithm._env.close()  # pylint: disable=protected-access

    primary_trajectory = primary.pop('trajectory')
    center_trajectory = center.pop('trajectory')
    common = {
        'training_seed': seed,
        'environment': 'SafetyPointGoal1-v0',
        'checkpoint': str(checkpoint),
        'evaluation_device': device,
        'spearman_method': 'average-rank tie-correct',
    }
    result = {**common, **primary, 'center_flow_diagnostic': center}
    atomic_json(output_dir / f'evaluation_seed_{seed}_step_{step}.json', result)
    atomic_json(
        output_dir / f'center_flow_diagnostic_seed_{seed}_step_{step}.json',
        {**common, **center},
    )
    if primary_trajectory:
        atomic_csv(output_dir / f'trajectory_seed_{seed}_step_{step}.csv', primary_trajectory)
    if center_trajectory:
        atomic_csv(
            output_dir / f'center_flow_trajectory_seed_{seed}_step_{step}.csv',
            center_trajectory,
        )
    return (
        public_row(payload, checkpoint, primary, device),
        public_row(payload, checkpoint, center, device),
    )


def parser() -> argparse.ArgumentParser:
    """Build checkpoint-evaluation arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument(
        '--evaluation-tier',
        choices=('development', 'formal'),
        default='development',
    )
    result.add_argument('--eval-seeds')
    result.add_argument('--eval-seed-start', type=int, default=DEFAULT_EVAL_SEED_START)
    result.add_argument('--flow-seed-offset', type=int, default=DEFAULT_FLOW_SEED_OFFSET)
    result.add_argument('--stall-displacement-threshold', type=float, default=0.25)
    result.add_argument('--step-stall-threshold', type=float, default=1e-3)
    return result


def main() -> None:
    """Evaluate requested checkpoints and emit separate primary/diagnostic CSVs."""
    args = parser().parse_args()
    eval_seeds = resolve_eval_seeds(args.eval_seeds, args.evaluation_tier, args.eval_seed_start)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    primary_rows: list[dict[str, Any]] = []
    center_rows: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        primary, center = evaluate_checkpoint(
            checkpoint,
            args.output_dir,
            args.work_dir,
            args.device,
            eval_seeds,
            flow_seed_offset=args.flow_seed_offset,
            stall_displacement_threshold=args.stall_displacement_threshold,
            step_stall_threshold=args.step_stall_threshold,
        )
        primary_rows.append(primary)
        center_rows.append(center)
    primary_rows.sort(key=lambda row: (row['milestone'], row['training_seed']))
    center_rows.sort(key=lambda row: (row['milestone'], row['training_seed']))
    atomic_csv(args.output_dir / 'pointgoal_checkpoint_evaluation.csv', primary_rows)
    atomic_csv(args.output_dir / 'pointgoal_center_flow_diagnostic.csv', center_rows)
    print(
        json.dumps(
            {
                'status': 'passed',
                'checkpoints': len(primary_rows),
                'evaluation_tier': args.evaluation_tier,
                'evaluation_seeds': eval_seeds,
                'primary_mode': 'fixed-random-flow',
                'diagnostic_mode': 'center-flow-diagnostic',
                'output_dir': str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == '__main__':
    main()
