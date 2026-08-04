#!/usr/bin/env python3
"""Render PointGoal action-risk, flow-storyboard, and trajectory mechanism artifacts."""

# The two standalone visualization CLIs intentionally share compact layout-rendering logic.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle

import omnisafe
from omnisafe.utils.star_validation import checkpoint_reconstruction_config, deterministic_indices


plt.switch_backend('Agg')


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write compact mechanism metadata atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Read only the configuration fields needed to reconstruct the model and replay."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'star-checkpoint' or payload.get('version') != 1:
        raise ValueError(f'{path} is not a STAR checkpoint')
    if payload.get('replay') is None:
        raise ValueError('mechanism visualization requires a full checkpoint with replay')
    if not any(key.startswith('violation_critic.') for key in payload['model']):
        raise ValueError('P4.1 mechanism visualization requires the probability head')
    return {
        'env_steps': int(payload['counters']['env_steps']),
        'environment': str(payload['extra']['environment']),
        'seed': int(payload['extra']['seed']),
        'replay_capacity': int(payload['replay']['capacity']),
        'reconstruction_cfgs': checkpoint_reconstruction_config(payload),
    }


def build_algorithm(
    checkpoint: Path,
    metadata: dict[str, Any],
    device: str,
    work_dir: Path,
) -> Any:
    """Restore the exact model, normalizer, and resident replay without training."""
    custom = metadata['reconstruction_cfgs']
    custom['seed'] = metadata['seed']
    custom['train_cfgs'] = {
        'device': device,
        'torch_threads': 1,
        'vector_env_nums': 1,
        'parallel': 1,
        'total_steps': metadata['env_steps'],
        'eval_episodes': 0,
    }
    algo_cfgs = custom.setdefault('algo_cfgs', {})
    algo_cfgs.update(
        {
            'steps_per_epoch': max(1, metadata['env_steps']),
            'size': metadata['replay_capacity'],
            'batch_size': max(1, min(256, metadata['replay_capacity'] - 1)),
            'eval_milestones': [],
            'resume_from': str(checkpoint),
            'milestone_checkpoints': False,
            'checkpoint_replay': False,
        },
    )
    custom['logger_cfgs'] = {
        'log_dir': str(work_dir),
        'use_tensorboard': False,
        'use_wandb': False,
    }
    wrapper = omnisafe.Agent('STAR', metadata['environment'], custom_cfgs=custom)
    return wrapper.agent


def representative_state(algorithm: Any, sample_size: int) -> dict[str, torch.Tensor]:
    """Select a deterministic temporal replay midpoint without consulting predicted risk."""
    resident = algorithm._buf  # pylint: disable=protected-access
    indices = deterministic_indices(resident.size, sample_size)
    index = int(indices[indices.numel() // 2].item())
    device = algorithm._device  # pylint: disable=protected-access
    raw_obs = resident.data['obs'][index : index + 1].to(device)
    context = resident.data['context'][index : index + 1].to(device)
    augmented = algorithm._augment(  # pylint: disable=protected-access
        raw_obs,
        context,
    )
    with torch.no_grad():
        action = algorithm._actor_critic.step(  # pylint: disable=protected-access
            augmented,
            deterministic=True,
        )
        critic = algorithm._actor_critic.violation_critic  # pylint: disable=protected-access
        if critic is None:
            raise RuntimeError('P4.1 probability critic is unavailable')
        risk_heads = critic.probabilities(augmented, action)
        risk = risk_heads.mean(dim=0)
    return {
        'raw_obs': raw_obs,
        'augmented': augmented,
        'context': context,
        'deterministic_action': action,
        'risk': risk,
        'risk_heads': risk_heads,
        'replay_index': torch.tensor(index),
    }


def action_risk_grid(
    algorithm: Any,
    state: dict[str, torch.Tensor],
    grid_size: int,
) -> dict[str, torch.Tensor]:
    """Evaluate every ensemble head on a dense two-dimensional action grid."""
    device = algorithm._device  # pylint: disable=protected-access
    algo_cfgs = algorithm._cfgs.algo_cfgs  # pylint: disable=protected-access
    axis = torch.linspace(
        -1.0,
        1.0,
        grid_size,
        device=device,
    )
    first, second = torch.meshgrid(axis, axis, indexing='ij')
    actions = torch.stack((first.reshape(-1), second.reshape(-1)), dim=-1)
    augmented = state['augmented'].expand(actions.shape[0], -1)
    with torch.no_grad():
        critic = algorithm._actor_critic.violation_critic  # pylint: disable=protected-access
        if critic is None:
            raise RuntimeError('P4.1 probability critic is unavailable')
        heads = critic.probabilities(augmented, actions)
    mean = heads.mean(dim=0)
    ucb = mean + float(
        algo_cfgs.risk_ucb_beta,
    ) * heads.std(
        dim=0,
        unbiased=False,
    )
    ucb = ucb.clamp(0.0, 1.0)
    return {'axis': axis, 'actions': actions, 'heads': heads, 'mean': mean, 'ucb': ucb}


def fixed_budget_risk_strata(  # pylint: disable=too-many-locals
    algorithm: Any,
    sample_size: int,
    grid_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select fixed-budget replay strata and measure each selected action-risk surface."""
    device = algorithm._device  # pylint: disable=protected-access
    algo_cfgs = algorithm._cfgs.algo_cfgs  # pylint: disable=protected-access
    critic = algorithm._actor_critic.violation_critic  # pylint: disable=protected-access
    if critic is None:
        raise RuntimeError('P4.1 probability critic is unavailable')
    resident = algorithm._buf  # pylint: disable=protected-access
    replay_indices = deterministic_indices(resident.size, sample_size)
    replay_obs = resident.data['obs'][replay_indices].to(device)
    replay_context = resident.data['context'][replay_indices].to(device)
    axis = torch.linspace(-1.0, 1.0, grid_size, device=device)
    first, second = torch.meshgrid(axis, axis, indexing='ij')
    base_actions = torch.stack((first.reshape(-1), second.reshape(-1)), dim=-1)
    boundary = float(algo_cfgs.risk_boundary)
    bandwidth = float(algo_cfgs.boundary_bandwidth)
    ucb_beta = float(algo_cfgs.risk_ucb_beta)
    rows: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []

    for budget_ratio in (1.0, 0.5, 0.2, 0.0):
        context = replay_context.clone()
        context[:, 0] = budget_ratio
        augmented_candidates = algorithm._augment(  # pylint: disable=protected-access
            replay_obs,
            context,
        )
        with torch.no_grad():
            candidate_actions = algorithm._actor_critic.step(  # pylint: disable=protected-access
                augmented_candidates,
                deterministic=True,
            )
            candidate_heads = critic.probabilities(augmented_candidates, candidate_actions)
            candidate_mean = candidate_heads.mean(dim=0)
            candidate_ucb = (
                candidate_mean + ucb_beta * candidate_heads.std(dim=0, unbiased=False)
            ).clamp(0.0, 1.0)
        strata = {
            'safe': candidate_ucb < boundary - bandwidth,
            'boundary': (candidate_ucb - boundary).abs() <= bandwidth,
            'high_risk': candidate_ucb > boundary + bandwidth,
        }
        for name, selected in strata.items():
            count = int(selected.sum().item())
            risk_span = None
            local_gradient_norm = None
            disagreement = None
            probability_disagreement = None
            normal = [None, None]
            selected_action = [None, None]
            selected_mean_risk = None
            selected_risk_ucb = None
            selected_replay_index = None
            surface_actions = torch.empty((0, 2))
            surface_risk = torch.empty(0)
            if count:
                positions = torch.nonzero(selected, as_tuple=False).flatten()
                stratum_risk = candidate_ucb[positions]
                if name == 'safe':
                    relative_index = int(stratum_risk.argmin().item())
                elif name == 'boundary':
                    relative_index = int((stratum_risk - boundary).abs().argmin().item())
                else:
                    relative_index = int(stratum_risk.argmax().item())
                position = int(positions[relative_index].item())
                chosen_augmented = augmented_candidates[position : position + 1]
                chosen_action = candidate_actions[position : position + 1]
                surface_augmented = chosen_augmented.expand(base_actions.shape[0], -1)
                with torch.no_grad():
                    surface_heads = critic.probabilities(surface_augmented, base_actions)
                    surface_mean = surface_heads.mean(dim=0)
                risk_span = float(
                    (torch.quantile(surface_mean, 0.95) - torch.quantile(surface_mean, 0.05))
                    .detach()
                    .item(),
                )
                differentiable_action = chosen_action.detach().clone().requires_grad_(True)
                parameter_flags = [parameter.requires_grad for parameter in critic.parameters()]
                critic.requires_grad_(False)
                try:
                    local_heads = critic.probabilities(chosen_augmented, differentiable_action)
                    head_gradients = torch.stack(
                        [
                            torch.autograd.grad(
                                local_heads[index].sum(),
                                differentiable_action,
                                retain_graph=True,
                            )[0]
                            for index in range(local_heads.shape[0])
                        ],
                        dim=0,
                    )
                finally:
                    for parameter, flag in zip(critic.parameters(), parameter_flags):
                        parameter.requires_grad_(flag)
                gradient = head_gradients.mean(dim=0)[0]
                gradient_norm = gradient.norm()
                local_gradient_norm = float(gradient_norm.detach().item())
                normal_tensor = (gradient / gradient_norm.clamp_min(1e-8)).detach().cpu()
                normal = [float(normal_tensor[0]), float(normal_tensor[1])]
                unit_heads = head_gradients / head_gradients.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)
                agreements = []
                for first_head in range(head_gradients.shape[0]):
                    for second_head in range(first_head + 1, head_gradients.shape[0]):
                        cosine = (unit_heads[first_head] * unit_heads[second_head]).sum(dim=-1)
                        agreements.append(0.5 * (1.0 + cosine.clamp(-1.0, 1.0)))
                consistency = (
                    torch.stack(agreements).mean() if agreements else torch.ones((), device=device)
                )
                disagreement = float((1.0 - consistency).detach().item())
                probability_disagreement = float(
                    local_heads.std(dim=0, unbiased=False).mean().detach().item(),
                )
                selected_action = [
                    float(chosen_action[0, 0].item()),
                    float(chosen_action[0, 1].item()),
                ]
                selected_mean_risk = float(candidate_mean[position].item())
                selected_risk_ucb = float(candidate_ucb[position].item())
                selected_replay_index = int(replay_indices[position].item())
                surface_actions = base_actions.detach().cpu()
                surface_risk = surface_mean.detach().cpu()
            row = {
                'budget_ratio': budget_ratio,
                'risk_stratum': name,
                'sample_count': count,
                'selected_replay_index': selected_replay_index,
                'selected_predicted_risk': selected_mean_risk,
                'selected_risk_ucb': selected_risk_ucb,
                'selected_action_0': selected_action[0],
                'selected_action_1': selected_action[1],
                'risk_p95_minus_p05': risk_span,
                'gradient_norm': local_gradient_norm,
                'ensemble_disagreement': disagreement,
                'probability_disagreement': probability_disagreement,
                'normal_0': normal[0],
                'normal_1': normal[1],
            }
            rows.append(row)
            panels.append(
                {
                    **row,
                    'actions': surface_actions,
                    'risk': surface_risk,
                },
            )
    return rows, panels


def write_fixed_budget_strata(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the fixed 4-by-3 P4.1 geometry evidence table."""
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_fixed_budget_strata(path: Path, panels: list[dict[str, Any]], step: int) -> None:
    """Render action-risk surfaces for fixed-budget safe/boundary/high-risk states."""
    figure, axes = plt.subplots(4, 3, figsize=(10.5, 12.5), sharex=True, sharey=True)
    for panel, axis in zip(panels, axes.flat):
        actions = panel['actions']
        risk = panel['risk']
        if actions.numel():
            axis.scatter(
                actions[:, 0],
                actions[:, 1],
                c=risk,
                cmap='coolwarm',
                vmin=0.0,
                vmax=1.0,
                s=7,
            )
            axis.quiver(
                panel['selected_action_0'],
                panel['selected_action_1'],
                panel['normal_0'],
                panel['normal_1'],
                color='black',
                scale=4.0,
                width=0.012,
            )
        axis.set_title(
            f"b={panel['budget_ratio']:.1f}, {panel['risk_stratum']}\n"
            f"states={panel['sample_count']}, |∇p|={panel['gradient_norm'] or 0.0:.3g}",
        )
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
    for axis in axes[-1]:
        axis.set_xlabel('action[0]')
    for axis in axes[:, 0]:
        axis.set_ylabel('action[1]')
    figure.suptitle(f'P4.1 fixed-budget probability geometry at step {step:,}')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def risk_normal(algorithm: Any, state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the normalized critic action gradient at the deterministic action."""
    action = state['deterministic_action'].detach().clone().requires_grad_(True)
    critic = algorithm._actor_critic.violation_critic  # pylint: disable=protected-access
    if critic is None:
        raise RuntimeError('P4.1 probability critic is unavailable')
    flags = [parameter.requires_grad for parameter in critic.parameters()]
    critic.requires_grad_(False)
    try:
        risk = critic.probabilities(state['augmented'], action).mean()
        gradient = torch.autograd.grad(risk, action, create_graph=False)[0]
    finally:
        for parameter, flag in zip(critic.parameters(), flags):
            parameter.requires_grad_(flag)
    return gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def flow_storyboard(
    algorithm: Any,
    state: dict[str, torch.Tensor],
    base_grid_size: int,
) -> dict[str, torch.Tensor]:
    """Transport a fixed grid of base actions through the learned action-space flow."""
    device = algorithm._device  # pylint: disable=protected-access
    axis = torch.linspace(
        -0.9,
        0.9,
        base_grid_size,
        device=device,
    )
    first, second = torch.meshgrid(axis, axis, indexing='ij')
    base = torch.stack((first.reshape(-1), second.reshape(-1)), dim=-1)
    augmented = state['augmented'].expand(base.shape[0], -1)
    flow = algorithm._actor_critic.flow(  # pylint: disable=protected-access
        augmented,
        base_action=base,
    )
    points = flow.points
    repeated_obs = (
        augmented.unsqueeze(1)
        .expand(-1, points.shape[1], -1)
        .reshape(
            -1,
            augmented.shape[-1],
        )
    )
    with torch.no_grad():
        critic = algorithm._actor_critic.violation_critic  # pylint: disable=protected-access
        if critic is None:
            raise RuntimeError('P4.1 probability critic is unavailable')
        risk = critic.probabilities(repeated_obs, points.reshape(-1, 2)).mean(dim=0)
    risk = risk.reshape(points.shape[0], points.shape[1])
    times = torch.linspace(0.0, 1.0, points.shape[1], device=points.device)
    return {'base': base, 'points': points, 'risk': risk, 'times': times}


def write_action_grid(path: Path, grid: dict[str, torch.Tensor]) -> None:
    """Write the plotted action-risk values as a compact CSV."""
    heads = grid['heads'].detach().cpu()
    with path.open('w', encoding='utf-8', newline='') as stream:
        fields = [
            'action_0',
            'action_1',
            'predicted_risk',
            'risk_ucb',
            *[f'head_{index}' for index in range(heads.shape[0])],
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, action in enumerate(grid['actions'].detach().cpu().tolist()):
            writer.writerow(
                {
                    'action_0': action[0],
                    'action_1': action[1],
                    'predicted_risk': float(grid['mean'][index].item()),
                    'risk_ucb': float(grid['ucb'][index].item()),
                    **{
                        f'head_{head}': float(heads[head, index].item())
                        for head in range(heads.shape[0])
                    },
                },
            )


def write_storyboard(path: Path, story: dict[str, torch.Tensor]) -> None:
    """Write every transported action point and its critic risk."""
    points = story['points'].detach().cpu()
    risks = story['risk'].detach().cpu()
    times = story['times'].detach().cpu()
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=['base_index', 'time', 'action_0', 'action_1', 'predicted_risk'],
        )
        writer.writeheader()
        for base_index in range(points.shape[0]):
            for time_index in range(points.shape[1]):
                writer.writerow(
                    {
                        'base_index': base_index,
                        'time': float(times[time_index].item()),
                        'action_0': float(points[base_index, time_index, 0].item()),
                        'action_1': float(points[base_index, time_index, 1].item()),
                        'predicted_risk': float(risks[base_index, time_index].item()),
                    },
                )


def plot_action_risk(  # pylint: disable=too-many-arguments
    path: Path,
    grid: dict[str, torch.Tensor],
    story: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    normal: torch.Tensor,
    step: int,
) -> None:
    """Plot predicted risk, the risk contour, flow terminals, and local critic normal."""
    axis = grid['axis'].detach().cpu().numpy()
    risk = grid['mean'].reshape(axis.size, axis.size).detach().cpu().numpy()
    figure, axes = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = axes.contourf(axis, axis, risk.T, levels=20, cmap='coolwarm', vmin=0.0, vmax=1.0)
    if float(grid['mean'].min().item()) <= 0.5 <= float(grid['mean'].max().item()):
        axes.contour(axis, axis, risk.T, levels=[0.5], colors='black', linewidths=1.5)
    terminals = story['points'][:, -1].detach().cpu()
    axes.scatter(
        terminals[:, 0],
        terminals[:, 1],
        s=16,
        c='white',
        edgecolors='black',
        linewidths=0.3,
    )
    action = state['deterministic_action'][0].detach().cpu()
    direction = normal[0].detach().cpu()
    axes.quiver(
        float(action[0]),
        float(action[1]),
        float(direction[0]),
        float(direction[1]),
        color='black',
        scale=5.0,
        width=0.008,
    )
    axes.scatter([float(action[0])], [float(action[1])], marker='*', s=120, color='gold')
    axes.set(xlabel='action[0]', ylabel='action[1]', xlim=(-1, 1), ylim=(-1, 1))
    axes.set_title(f'STAR predicted action risk at step {step:,}')
    figure.colorbar(image, ax=axes, label='predicted exceedance risk')
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_storyboard(
    path: Path,
    grid: dict[str, torch.Tensor],
    story: dict[str, torch.Tensor],
    step: int,
) -> None:
    """Plot action samples at every ODE integration time over the same risk surface."""
    axis = grid['axis'].detach().cpu().numpy()
    risk_surface = grid['mean'].reshape(axis.size, axis.size).detach().cpu().numpy()
    times = story['times'].detach().cpu().tolist()
    figure, axes = plt.subplots(
        1,
        len(times),
        figsize=(3.2 * len(times), 3.2),
        sharex=True,
        sharey=True,
    )
    for index, (time_value, panel) in enumerate(zip(times, axes)):
        panel.contourf(
            axis,
            axis,
            risk_surface.T,
            levels=20,
            cmap='coolwarm',
            vmin=0.0,
            vmax=1.0,
        )
        points = story['points'][:, index].detach().cpu()
        panel.scatter(
            points[:, 0],
            points[:, 1],
            c=story['risk'][:, index].detach().cpu(),
            cmap='coolwarm',
            vmin=0.0,
            vmax=1.0,
            s=18,
            edgecolors='black',
            linewidths=0.2,
        )
        panel.set(title=f't={time_value:.2f}', xlabel='action[0]', xlim=(-1, 1), ylim=(-1, 1))
    axes[0].set_ylabel('action[1]')
    figure.suptitle(f'STAR action-space flow storyboard at step {step:,}')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_trajectories(  # pylint: disable=too-many-locals
    artifact_dir: Path,
    output: Path,
) -> int:
    """Plot every fixed-seed PointGoal trajectory found in a run artifact directory."""
    paths = sorted(
        (
            *artifact_dir.glob('trajectory_step_*.csv'),
            *artifact_dir.glob('trajectory_seed_*_step_*.csv'),
        ),
        key=lambda path: int(path.stem.rsplit('_', maxsplit=1)[-1]),
    )
    if not paths:
        return 0
    figure, axes = plt.subplots(figsize=(6.0, 5.5), constrained_layout=True)
    hazards_drawn = False
    goals_labeled = False
    for path in paths:
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        step = int(rows[0]['milestone'])
        x_values = [float(row['x']) for row in rows]
        y_values = [float(row['y']) for row in rows]
        axes.plot(x_values, y_values, linewidth=1.4, label=f'{step // 1000}k')
        axes.scatter([x_values[0]], [y_values[0]], s=18, marker='o')
        axes.scatter([x_values[-1]], [y_values[-1]], s=28, marker='x')
        if not hazards_drawn:
            hazards_json = next(
                (row.get('hazards_json', '') for row in rows if row.get('hazards_json')),
                '',
            )
            hazards = json.loads(hazards_json) if hazards_json else []
            radius_text = rows[0].get('hazard_radius', '')
            hazard_radius = float(radius_text) if radius_text else 0.2
            for hazard_index, center in enumerate(hazards):
                axes.add_patch(
                    Circle(
                        center,
                        hazard_radius,
                        color='tab:red',
                        alpha=0.22,
                        label='hazard' if hazard_index == 0 else None,
                    ),
                )
            hazards_drawn = bool(hazards)
        goals = {
            (float(row['goal_x']), float(row['goal_y']))
            for row in rows
            if row.get('goal_x') and row.get('goal_y')
        }
        if goals:
            goal_x, goal_y = zip(*sorted(goals))
            axes.scatter(
                goal_x,
                goal_y,
                s=70,
                marker='*',
                color='tab:green',
                label=None if goals_labeled else 'goal locations',
                zorder=4,
            )
            goals_labeled = True
    axes.set(xlabel='simulator x', ylabel='simulator y', title='Fixed-seed PointGoal trajectories')
    axes.axis('equal')
    axes.legend(title='checkpoint')
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return len(paths)


def parser() -> argparse.ArgumentParser:
    """Build mechanism-rendering arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--work-dir', type=Path, required=True)
    result.add_argument('--trajectory-artifact-dir', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--sample-size', type=int, default=50_000)
    result.add_argument('--grid-size', type=int, default=51)
    result.add_argument('--base-grid-size', type=int, default=7)
    return result


def main() -> None:  # pylint: disable=too-many-locals
    """Restore a full checkpoint and emit all required PointGoal mechanism visuals."""
    args = parser().parse_args()
    if min(args.sample_size, args.grid_size, args.base_grid_size) <= 1:
        raise ValueError('visualization grid and sample sizes must exceed one')
    metadata = checkpoint_metadata(args.checkpoint)
    if metadata['environment'] != 'SafetyPointGoal1-v0':
        raise ValueError('this visualization is specific to SafetyPointGoal1-v0')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    algorithm = build_algorithm(args.checkpoint, metadata, args.device, args.work_dir)
    try:
        if algorithm._act_dim != 2:  # pylint: disable=protected-access
            raise ValueError('PointGoal action-risk map requires a two-dimensional action')
        state = representative_state(algorithm, args.sample_size)
        grid = action_risk_grid(algorithm, state, args.grid_size)
        normal = risk_normal(algorithm, state)
        story = flow_storyboard(algorithm, state, args.base_grid_size)
        strata_rows, strata_panels = fixed_budget_risk_strata(
            algorithm,
            args.sample_size,
            args.grid_size,
        )
        write_action_grid(args.output_dir / 'action_risk_map.csv', grid)
        write_storyboard(args.output_dir / 'flow_storyboard.csv', story)
        write_fixed_budget_strata(
            args.output_dir / 'fixed_budget_risk_strata.csv',
            strata_rows,
        )
        plot_action_risk(
            args.output_dir / 'action_risk_map.png',
            grid,
            story,
            state,
            normal,
            metadata['env_steps'],
        )
        plot_storyboard(
            args.output_dir / 'flow_storyboard.png',
            grid,
            story,
            metadata['env_steps'],
        )
        plot_fixed_budget_strata(
            args.output_dir / 'fixed_budget_risk_strata.png',
            strata_panels,
            metadata['env_steps'],
        )
        trajectory_count = 0
        if args.trajectory_artifact_dir is not None:
            trajectory_count = plot_trajectories(
                args.trajectory_artifact_dir,
                args.output_dir / 'pointgoal_trajectories.png',
            )
        atomic_json(
            args.output_dir / 'mechanism_snapshot.json',
            {
                'env_steps': metadata['env_steps'],
                'environment': metadata['environment'],
                'seed': metadata['seed'],
                'replay_capacity': metadata['replay_capacity'],
                'checkpoint': str(args.checkpoint),
                'representative_selection': (
                    'deterministic_time_stratified_midpoint_independent_of_predicted_risk'
                ),
                'representative_replay_index': int(state['replay_index'].item()),
                'representative_context': state['context'].flatten().cpu().tolist(),
                'representative_predicted_risk': float(state['risk'].item()),
                'representative_head_risks': state['risk_heads'].flatten().detach().cpu().tolist(),
                'representative_action': state['deterministic_action'].flatten().cpu().tolist(),
                'risk_normal': normal.flatten().detach().cpu().tolist(),
                'risk_grid_min': float(grid['mean'].min().item()),
                'risk_grid_max': float(grid['mean'].max().item()),
                'flow_terminal_risk_mean': float(story['risk'][:, -1].mean().item()),
                'fixed_budget_risk_strata': strata_rows,
                'trajectory_files': trajectory_count,
            },
        )
    finally:
        algorithm._logger.close()  # pylint: disable=protected-access
        algorithm._env.close()  # pylint: disable=protected-access


if __name__ == '__main__':
    main()
