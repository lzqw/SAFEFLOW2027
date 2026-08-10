#!/usr/bin/env python3
"""Validate the four non-scientific SHAPO P4.7 engineering smoke runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


ENVIRONMENTS = {
    'point-goal': ('SafetyPointGoal1-v0', 'remaining_cost_budget', 10.0, 10.0),
    'point-button': ('SafetyPointButton1-v0', 'remaining_cost_budget', 10.0, 10.0),
    'ant': ('Ant-v4', 'future_failure_probability', 1.0, 0.0),
    'walker2d': ('Walker2d-v4', 'future_failure_probability', 1.0, 0.0),
}


def load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON mapping."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a JSON mapping')
    return payload


def require_finite(label: str, value: Any) -> float:
    """Return one finite numeric value or fail the gate."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f'{label} is not finite: {value!r}')
    return numeric


def validate_run(container: Path, expected: tuple[str, str, float, float]) -> dict[str, Any]:
    """Validate one environment receipt and its method/evaluation artifacts."""
    env_id, geometry_mode, geometry_scale, formal_budget = expected
    receipt = load_json(container / 'receipt.json')
    if receipt.get('status') != 'passed' or receipt.get('environment') != env_id:
        raise ValueError(f'{container} has an invalid training receipt')
    if int(receipt.get('steps', -1)) != 32 or int(receipt.get('run_steps', -1)) != 32:
        raise ValueError(f'{container} did not complete the 32-step engineering run')
    if receipt.get('source_dirty') is not False:
        raise ValueError(f'{container} was not run from clean committed source')

    config = yaml.safe_load((container / 'resolved_smoke_config.yaml').read_text('utf-8'))
    metadata = config['shapo_four_env_metadata']
    if metadata['engineering_smoke_only'] is not True:
        raise ValueError(f'{container} is not labelled as engineering-only')
    if metadata['geometry_mode'] != geometry_mode:
        raise ValueError(f'{container} config has the wrong geometry mode')
    if float(metadata['cost_normalization_scale']) != geometry_scale:
        raise ValueError(f'{container} config has the wrong geometry scale')
    if float(metadata['formal_cost_budget']) != formal_budget:
        raise ValueError(f'{container} config has the wrong formal cost budget')

    log_dir = Path(receipt['log_dir'])
    summary = load_json(log_dir / 'artifacts/training_summary.json')
    manifest = load_json(log_dir / 'artifacts/run_manifest.json')
    evaluation = load_json(log_dir / 'artifacts/evaluation_step_32.json')
    if summary.get('status') != 'completed' or summary.get('engineering_hard_stop_reason'):
        raise ValueError(f'{container} did not complete cleanly')
    if summary.get('cost_geometry_mode') != geometry_mode:
        raise ValueError(f'{container} training summary has the wrong geometry mode')
    if float(summary.get('cost_geometry_normalization_scale')) != geometry_scale:
        raise ValueError(f'{container} training summary has the wrong geometry scale')

    identity = manifest['p4_5_unified_cost_geometry']
    if identity['geometry_mode'] != geometry_mode:
        raise ValueError(f'{container} run manifest has the wrong geometry mode')
    if float(identity['cost_normalization_scale']) != geometry_scale:
        raise ValueError(f'{container} run manifest has the wrong geometry scale')
    if float(identity['formal_cost_budget']) != formal_budget:
        raise ValueError(f'{container} run manifest has the wrong formal budget')
    if manifest['p4_7_safety_aware_geometry']['enabled'] is not True:
        raise ValueError(f'{container} did not exercise frozen P4.7 geometry')

    geometry = evaluation['direction_geometry']
    high_risk = evaluation['direction_geometry_high_risk_top_20_percent']
    require_finite(
        f'{env_id} V_N/V_T',
        geometry['normal_to_tangent_variance_ratio'],
    )
    require_finite(
        f'{env_id} dangerous outward velocity',
        geometry['dangerous_normal_velocity'],
    )
    require_finite(
        f'{env_id} high-risk V_N/V_T',
        high_risk['normal_to_tangent_variance_ratio'],
    )
    require_finite(
        f'{env_id} high-risk dangerous outward velocity',
        high_risk['dangerous_outward_velocity'],
    )
    require_finite(f'{env_id} return', evaluation['mean_return'])
    require_finite(f'{env_id} cost', evaluation['mean_cost'])
    require_finite(f'{env_id} cumulative training cost', evaluation['training_cumulative_cost'])
    if int(evaluation['episodes']) != 1:
        raise ValueError(f'{container} evaluation episode count is not one')
    if geometry_mode == 'future_failure_probability' and not math.isclose(
        float(evaluation['mean_cost']),
        float(evaluation['failure_rate']),
        abs_tol=1e-12,
    ):
        raise ValueError(f'{container} binary terminal cost does not equal failure rate')

    required_artifacts = (
        log_dir / 'artifacts/best_safe_return_selection.json',
        log_dir / 'checkpoints/best_safe_return.pt',
        log_dir / 'checkpoints/milestone_32.pt',
    )
    for path in required_artifacts:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'{container} is missing {path}')
    return {
        'environment': env_id,
        'status': 'passed',
        'source_commit': receipt['source_commit'],
        'elapsed_seconds': require_finite(f'{env_id} elapsed time', receipt['elapsed_seconds']),
        'end_to_end_fps': require_finite(f'{env_id} FPS', receipt['end_to_end_fps']),
        'peak_cuda_memory_mib': require_finite(
            f'{env_id} peak memory',
            receipt['peak_cuda_memory_mib'],
        ),
        'mean_return': evaluation['mean_return'],
        'mean_cost': evaluation['mean_cost'],
        'failure_rate': evaluation['failure_rate'],
        'high_risk_vn_vt': high_risk['normal_to_tangent_variance_ratio'],
        'high_risk_dangerous_outward_velocity': high_risk['dangerous_outward_velocity'],
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write the aggregate gate receipt."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def main() -> None:
    """Validate every environment and seal a source-commit-specific gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    rows = [validate_run(args.run_root / slug, expected) for slug, expected in ENVIRONMENTS.items()]
    commits = {row['source_commit'] for row in rows}
    if len(commits) != 1:
        raise ValueError(f'smoke runs used different source commits: {sorted(commits)}')
    payload = {
        'status': 'passed',
        'scientific_evidence': False,
        'excluded_from_formal_aggregation': True,
        'source_commit': rows[0]['source_commit'],
        'environments': rows,
    }
    atomic_json(args.run_root / 'smoke_gate.json', payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
