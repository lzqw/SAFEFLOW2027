"""Targeted tests for the metadata-only PointCircle artifact repair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.star.repair_pointcircle_screen_metadata import (
    ENVIRONMENT,
    repair_available,
    scientific_sha256,
)


def evaluation_path(root: Path, arm: str, milestone: int) -> Path:
    """Return one exact synthetic screen artifact path."""
    return (
        root
        / f'{arm}-2027/training/STAR-{{SafetyPointCircle1-v0}}'
        / 'seed-2027-test/artifacts'
        / f'evaluation_step_{milestone}.json'
    )


def payload(milestone: int, environment: str | None = None) -> dict:
    """Build the frozen metadata fields plus representative scientific values."""
    result = {
        'milestone': milestone,
        'episodes': 20,
        'base_seeds': [22027, 22028, 22029, 22030, 22031],
        'episodes_per_base_seed': 4,
        'mean_return': 28.70773493,
        'mean_cost': 0.0,
        'cost_cvar_20': 0.0,
        'direction_geometry_high_risk_top_20_percent': {
            'normal_to_tangent_variance_ratio': 0.9999995,
            'dangerous_outward_velocity': 0.01738491,
        },
    }
    if environment is not None:
        result['environment'] = environment
    return result


def test_repair_adds_only_environment_and_is_restartable(tmp_path: Path) -> None:
    screen = tmp_path / 'screen'
    artifact = evaluation_path(screen, 'P47', 150_000)
    artifact.parent.mkdir(parents=True)
    original = payload(150_000)
    artifact.write_text(json.dumps(original), encoding='utf-8')
    state_path = tmp_path / 'repair-state.json'

    first = repair_available(screen, state_path, 'a' * 40, 'b' * 40)
    repaired = json.loads(artifact.read_text(encoding='utf-8'))
    assert repaired['environment'] == ENVIRONMENT
    assert scientific_sha256(repaired) == scientific_sha256(original)
    entry = first['artifacts']['P47:150000']
    assert entry['action'] == 'added_missing_environment'
    assert entry['scientific_metrics_changed'] is False

    second = repair_available(screen, state_path, 'a' * 40, 'b' * 40)
    assert second['artifacts']['P47:150000'] == entry


def test_repair_rejects_another_environment(tmp_path: Path) -> None:
    screen = tmp_path / 'screen'
    artifact = evaluation_path(screen, 'C0', 150_000)
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(payload(150_000, environment='SafetyPointGoal1-v0')),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='unexpected environment'):
        repair_available(screen, tmp_path / 'state.json', 'a' * 40, 'b' * 40)
