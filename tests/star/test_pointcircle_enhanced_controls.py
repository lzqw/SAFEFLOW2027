"""Targeted checks for enhanced PointCircle mechanism/ratio controls."""

from __future__ import annotations

from pathlib import Path

import yaml
from scripts.star.decide_pointcircle_geometry_enhancement import ADDENDUM_SHA256
from scripts.star.run_single_gpu_pointcircle_controls import ENHANCED_CONFIGS

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = {
    'Iso-Low': 'single_gpu_pointcircle_iso_low.yaml',
    'Random-Aniso': 'single_gpu_pointcircle_random_aniso.yaml',
    'Safety-Aniso-0.25': 'single_gpu_pointcircle_ratio_025.yaml',
}


def load(name: str) -> dict:
    """Load one PointCircle config."""
    return yaml.safe_load((ROOT / 'configs/star' / name).read_text(encoding='utf-8'))


def test_enhanced_controls_only_inherit_the_locked_geometry_scales() -> None:
    allowed = {
        'safety_protocol_version',
        'safety_protocol_sha256',
        'safety_gradient_scale',
        'safety_ensemble_scale',
    }
    for label, relative in ENHANCED_CONFIGS.items():
        original = load(ORIGINAL[label])['custom_cfgs']['algo_cfgs']
        enhanced = load(relative.name)['custom_cfgs']['algo_cfgs']
        changed = {key for key in original if original[key] != enhanced[key]}
        assert changed == allowed
        assert enhanced['safety_gradient_scale'] == 0.001
        assert enhanced['safety_ensemble_scale'] == 0.10
        assert enhanced['safety_protocol_sha256'] == ADDENDUM_SHA256


def test_ratio_point_differs_from_enhanced_method_only_by_ratio_and_schedule() -> None:
    method = load('single_gpu_pointcircle_p47_geometry_enhanced.yaml')['custom_cfgs']['algo_cfgs']
    ratio = load('single_gpu_pointcircle_ratio_025_geometry_enhanced.yaml')['custom_cfgs'][
        'algo_cfgs'
    ]
    schedule = {'eval_milestones', 'checkpoint_milestones', 'full_checkpoint_freq'}
    scientific_changes = {
        key
        for key in method
        if key not in schedule and method[key] != ratio[key]
    }
    assert scientific_changes == {'flow_normal_variance_ratio'}
    assert ratio['flow_normal_variance_ratio'] == 0.25
