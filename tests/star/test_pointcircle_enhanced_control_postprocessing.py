"""Path checks for enhanced PointCircle control postprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.star.run_pointcircle_control_postprocessing import checkpoint


def test_enhanced_safety_aniso_uses_the_one_time_screen_checkpoint(tmp_path: Path) -> None:
    enhanced = (
        tmp_path
        / 'enhancement/P47-geometry-enhanced-2027/training/STAR-{SafetyPointCircle1-v0}'
        / 'seed-2027-run/checkpoints'
    )
    enhanced.mkdir(parents=True)
    expected = enhanced / 'milestone_300000.pt'
    expected.write_bytes(b'enhanced')
    args = argparse.Namespace(
        star_root=tmp_path / 'original',
        control_root=tmp_path / 'controls',
        safety_aniso_container=tmp_path / 'enhancement/P47-geometry-enhanced-2027',
    )

    assert checkpoint(args, 'Safety-Aniso') == expected
