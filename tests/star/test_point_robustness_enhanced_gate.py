"""Enhanced-method gate checks for checkpoint-only robustness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.star.run_point_robustness_queue import wait_for_prerequisite

PROTOCOL_SHA = '0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8'
ENHANCEMENT_SHA = 'b' * 64


def write_prerequisite(tmp_path: Path, enhancement_sha: str) -> Path:
    """Write one completed control-postprocessing prerequisite."""
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(
        json.dumps(
            {
                'status': 'complete',
                'enhancement_preregistration_sha256': enhancement_sha,
            },
        ),
        encoding='utf-8',
    )
    state = tmp_path / 'state.json'
    state.write_text(
        json.dumps(
            {
                'state': 'complete',
                'protocol_sha256': PROTOCOL_SHA,
                'final_manifest': str(manifest),
            },
        ),
        encoding='utf-8',
    )
    return state


def test_robustness_accepts_the_exact_enhanced_control_manifest(tmp_path: Path) -> None:
    state = write_prerequisite(tmp_path, ENHANCEMENT_SHA)
    assert wait_for_prerequisite(state, 1, ENHANCEMENT_SHA) is True


def test_robustness_rejects_another_enhancement(tmp_path: Path) -> None:
    state = write_prerequisite(tmp_path, 'c' * 64)
    with pytest.raises(ValueError, match='does not use the enhanced method'):
        wait_for_prerequisite(state, 1, ENHANCEMENT_SHA)
