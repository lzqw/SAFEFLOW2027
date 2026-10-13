"""Storage-only relocation must preserve evidence and same-run membership checks."""
import json
from pathlib import Path

import pytest
from scripts.star.validate_shapo_p47_run import resolve_run_path, validate_checkpoint_resolved_config


@pytest.fixture
def relocated(tmp_path):
    container = tmp_path / 'runs' / 'point-goal-2028'
    checkpoint = container / 'training' / 'segment' / 'checkpoints' / 'latest_full.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'unchanged-evidence')
    origin = Path('/old-host/shapo_p47_four_env/runs/point-goal-2028')
    manifest = {'format': 'shapo-artifact-relocation-v1', 'job': container.name,
                'original_container': str(origin)}
    (container / 'artifact_relocation.json').write_text(json.dumps(manifest))
    return container, origin, checkpoint


def test_relocated_checkpoint_preserves_bytes_and_original_record(relocated):
    container, origin, checkpoint = relocated
    recorded = str(origin / checkpoint.relative_to(container))
    assert resolve_run_path(container, recorded) == checkpoint
    expected = {'algo_cfgs': {'resume_from': '', 'directional_loss_weight': 1.0}}
    observed = {'algo_cfgs': {**expected['algo_cfgs'], 'resume_from': recorded}}
    assert validate_checkpoint_resolved_config('config', observed, expected, {checkpoint}, container) == recorded
    assert checkpoint.read_bytes() == b'unchanged-evidence'
    assert observed['algo_cfgs']['resume_from'] == recorded


def test_relocation_cannot_cross_training_seed(relocated):
    container, origin, _ = relocated
    with pytest.raises(ValueError):
        resolve_run_path(container, str(origin.parent / 'point-goal-2029/training/segment/checkpoints/latest_full.pt'))


def test_relocation_does_not_admit_missing_historical_checkpoints(relocated):
    container, origin, _ = relocated
    with pytest.raises(FileNotFoundError):
        resolve_run_path(container, str(origin / 'training/missing/checkpoints/latest_full.pt'))


def test_relocation_rejects_path_traversal(relocated):
    container, origin, _ = relocated
    with pytest.raises(ValueError):
        resolve_run_path(container, str(origin / 'training/../../other/checkpoints/latest_full.pt'))
