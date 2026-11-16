"""CPU-only compact-package checks; no SSH, Torch, rollouts, or training."""
import json
from pathlib import Path

import pytest

from scripts.paper_crax_payload.admit_cvpo_4080 import (
    MILESTONES, SEED, SOURCE_COMMIT, admit_stage, file_sha, inventory)


def package(root):
    config = json.loads((Path(__file__).resolve().parents[2]
                         / 'configs/paper_crax_payload/cvpo_formal_v4.json').read_text())
    root.mkdir()
    (root / 'run_config.json').write_text(json.dumps(dict(
        configuration=config, training_seed=SEED, source_commit=SOURCE_COMMIT)))
    (root / 'receipt.json').write_text(json.dumps(dict(status='completed', steps=MILESTONES[-1],
        training_seed=SEED, source_commit=SOURCE_COMMIT, evaluated_milestones=MILESTONES)))
    for name in ['progress.jsonl', 'episodes.jsonl', 'status.json']:
        (root / name).write_text('{}\n')
    for step in MILESTONES:
        path = root / f'policy_{step}.pt'
        path.write_bytes(f'fixture-only:{step}'.encode())
        record = dict(training_seed=SEED, steps=step, split='common_validation',
            independent_test_used=False, policy_sha256=file_sha(path), results=[
                dict(seed=seed, episodes=50, records=[{}] * 50)
                for seed in config['development_evaluation_seeds']])
        (root / f'validation_{step}.json').write_text(json.dumps(record))
    return config


def test_inventory_rejects_prefix_and_changed_policy_bytes(tmp_path):
    root = tmp_path / 'seed'
    config = package(root)
    evidence = inventory(root, config)
    assert len(evidence) == 15 and not any('full_' in name for name in evidence)
    path = root / 'receipt.json'
    receipt = json.loads(path.read_text())
    path.write_text(json.dumps(dict(receipt, steps=4000000)))
    with pytest.raises(ValueError, match='prefix'):
        inventory(root, config)
    path.write_text(json.dumps(receipt))
    (root / 'policy_4000000.pt').write_bytes(b'not-original')
    with pytest.raises(ValueError, match='policy bytes'):
        inventory(root, config)


def test_admission_is_atomic_and_preserves_existing_directory(tmp_path, monkeypatch):
    import scripts.paper_crax_payload.admit_cvpo_4080 as module
    stage = tmp_path / '.cvpo-51029-incoming-fixture'
    config = package(stage)
    expected = inventory(stage, config)
    monkeypatch.setattr(module, 'require_data_disk', lambda target, size: None)
    target = tmp_path / 'cvpo-51029-10m'
    receipt = admit_stage(stage, target, config, expected)
    assert not stage.exists() and inventory(target, config) == expected
    assert receipt['training_or_evaluation_controls'] == 0
    second_stage = tmp_path / '.cvpo-51029-incoming-second'
    second_config = package(second_stage)
    with pytest.raises(FileExistsError):
        admit_stage(second_stage, target, second_config, inventory(second_stage, second_config))
    assert second_stage.exists() and inventory(target, config) == expected
