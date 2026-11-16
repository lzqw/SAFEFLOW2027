"""Small CPU-only lineage/merge tests; never run a policy or simulator."""
import json
from pathlib import Path

import pytest

import scripts.paper_crax_payload.close_cvpo_resource_resume as module


def fixture(tmp_path):
    config = dict(development_evaluation_seeds=[61027,61028,61029,61030,61031])
    prefix, suffix = tmp_path / 'prefix', tmp_path / 'suffix'
    prefix.mkdir(); suffix.mkdir()
    original = dict(configuration=config, training_seed=51027, source_commit=module.COMMIT)
    stopped = dict(state='engineering_stop_at_complete_collection_boundary', steps=4401408,
                   official_updates=274064, evaluated_milestones=module.MILESTONES[:2])
    module.write(prefix / 'run_config.json', original)
    module.write(prefix / 'status.json', stopped)
    (prefix / 'interrupted_full.pt').write_bytes(b'synthetic-complete-state')
    module.write(suffix / 'run_config.json', dict(original, resume_parent=module.LOGICAL_PARENT))
    module.write(suffix / 'status.json', dict(state='completed'))
    module.write(suffix / 'receipt.json', dict(status='completed', steps=10000128,
        training_seed=51027, source_commit=module.COMMIT, actual_new_transitions=5598720,
        official_updates=623984, evaluated_milestones=module.MILESTONES,
        evaluation_controls=1250000, independent_test_used=False))
    for label, source in [('prefix', prefix), ('suffix', suffix)]:
        for name in ['progress.jsonl', 'episodes.jsonl']:
            (source / name).write_text(json.dumps(dict(segment=label)) + '\n')
    for step in module.MILESTONES:
        source = prefix if step <= 4000000 else suffix
        policy = source / f'policy_{step}.pt'
        policy.write_bytes(f'synthetic-policy:{step}'.encode())
        module.write(source / f'validation_{step}.json', dict(steps=step, training_seed=51027,
            split='common_validation', independent_test_used=False, policy_sha256=module.sha(policy),
            results=[dict(seed=s, episodes=50, records=[{}] * 50) for s in config['development_evaluation_seeds']]))
    return prefix, suffix, module.sha(prefix / 'interrupted_full.pt')


def test_merge_retains_originals_and_rejects_duplicate_milestone(tmp_path):
    prefix, suffix, parent_sha = fixture(tmp_path)
    combined = module.assemble(prefix, suffix, parent_sha, tmp_path)
    root = Path(combined['stage'])
    assert combined['lineage']['cumulative_transitions'] == 10000128
    assert combined['lineage']['suffix_actual_new_transitions'] == 5598720
    assert len((root / 'progress.jsonl').read_text().splitlines()) == 2
    assert module.sha(root / 'prefix_a100/run_config.json') == module.sha(prefix / 'run_config.json')
    (suffix / 'validation_2000128.json').write_bytes((prefix / 'validation_2000128.json').read_bytes())
    with pytest.raises(ValueError, match='Repeated milestone'):
        module.assemble(prefix, suffix, parent_sha, tmp_path)


def test_admission_preserves_original_prefix_and_exposes_only_complete_seed(tmp_path, monkeypatch):
    prefix, suffix, parent_sha = fixture(tmp_path)
    combined = module.assemble(prefix, suffix, parent_sha, tmp_path)
    stage = tmp_path / '.cvpo-51027-incoming-test'
    Path(combined['stage']).rename(stage)
    archive = tmp_path / 'archived-prefix'
    monkeypatch.setattr(module, 'TARGET', prefix)
    monkeypatch.setattr(module, 'ARCHIVE', archive)
    monkeypatch.setattr(module, 'require_a100_space', lambda size: None)
    result = module.admit(stage, combined['files'])
    assert module.sha(archive / 'interrupted_full.pt') == parent_sha
    assert result['actual_new_transitions'] == 5598720
    assert module.read(prefix / 'receipt.json')['steps'] == 10000128
    assert not stage.exists()
