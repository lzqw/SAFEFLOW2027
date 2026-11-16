"""Synthetic tiny files only; no policy training, simulator, or real test seeds."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.paper_crax_payload import close_direction_confirmation as closure


class DummyLearner:
    def load_state_dict(self, state):
        assert state == {'synthetic': True}


@pytest.fixture
def tiny_runs(tmp_path, monkeypatch):
    config = dict(arms=[dict(name=n) for n in ('iso', 'metric-only', 'normal-shrink')],
        training_seeds=[2031, 2032, 2033], evaluation=dict(milestones=[300000]))
    paths = []
    for arm in config['arms']:
        for seed in config['training_seeds']:
            path = tmp_path/f"{arm['name']}-{seed}"; path.mkdir(); paths.append(path)
            manifest = dict(arm=arm, training_seed=seed, configuration=config, format=closure.FORMAT,
                engineering_only=False, target_transitions=300000, source_commit='synthetic',
                initial_network_sha256=str(seed))
            receipt = dict(arm=arm, training_seed=seed, status='completed', engineering_only=False,
                steps=300000, new_training_transitions=300000, source_commit='synthetic',
                evaluations=[300000], independent_endpoint_test_executed=False)
            (path/'run_config.json').write_text(json.dumps(manifest))
            (path/'receipt.json').write_text(json.dumps(receipt))
            torch.save(dict(format=closure.FORMAT, configuration=config, arm=arm, seed=seed,
                steps=300000, source_commit='synthetic', learner={'synthetic': True}), path/'full_300000.pt')
    monkeypatch.setattr(closure, 'build_confirmation', lambda *a: DummyLearner())
    return config, paths


def test_all_nine_fixed_endpoints_required(tiny_runs):
    config, paths = tiny_runs
    assert len(closure.inspect_policies(config, paths)) == 9
    with pytest.raises(ValueError, match='all nine'):
        closure.inspect_policies(config, paths[:-1])
    with pytest.raises(ValueError, match='matching original'):
        closure.inspect_policies(config, paths[:-1]+paths[:1])
    checkpoint_path = paths[-1]/'full_300000.pt'
    checkpoint = torch.load(checkpoint_path, weights_only=False); checkpoint['steps'] = 250000
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='earlier'):
        closure.inspect_policies(config, paths)


def test_seed_byte_commitment_and_known_range_exclusion(tmp_path):
    path = tmp_path/'synthetic-plan.json'
    path.write_text(json.dumps(dict(layout_seed_start=100000000, action_seed=0,
        fresh_layouts_not_previously_evaluated=True)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert closure.open_seed_plan(path, digest)['layout_seed_start'] == 100000000
    with pytest.raises(ValueError, match='byte commitment'):
        closure.open_seed_plan(path, '0'*64)
    path.write_text(json.dumps(dict(layout_seed_start=7100027, action_seed=0,
        fresh_layouts_not_previously_evaluated=True)))
    with pytest.raises(ValueError, match='reserved layout range'):
        closure.open_seed_plan(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_freeze_precedes_seed_opening_and_every_episode_preserved(tiny_runs, tmp_path, monkeypatch):
    config, paths = tiny_runs
    config['independent_closure'] = dict(fresh_layout_seed_commitment_sha256='synthetic')
    # Keep the fixture's model manifests fixed; scientific validation is tested
    # separately. This test exercises order and output preservation only.
    policies = closure.inspect_policies({k: v for k, v in config.items() if k != 'independent_closure'}, paths)
    output = tmp_path/'closure'
    monkeypatch.setattr(closure, 'validate_confirmation', lambda c: None)
    monkeypatch.setattr(closure, 'inspect_policies', lambda c, p: policies)
    def opened(*args):
        assert (output/'all_final_policies_frozen.json').exists()
        assert not (output/'opened_seed_plan.json').exists()
        return dict(layout_seed_start=100000000, action_seed=0)
    monkeypatch.setattr(closure, 'open_seed_plan', opened)
    calls = []
    def evaluate(learner, evaluation_config, milestone):
        assert len(json.loads((output/'all_final_policies_frozen.json').read_text())['policies']) == 9
        assert (output/'opened_seed_plan.json').exists() and milestone == 300000
        assert evaluation_config['evaluation']['independent_episodes_each'] == 100
        calls.append(True)
        return dict(episodes=300, native_controls=300)
    monkeypatch.setitem(sys.modules, 'scripts.paper_crax_payload.train_component_study',
        SimpleNamespace(evaluate=evaluate))
    seed_path = tmp_path/'commitment-bytes'; seed_path.write_text('synthetic')
    closure.close(config, paths, seed_path, output)
    receipt = json.loads((output/'receipt.json').read_text())
    assert len(calls) == 9 and receipt['independent_test_episodes'] == 2700
    assert receipt['unique_layout_seeds'] == 100 and receipt['new_training_transitions'] == 0
    with pytest.raises(FileExistsError, match='no automatic retry'):
        closure.close(config, paths, seed_path, output)
