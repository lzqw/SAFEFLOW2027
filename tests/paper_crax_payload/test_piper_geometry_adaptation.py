"""Explicit same-task fork isolation and native finite-radius reset/recovery."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.paper_crax_payload.piper_helpers import capture_piper_environment, restore_piper_environment
from scripts.paper_crax_payload.piper_parallel import PiperProcessEnvironment
from scripts.paper_crax_payload.piper_train import FORMAT, PiperFlowLearner, initialize_geometry_adaptation, make_piper_env
from scripts.star.arm_safe_entropy.train import Replay


def test_geometry_fork_retains_learning_state_and_rejects_noise_reward_changes(tmp_path):
    torch.set_num_threads(1)
    config = json.loads(Path('configs/paper_crax_payload/piper_sim_v1.json').read_text())
    config['shared_learner'].update(hidden_width=8, replay_capacity=16)
    parent = PiperFlowLearner(config, 'EA', 'cpu')
    parent.set_transitions(300000)
    parent.updates = 123
    replay = Replay(16, 34)
    replay.add(obs=np.zeros(34), next_obs=np.ones(34), action=np.zeros(3), reward=1, cost=0, done=0, margin=-.02)
    checkpoint = tmp_path/'full_300000.pt'
    saved = dict(format=FORMAT, robot_family='Piper', task='Reach', method='EA', training_seed=41027,
        engineering=False, steps=300000, source_commit='native-parent', model_commit=config['model_commit'],
        model_source_urdf_sha256='native-asset', configuration=config,
        learner=parent.state_dict(), replay=replay.state_dict())
    torch.save(saved, checkpoint)
    adapted = deepcopy(config)
    adapted.update(training_radius_values=[.045, .05, .055], parallel_environments=True, num_envs=8)
    adapted['shared_learner'].update(batched_targets=True, batched_geometry=True,
        batch_size=512, updates_per_vector_step=4)
    child = PiperFlowLearner(adapted, 'EA', 'cpu')
    child_replay = Replay(16, 34)
    args = SimpleNamespace(adapt_from=checkpoint, task='Reach', method='EA', seed=41027, steps=400000)
    _, receipt = initialize_geometry_adaptation(child, child_replay, adapted,
        dict(model_commit=config['model_commit'], source_sha256='native-asset'), args)
    assert child.transitions == 300000 and child.updates == 123 and child_replay.count == 1
    torch.testing.assert_close(next(child.margin_geometry.parameters()), next(parent.margin_geometry.parameters()))
    assert receipt['retained'][-1] == 'learner_counters'
    changed = deepcopy(adapted); changed['shared_learner']['rho'] = .5
    with pytest.raises(ValueError, match='scientific setting'):
        initialize_geometry_adaptation(child, child_replay, changed,
            dict(model_commit=config['model_commit'], source_sha256='native-asset'), args)


def test_finite_radius_reset_process_and_snapshot_preserve_physics():
    receipt = os.environ.get('SAFEFLOW_PIPER_ASSET_RECEIPT')
    if not receipt:
        pytest.skip('Pinned official Piper assets required')
    assets = json.loads(Path(receipt).read_text())
    config = json.loads(Path('configs/paper_crax_payload/piper_sim_v1.json').read_text())
    config['training_radius_values'] = [.045, .05, .055]
    direct = make_piper_env(config, assets, 'Reach', training=True)
    process = PiperProcessEnvironment(assets, dict(task='Reach', training_radius_values=config['training_radius_values']))
    evaluation = make_piper_env(config, assets, 'Reach')
    try:
        sequence = []
        for index in range(3):
            opts = dict(seed=41027) if index == 0 else {}
            left, _ = direct.reset(**opts); right, _ = process.reset(**opts)
            sequence.append(float(left[21]))
            np.testing.assert_allclose(left, right, rtol=0, atol=1e-7)
            direct.step(np.zeros(3)); process.step(np.zeros(3))
        assert sorted(sequence) == pytest.approx([.045, .05, .055])
        state = capture_piper_environment(direct)
        direct.reset()
        restored = restore_piper_environment(direct, state)
        np.testing.assert_allclose(restored, state['observation'], rtol=0, atol=2e-6)
        assert direct.total_physics_substeps == state['total_physics_substeps']
        assert direct.radius_next_index == state['radius_schedule']['next_index']
        nominal, _ = evaluation.reset(seed=41027)
        assert nominal[21] == pytest.approx(.045)
        assert not hasattr(evaluation, 'radius_next_index')
    finally:
        direct.close(); process.close(); evaluation.close()
