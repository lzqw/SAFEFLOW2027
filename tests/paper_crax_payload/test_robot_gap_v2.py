"""Transfer isolation, batched-target update ordering, and physical vector parity."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.paper_crax_payload.piper_train import (
    FORMAT, PiperFlowLearner, initialize_actor_transfer)
from scripts.paper_crax_payload.piper_parallel import PiperProcessEnvironment
from scripts.paper_crax_payload.piper_helpers import PiperTrainingEnvironment
from deployment.piper.simulation import PiperSimConfig


def base_config():
    config = json.loads(Path('configs/paper_crax_payload/piper_sim_v1.json').read_text())
    config['shared_learner']['hidden_width'] = 8
    return config


def test_transfer_copies_only_fixed_actor_and_rebuilds_task_learning(tmp_path):
    torch.set_num_threads(1)
    config = base_config()
    parent = PiperFlowLearner(config, 'Iso', 'cpu')
    with torch.no_grad():
        for p in parent.actor.parameters():
            p.fill_(.123)
        for p in parent.q.parameters():
            p.fill_(.789)
    checkpoint = tmp_path/'parent.pt'
    saved = dict(format=FORMAT, robot_family='Piper', task='Reach', method='Iso',
        engineering=False, model_commit=config['model_commit'], model_source_urdf_sha256='test-model',
        configuration=config, learner=parent.state_dict(), steps=300000, source_commit='pinned-parent')
    torch.save(saved, checkpoint)
    assets = dict(source_sha256='test-model')
    children = []
    for method in ('Iso', 'EA'):
        child = PiperFlowLearner(config, method, 'cpu')
        q_before = deepcopy(child.q.state_dict())
        margin_before = deepcopy(child.margin_online.state_dict())
        args = SimpleNamespace(task_transfer='Reach_to_Payload', task='Payload', init_from=checkpoint)
        provenance = initialize_actor_transfer(child, config, assets, args)
        for name, tensor in child.actor.state_dict().items():
            torch.testing.assert_close(tensor, parent.actor.state_dict()[name])
            torch.testing.assert_close(tensor, child.target_actor.state_dict()[name])
        for name, tensor in child.q.state_dict().items():
            torch.testing.assert_close(tensor, q_before[name])
        for name, tensor in child.margin_online.state_dict().items():
            torch.testing.assert_close(tensor, margin_before[name])
        assert child.updates == child.transitions == child.actor_updates == 0
        assert not child.actor_opt.state and not child.q_opt.state and not child.margin_opt.state
        assert provenance['parent_training_steps'] == 300000
        children.append(provenance)
    assert children[0]['parent_sha256'] == children[1]['parent_sha256']
    args.task = 'Reach'
    with pytest.raises(ValueError, match='declared'):
        initialize_actor_transfer(child, config, assets, args)


def test_packed_piper_targets_keep_online_cost_actor_and_two_polyak_rounds():
    torch.set_num_threads(1)
    config = base_config()
    config['shared_learner'].update(batched_targets=True, batched_geometry=True)
    learner = PiperFlowLearner(config, 'EA', 'cpu')
    learner.set_transitions(40000)
    count = 3
    batch = dict(obs=torch.randn(count, 34), next_obs=torch.randn(count, 34),
        action=torch.zeros(count, 3), reward=torch.ones(count), cost=torch.zeros(count),
        done=torch.zeros(count), margin=torch.zeros(count))
    target_batches, online_batches, qt_before, q_after = [], [], [], []
    def flat(module):
        return torch.cat([p.detach().flatten() for p in module.parameters()]).clone()
    hooks = [learner.target_actor.register_forward_pre_hook(
                 lambda m, x: target_batches.append(len(x[0]))),
             learner.actor.register_forward_pre_hook(lambda m, x: online_batches.append(len(x[0]))),
             learner.qt.register_forward_pre_hook(lambda m, x: qt_before.append(flat(m)))]
    original_step = learner.q_opt.step
    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        q_after.append(flat(learner.q))
        return result
    learner.q_opt.step = step
    try:
        learner.update(batch)
    finally:
        for hook in hooks:
            hook.remove()
    assert target_batches == [8*count] and online_batches == [count]
    assert learner.updates == 1 and learner.reward_updates == 2
    torch.testing.assert_close(qt_before[1], qt_before[0].lerp(q_after[0], .001))
    restored = PiperFlowLearner(config, 'EA', 'cpu')
    restored.load_state_dict(learner.state_dict())
    assert restored.transitions == 40000 and restored.updates == 1


def test_process_simulator_keeps_same_actions_resets_and_every_substep():
    receipt = os.environ.get('SAFEFLOW_PIPER_ASSET_RECEIPT')
    if not receipt:
        pytest.skip('Pinned official Piper assets required')
    assets = json.loads(Path(receipt).read_text())
    direct = PiperTrainingEnvironment(assets, PiperSimConfig(task='Payload'))
    process = PiperProcessEnvironment(assets, dict(task='Payload'))
    try:
        left, _ = direct.reset(seed=650000)
        right, _ = process.reset(seed=650000)
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-7)
        for _ in range(3):
            action = np.array([.05, -.05, .05])
            process.begin_step(action)
            left = direct.step(action)
            right = process.step(action)
            np.testing.assert_allclose(left[0], right[0], rtol=0, atol=1e-7)
            assert left[1:4] == right[1:4]
            assert direct.total_physics_substeps == process.total_physics_substeps
            assert direct.total_distance_queries == process.total_distance_queries
        state = process.capture()
        np.testing.assert_allclose(state['observation'], right[0], rtol=0, atol=1e-7)
    finally:
        direct.close(); process.close()
