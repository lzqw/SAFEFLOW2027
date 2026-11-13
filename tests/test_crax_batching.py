"""Few CPU-only batching contracts; no simulator or real training samples."""
from copy import deepcopy

import pytest
import torch

from scripts.paper_crax_payload.crax_models import (
    CumulativeFlowLearner, FlowActor, MarginEnsemble, UpdateGeometry)
from scripts.star.arm_safe_entropy.endpoint_geometry import EndpointGeometry


@pytest.mark.parametrize('mode', ['local', 'endpoint_direct', 'endpoint_pullback'])
def test_batched_head_vjp_matches_original_geometry_and_real_clamps(mode):
    torch.manual_seed(81027)
    actor = FlowActor(5, 2, width=8)
    margin = MarginEnsemble(5, 2, width=8).requires_grad_(False)
    geometry = UpdateGeometry(margin, mode)
    geometry.set_margin_updates(3000)  # A partial ramp must occur exactly once.
    original = EndpointGeometry(margin, mode, transitions=geometry.transitions)
    obs = torch.randn(5, 5)
    midpoint = torch.randn(5, 2).clamp(-1, 1)
    proposal = torch.tensor([[1.4, -1.3], [-1.2, .2], [.3, 1.6], [.1, -.4], [.2, .6]])
    flags = [p.requires_grad for module in [actor, margin] for p in module.parameters()]
    expected = original.query(actor, obs, midpoint, proposal, 1, True)
    actual = geometry.query(actor, obs, midpoint, proposal, 1, True)
    for result, reference in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)
        assert not result.requires_grad
    assert actual[2].keys() == expected[2].keys()
    for key, reference in expected[2].items():
        torch.testing.assert_close(actual[2][key], reference, rtol=2e-5, atol=2e-6)
    assert flags == [p.requires_grad for module in [actor, margin] for p in module.parameters()]
    assert torch.equal(actual[2]['proposal_clipped'], torch.tensor([True, True, True, False, False]))


@pytest.mark.parametrize('action_dim', [2, 6])
def test_target_nine_groups_match_chunks_with_common_noise(action_dim):
    torch.manual_seed(82027)
    actor = FlowActor(5, action_dim, width=8)
    margin = MarginEnsemble(5, action_dim, width=8).requires_grad_(False)
    geometry = UpdateGeometry(margin, 'endpoint_pullback')
    geometry.set_margin_updates(5000)
    batch = 3
    obs = torch.randn(batch, 5)
    expanded = obs.repeat(9, 1)
    # Both moderate and clipping-sized draws; identical independent groups are
    # supplied to the packed and original1B/4B/4B executions.
    noise = torch.randn(actor.num_steps, 9*batch, action_dim) * 3
    with torch.no_grad():
        packed = actor(expanded, geometry=geometry, noise_increments=noise)
        chunks = [actor(expanded[lo:hi], geometry=geometry,
                        noise_increments=noise[:, lo:hi])
                  for lo, hi in [(0, batch), (batch, 5*batch), (5*batch, 9*batch)]]
    for index in [0, 1]:
        torch.testing.assert_close(packed[index], torch.cat([x[index] for x in chunks]),
                                   rtol=3e-5, atol=3e-6)
    for key in packed[2]:
        torch.testing.assert_close(packed[2][key], torch.cat([x[2][key] for x in chunks]),
                                   rtol=3e-5, atol=3e-6)
    with pytest.raises(ValueError):
        actor(expanded, noise_increments=noise[:, :-1])


def test_one_target_actor_call_preserves_second_reward_polyak_target_and_counts():
    torch.manual_seed(83027)
    learner = CumulativeFlowLearner(5, 2, device='cpu', width=8)
    assert 'reward_loss' not in learner.metrics()  # Preserve warmup fix.
    batch_size = 3
    batch = dict(obs=torch.randn(batch_size, 5), action=torch.randn(batch_size, 2).clamp(-1, 1),
        reward=torch.full((batch_size,), 5.), next_obs=torch.randn(batch_size, 5),
        terminated=torch.zeros(batch_size), margin=torch.zeros(batch_size), cost=torch.ones(batch_size))
    actor_batches, qt_before, q_after = [], [], []
    actor_hook = learner.target_actor.register_forward_pre_hook(
        lambda module, inputs: actor_batches.append(inputs[0].shape[0]))
    def flat(module):
        return torch.cat([p.detach().flatten() for p in module.parameters()]).clone()
    qt_hook = learner.qt.register_forward_pre_hook(
        lambda module, inputs: qt_before.append(flat(module)))
    original_step = learner.q_opt.step
    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        q_after.append(flat(learner.q))
        return result
    learner.q_opt.step = step
    try:
        learner.update(batch)
    finally:
        actor_hook.remove()
        qt_hook.remove()
    assert actor_batches == [9*batch_size]
    assert len(qt_before) == len(q_after) == 2
    torch.testing.assert_close(qt_before[1], qt_before[0].lerp(q_after[0], .001))
    assert not torch.equal(qt_before[0], qt_before[1])
    assert learner.updates == learner.margin_updates == 1
    assert learner.reward_updates == 2 and learner.actor_updates == 0
    assert learner.settings['batching_version'] == 'head_vjp_target9_v1'
    legacy = deepcopy(learner.state_dict())
    legacy['settings'].pop('batching_version')
    with pytest.raises(ValueError, match='identical configuration'):
        learner.load_state_dict(legacy)
