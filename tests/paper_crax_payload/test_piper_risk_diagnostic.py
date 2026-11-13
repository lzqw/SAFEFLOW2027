"""Check radius units/input gradients and physical counterfactual accounting."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.paper_crax_payload.piper_helpers import capture_piper_environment
from scripts.paper_crax_payload.piper_risk_diagnostic import RadiusResponse, counterfactual
from scripts.paper_crax_payload.piper_train import make_piper_env


def test_radius_prior_preserves_action_gradient_and_exact_reference():
    class Margin(nn.Module):
        temperature = .02
        def forward(self, obs, action):
            return (2*obs[:, 21]+action[:, 0]).repeat(2, 1)
    obs = torch.zeros(2, 34)
    obs[:, 21] = torch.tensor([.045, .055])
    action = torch.zeros(2, 3, requires_grad=True)
    base = Margin()
    response = RadiusResponse(base)
    expected = torch.tensor([[.09, .10], [.09, .10]])
    torch.testing.assert_close(response(obs, action), expected)
    torch.testing.assert_close(response(obs[:1], action[:1]), base(obs[:1], action[:1]))
    gradient = torch.autograd.grad(response(obs, action).sum(), action)[0]
    torch.testing.assert_close(gradient, torch.tensor([[2., 0., 0.], [2., 0., 0.]]))
    assert obs[1, 21] == pytest.approx(.055)


def test_counterfactual_radius_uses_same_state_and_counts_only_new_substeps():
    receipt = os.environ.get('SAFEFLOW_PIPER_ASSET_RECEIPT')
    if not receipt:
        pytest.skip('Requires pinned official Piper assets')
    assets = json.loads(Path(receipt).read_text())
    config = json.loads(Path('configs/paper_crax_payload/piper_sim_v1.json').read_text())
    original = make_piper_env(config, assets, 'Reach')
    larger = make_piper_env(config, assets, 'Reach', dict(obstacle_radius_m=.055))
    try:
        obs, _ = original.reset(seed=650000)
        original.step(np.zeros(3))
        state = capture_piper_environment(original)
        info, counts = counterfactual(larger, state, np.zeros(3))
        repeat, repeat_counts = counterfactual(larger, state, np.zeros(3))
        assert counts == repeat_counts == (1, 20, 21)
        assert info['local_margin'] == pytest.approx(repeat['local_margin'], abs=2e-6)
        assert state['observation'][21] == pytest.approx(.045)
        assert larger._obs()[21] == pytest.approx(.055)
        np.testing.assert_allclose(larger.goal, original.goal, atol=0, rtol=0)
    finally:
        original.close(); larger.close()
