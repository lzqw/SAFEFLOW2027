import json
from pathlib import Path

import numpy as np
import pytest
import torch

from deployment.piper.simulation import PiperSimConfig
from scripts.paper_crax_payload.piper_helpers import (
    PiperTrainingEnvironment, capture_piper_environment, restore_piper_environment)
from scripts.paper_crax_payload.piper_train import PiperFlowLearner


def test_native_piper_methods_share34d_contract_but_low_changes_true_energy_units():
    config = json.loads(Path('configs/paper_crax_payload/piper_sim_v1.json').read_text())
    torch.set_num_threads(1)
    for method in config['methods']:
        learner = PiperFlowLearner(config, method, 'cpu')
        variance = .01 if method == 'Iso-Low' else .04
        assert learner.actor.variance == learner.target_actor.variance == variance
        assert learner.energy_target == .75/variance
        assert abs(learner.log_alpha.detach().exp().item()-.02*variance) < 1e-8
        learner.set_transitions(40000)
        obs = torch.zeros(4, 34)
        action, energy, _, reference = learner.actor(obs, geometry=learner.geometry, return_reference=True)
        assert action.shape == (4, 3) and torch.isfinite(energy).all()
        torch.testing.assert_close(reference['vn']+2*reference['vt'],
            torch.full_like(reference['vn'], 3*variance))
        batch = dict(obs=obs, next_obs=obs, action=torch.zeros(4, 3), reward=torch.zeros(4),
                     done=torch.zeros(4), cost=torch.zeros(4), margin=torch.zeros(4))
        learner.update(batch)
        state = learner.state_dict()
        restored = PiperFlowLearner(config, method, 'cpu')
        restored.load_state_dict(state)
        assert restored.transitions == 40000 and restored.updates == 1
        if method == 'Iso-Low':
            with pytest.raises(ValueError):
                PiperFlowLearner(config, 'Iso', 'cpu').load_state_dict(state)


@pytest.mark.parametrize('task', ['Reach', 'Payload'])
def test_piper_joint_finger_and_payload_observed_restore_without_physics(task):
    # The official asset cache is external to GitHub; test only when explicitly
    # provided rather than downloading models or guessing another robot.
    import os
    receipt_path = os.environ.get('SAFEFLOW_PIPER_ASSET_RECEIPT')
    if receipt_path is None:
        pytest.skip('Provide pinned official Piper asset receipt for physical state contract')
    assets = json.loads(Path(receipt_path).read_text())
    a, b = [PiperTrainingEnvironment(assets, PiperSimConfig(task=task)) for _ in range(2)]
    try:
        a.reset(seed=62027, options={'fixed_layout': True})
        a.step(np.array([.05, -.05, .05]))
        saved = capture_piper_environment(a)
        before = b.total_physics_substeps
        restored = restore_piper_environment(b, saved)
        assert before == 0 and b.total_physics_substeps == saved['total_physics_substeps']
        np.testing.assert_allclose(restored, saved['observation'], rtol=0., atol=2e-6)
        assert a.elapsed == b.elapsed == 1
        assert a.previous_goal_distance == b.previous_goal_distance
        action = np.array([.05, -.05, .05])
        left, right = a.step(action), b.step(action)
        np.testing.assert_allclose(left[0], right[0], rtol=0., atol=5e-4)
        assert abs(left[1]-right[1]) < 5e-4
        assert left[2:4] == right[2:4]
        assert (a.total_control_intervals, b.total_control_intervals) == (2, 2)
        # Three actual controls were performed; b's restored cumulative1 is not
        # a fourth query and no simulation occurs inside reconstruction.
    finally:
        a.close(); b.close()
