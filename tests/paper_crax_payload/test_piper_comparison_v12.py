from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.paper_crax_payload.native_p5_adaptation import build_native
from scripts.paper_crax_payload.train_piper_comparison_v12 import attach


def test_metric_is_same_native_weights_but_isotropic_sampling_and_no_direction_loss():
    torch.set_num_threads(1)
    native = json.loads(Path('reports/performance_gaps_v11/first_stage_complete/a100/piper_reach-p5/run_config.json').read_text())['configuration']
    plan = json.loads(Path('configs/paper_crax_payload/simulation_v12_piper_round3.json').read_text())
    learner = build_native(native, plan['lines']['piper_reach'], 'cpu')
    learner.set_transitions(400000)
    actor = deepcopy(learner.actor.state_dict()); opt = learner.actor_opt
    attach(learner, plan, 'piper_reach', 'Metric-only')
    assert learner.actor.sampling_directional is False and learner.actor.energy_directional is True
    assert learner.target_actor.sampling_directional is False
    assert learner.actor_regularizer({}) == 0.
    assert learner.actor_opt is opt
    assert all(torch.equal(v, learner.actor.state_dict()[k]) for k, v in actor.items())
    saved = deepcopy(learner.state_dict())
    other = build_native(native, plan['lines']['piper_reach'], 'cpu'); other.set_transitions(400000)
    attach(other, plan, 'piper_reach', 'Metric-only'); other.load_state_dict(saved)
    wrong = build_native(native, plan['lines']['piper_reach'], 'cpu'); wrong.set_transitions(400000)
    attach(wrong, plan, 'piper_reach', 'R3-activation2mm-quarter75k')
    assert wrong.margin_geometry.activation_offset_m == .002
    assert wrong.warmup.weight(0) == .25 and wrong.warmup.weight(75000) == 1.
    with pytest.raises(ValueError, match='comparison identity'):
        wrong.load_state_dict(saved)
