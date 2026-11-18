from copy import deepcopy
import json

import pytest
import torch

from scripts.paper_crax_payload.train_matched_ablation_v12 import build, PLAN, validate
from scripts.paper_crax_payload.train_safereach_early_v12 import ScreenLearner
from scripts.paper_crax_payload.direction_warmup_v12 import DirectionWarmup


def test_identical_initialization_and_isolated_factors_with_strict_restore():
    torch.set_num_threads(1)
    config = json.loads(PLAN.read_text()); validate(config); reference = None
    for arm in config['arms']:
        torch.manual_seed(2034); learner = build(config, arm, 2034, 'cpu')
        weights = {name: deepcopy(getattr(learner, name).state_dict()) for name in
                   ('actor', 'target_actor', 'q', 'cost_q', 'margin_online', 'margin_geometry')}
        if reference is None: reference = weights
        else:
            assert all(torch.equal(v, weights[name][k]) for name, values in reference.items() for k, v in values.items())
        learner.set_transitions(175000)
        expected = .000065 if arm['late_actor_lr_decay'] else .0001
        assert learner.actor_opt.param_groups[0]['lr'] == pytest.approx(expected)
        assert learner.current_direction_weight == arm['direction_weight']
        assert learner.development_settings['normal_shrink'] == arm['normal_shrink']
        if arm['component'] == 'metric_only':
            assert learner.actor.sampling_directional is False and learner.actor.energy_directional is True
        saved = deepcopy(learner.state_dict()); restored = build(config, arm, 2034, 'cpu')
        restored.load_state_dict(saved)
        assert restored.transitions == 175000
        wrong = deepcopy(saved); wrong['matched_ablation_v12']['direction_weight'] = 3.
        with pytest.raises(ValueError, match='identical matched'):
            build(config, arm, 2034, 'cpu').load_state_dict(wrong)
    torch.manual_seed(2034)
    old = ScreenLearner(DirectionWarmup(0), device='cpu')
    torch.manual_seed(2034)
    full = build(config, config['arms'][0], 2034, 'cpu')
    for step in (0, 100000, 150000, 175000, 200000, 250000):
        old.set_transitions(step); full.set_transitions(step)
        assert old.actor_opt.param_groups[0]['lr'] == full.actor_opt.param_groups[0]['lr']
    assert old.actor.__class__ is full.actor.__class__
