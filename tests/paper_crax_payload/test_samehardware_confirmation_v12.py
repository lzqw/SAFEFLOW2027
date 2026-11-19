import json

import pytest
import torch

from scripts.paper_crax_payload.train_component_study import initial_digest
from scripts.paper_crax_payload.train_matched_ablation_v12 import PLAN as OLD_PLAN
from scripts.paper_crax_payload.train_samehardware_confirmation_v12 import PLAN, build, validate


def test_samehardware_controls_preserve_original_learning_and_random_initialization():
    torch.set_num_threads(1)
    config = json.loads(PLAN.read_text()); validate(config)
    old = json.loads(OLD_PLAN.read_text())
    for key in ('task_version', 'num_envs', 'steps_per_run', 'development_training_seeds',
                'shared_learner', 'geometry', 'deployment', 'evaluation'):
        assert config[key] == old[key]
    assert config['arms'][0] == old['arms'][0]
    for arm in config['arms']:
        torch.manual_seed(2034)
        learner = build(config, arm, 2034, 'cpu')
        assert initial_digest(learner) == '6eed16a4650593305c7ecb157ffe6d61c990b7c2fa1dad307a1a8025b6f31e6a'
        for step, lr in ((0, .0001), (175000, .000065), (250000, .00003)):
            learner.set_transitions(step)
            assert learner.actor_opt.param_groups[0]['lr'] == pytest.approx(lr)
        if arm['component'] == 'iso':
            assert not learner.actor.sampling_directional
            assert not learner.actor.energy_directional
            assert not learner.development_settings['normal_shrink']
            assert learner.current_direction_weight == 0
