import json
from copy import deepcopy

import pytest

from scripts.paper_crax_payload.train_third_seed_confirmation_20261121 import (
    PLAN, TEMPLATE, configuration)


def test_third_seed_preserves_every_scientific_recipe_field():
    spec = json.loads(PLAN.read_text())
    template = json.loads(TEMPLATE.read_text())
    plan = configuration(spec)
    for key in ('arms','candidates','actor_lr','p5','deployment','milestones',
                'development_episodes_per_condition','same_mode_final_deployment_review'):
        assert plan[key] == template[key]
    line = plan['lines']['piper_reach']
    for key in ('conditions','action_sampling_seed','development_layout_seed_start',
                'training_radius_values','parent_rng_cuda_index'):
        assert line[key] == template[key]
    assert line['seed'] == 41030
    assert line['parent_steps'] == 400000


def test_reject_replacement_parent_or_changed_budget():
    spec = json.loads(PLAN.read_text())
    for key, value in [('new_training_steps',300000), ('parent_sha256','0'*64),
                       ('compiled_training_seed',41029)]:
        modified = deepcopy(spec)
        modified[key] = value
        with pytest.raises(ValueError):
            configuration(modified)
