import json
from copy import deepcopy

import pytest

from scripts.paper_crax_payload.train_payload_fresh_parents_20261120 import (
    PLAN, PROJECT, configuration, required_free_bytes)


def test_independent_parents_preserve_the_native_task_and_learner():
    plan = json.loads(PLAN.read_text())
    base = json.loads((PROJECT/plan['base_config']).read_text())
    for seed in plan['training_seeds']:
        actual = configuration(plan, seed)
        assert actual['development_seed'] == seed
        assert 'training_radius_values' not in actual
        assert actual['evaluation']['development_layout_seed_start'] == 12010000
        assert actual['evaluation']['action_sampling_seed'] == 12011000
        actual['development_seed'] = base['development_seed']
        actual['evaluation']['development_layout_seed_start'] = base['evaluation']['development_layout_seed_start']
        actual['evaluation']['action_sampling_seed'] = base['evaluation']['action_sampling_seed']
        assert actual == base
    changed = deepcopy(plan)
    changed['training_steps_each'] = 100000
    with pytest.raises(ValueError):
        configuration(changed, 42028)
    with pytest.raises(ValueError):
        configuration(plan, 41027)


def test_a100_reserves_all_existing_outputs_and_entire_new_budget(tmp_path):
    plan = json.loads(PLAN.read_text())
    job = plan['jobs'][0]
    path = tmp_path/job['existing_crossseed_registration']
    path.parent.mkdir()
    path.write_text(json.dumps(dict(status='admitted_fixed_crossseed900K_development',
        runner_source_commit='77a517d5', data_roots_by_host={'CTAlab':str(tmp_path)},
        jobs=[dict(host='CTAlab', output=str(tmp_path/f'arm{i}')) for i in range(3)])))
    assert required_free_bytes(plan, job, tmp_path) == 48*1024**3
    assert required_free_bytes(plan, plan['jobs'][1], tmp_path) == 44*1024**3
    value = json.loads(path.read_text())
    value['jobs'].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        required_free_bytes(plan, job, tmp_path)
