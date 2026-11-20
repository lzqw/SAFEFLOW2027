import json
from copy import deepcopy

import pytest

from scripts.paper_crax_payload.train_reach_fresh_parent_v12 import configuration as reference_configuration
from scripts.paper_crax_payload.train_reach_third_parent_20261120 import PLAN, PROJECT, configuration


def test_third_parent_changes_only_the_independent_training_seed():
    plan = json.loads(PLAN.read_text())
    reference = json.loads((PROJECT/plan['reference_plan']).read_text())
    expected = reference_configuration(reference, 41028)
    actual = configuration(plan)
    assert actual['development_seed'] == 41030
    actual['development_seed'] = expected['development_seed']
    assert actual == expected


def test_reject_changed_budget_or_reused_seed():
    plan = json.loads(PLAN.read_text())
    for key, value in [('training_seed', 41028), ('training_steps', 200000)]:
        changed = deepcopy(plan)
        changed[key] = value
        with pytest.raises(ValueError):
            configuration(changed)
