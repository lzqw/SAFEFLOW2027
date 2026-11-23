from copy import deepcopy
import json

import pytest

from scripts.paper_crax_payload.evaluate_selected_seed_holdout_20261123 import PLAN,validate_spec


def test_frozen_single_seed_protocol_preserves_fair_modes_and_episode_budget():
    spec=json.loads(PLAN.read_text())
    reach=validate_spec(spec,'Reach');payload=validate_spec(spec,'Payload')
    assert reach['training_seed']==41029 and payload['training_seed']==41027
    assert spec['deployment_mode']=='zero_noise' and spec['episodes_per_condition']==200
    assert spec['independent_test_used'] and not spec['holdout_used_for_selection']
    for task in [reach,payload]:
        assert len(task['cases'])==2 and task['cases'][0]['name'].startswith('SAFEFlow')
        assert task['cases'][1]['name'].startswith('Metric')
        assert len({c['new_training_steps'] for c in task['cases']})==1
    assert reach['layout_seed_start']+200<payload['layout_seed_start']


def test_reject_outcome_dependent_case_or_holdout_changes():
    spec=json.loads(PLAN.read_text())
    for field,value in [('episodes_per_condition',20),('deployment_mode','sampled'),
                         ('holdout_used_for_selection',True)]:
        changed=deepcopy(spec);changed[field]=value
        with pytest.raises(ValueError):validate_spec(changed,'Reach')
