from copy import deepcopy
import json

import pytest

from scripts.paper_crax_payload.train_reach_fresh_parent_v12 import PLAN, PROJECT, configuration
from scripts.paper_crax_payload.run_queue import receipt_completed


def test_only_seed_and_new_DEV_stream_differ_from_native_recipe():
    plan = json.loads(PLAN.read_text()); original = json.loads((PROJECT/plan['base_config']).read_text())
    for seed in (41028,41029):
        config = configuration(plan,seed)
        assert config['development_seed']==seed
        assert config['num_envs']==2 and config['shared_learner']['batch_size']==256
        left,right = deepcopy(config),deepcopy(original)
        for value in (left,right):
            value.pop('development_seed')
            value['evaluation'].pop('development_layout_seed_start')
            value['evaluation'].pop('action_sampling_seed')
        assert left==right
    with pytest.raises(ValueError,match='registered'):
        configuration(plan,41027)
    assert receipt_completed(dict(status='completed',training_steps=300000),dict(required_steps=300000,required_steps_field='training_steps'))
    assert not receipt_completed(dict(status='completed',training_steps=299000),dict(required_steps=300000,required_steps_field='training_steps'))
    assert receipt_completed(dict(status='completed',steps=250000),dict(required_steps=250000))
