from copy import deepcopy
import json

import pytest
import torch

from scripts.paper_crax_payload.run_queue import prerequisites_ready
from scripts.paper_crax_payload.train_direction_taper_v12 import PLAN, build, validate
from scripts.paper_crax_payload.train_matched_ablation_v12 import PLAN as ORIGINAL


def test_exact_protocol_and_resume_metadata(tmp_path):
    torch.set_num_threads(1)
    config = json.loads(PLAN.read_text()); validate(config); old = json.loads(ORIGINAL.read_text())
    for key in ('task_version','num_envs','development_training_seeds','steps_per_run',
                'shared_learner','geometry','deployment','evaluation'):
        assert config[key] == old[key]
    arm = config['arms'][0]; learner = build(config, arm, 2034, 'cpu')
    learner.set_transitions(175000)
    saved = deepcopy(learner.state_dict())
    restored = build(config, arm, 2034, 'cpu'); restored.load_state_dict(saved)
    assert restored.transitions == 175000
    assert restored.actor_opt.param_groups[0]['lr'] == pytest.approx(.000065)
    saved['direction_taper_v12']['support_start_m'] = -.02
    with pytest.raises(ValueError, match='reinterpret'):
        build(config, arm, 2034, 'cpu').load_state_dict(saved)
    receipt = tmp_path/'receipt.json'
    plan = dict(prerequisite_receipts=[dict(path=str(receipt),expected_status='complete_offline_review')])
    assert not prerequisites_ready(plan)
    receipt.write_text(json.dumps(dict(status='complete_offline_review')))
    assert prerequisites_ready(plan)
    receipt.write_text(json.dumps(dict(status='complete_offline_review',engineering_only=True)))
    with pytest.raises(ValueError, match='Engineering'):
        prerequisites_ready(plan)
