from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.paper_crax_payload.native_p5_adaptation import build_native, validate_parent, actor_digest
from scripts.paper_crax_payload.train_native_p5_adaptation import sha256
from scripts.paper_crax_payload.train_piper_direction_taper_v12 import PLANS, attach, validate


@pytest.mark.parametrize('key', ['reach','payload'])
def test_native_full_parent_sampler_and_resume_identity(key):
    torch.set_num_threads(1)
    plan = json.loads(PLANS[key].read_text()); line_name = 'piper_'+key; line = plan['lines'][line_name]
    parent = torch.load(line['parent_path'],map_location='cpu',weights_only=False)
    validate_parent(parent,plan,line_name,sha256(line['parent_path']))
    values = []
    for arm in plan['arms']:
        validate(plan,SimpleNamespace(line=line_name,arm=arm,resume=None))
        learner = build_native(parent['configuration'],line,'cpu'); learner.load_state_dict(parent['learner'])
        identity = actor_digest(learner.actor)
        attach(learner,plan,line_name,arm)
        assert actor_digest(learner.actor) == identity
        assert learner.safety_multiplier == parent['learner']['safety_multiplier']
        obs = torch.as_tensor(parent['replay']['data']['obs'][:16])
        torch.manual_seed(992900); noise = torch.randn(4,16,3)
        result = learner.actor(obs,geometry=learner.geometry,noise_increments=noise,return_reference=True)
        values.append(tuple(v.detach().clone() for v in result[:3]))
        learner.set_transitions(line['parent_steps']+1000)
        state = deepcopy(learner.state_dict())
        restored = build_native(parent['configuration'],line,'cpu'); restored.load_state_dict(parent['learner'])
        attach(restored,plan,line_name,arm); restored.load_state_dict(state)
        assert restored.transitions == line['parent_steps']+1000
        state['native_piper_direction_taper_v12']['enabled'] = not plan['taper_enabled'][arm]
        with pytest.raises(ValueError,match='same native Piper'):
            restored.load_state_dict(state)
    for left,right in zip(*values):
        assert torch.equal(left,right)
