from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.paper_crax_payload.native_p5_adaptation import build_native
from scripts.paper_crax_payload.train_piper_refinement_v12 import attach


def test_geometry_offset_does_not_change_supervision_weights_or_action_gradient():
    torch.set_num_threads(1)
    p=json.loads(Path('configs/paper_crax_payload/simulation_v12_piper_round2.json').read_text())
    native=json.loads(Path('reports/performance_gaps_v11/first_stage_complete/a100/piper_reach-p5/run_config.json').read_text())['configuration']
    l=build_native(native,p['lines']['piper_reach'],'cpu');l.set_transitions(400000)
    o=torch.zeros(3,34);a=torch.randn(3,3,requires_grad=True)
    y=l.margin_geometry(o,a);g=torch.autograd.grad(y.sum(),a)[0]
    online=l.margin_online(o,a).detach().clone();original=deepcopy(l.margin_geometry.state_dict())
    attach(l,p,'piper_reach','R2-activation2mm')
    shifted=l.margin_geometry(o,a);new_g=torch.autograd.grad(shifted.sum(),a)[0]
    assert torch.allclose(shifted,y+.002) and torch.allclose(new_g,g)
    assert torch.equal(l.margin_online(o,a),online)
    assert all(torch.equal(v,l.margin_geometry.state_dict()[k]) for k,v in original.items())
    saved=deepcopy(l.state_dict());other=build_native(native,p['lines']['piper_reach'],'cpu');other.set_transitions(400000)
    attach(other,p,'piper_reach','R2-activation2mm');other.load_state_dict(saved)
    assert torch.equal(other.margin_geometry(o,a),shifted)
    wrong=build_native(native,p['lines']['piper_reach'],'cpu');wrong.set_transitions(400000)
    attach(wrong,p,'piper_reach','R1-quarter75k')
    with pytest.raises(ValueError,match='identical activation'):
        wrong.load_state_dict(saved)
