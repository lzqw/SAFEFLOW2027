from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.paper_crax_payload.direction_development import outward_loss
from scripts.paper_crax_payload.endpoint_residual_v12 import endpoint_residual
from scripts.paper_crax_payload.train_late_endpoint_v12 import PLAN,PROJECT,attach,build,initialize_fork,ramp
from scripts.paper_crax_payload.train_matched_ablation_v12 import AblationLearner


def test_continuous_late_schedule_is_zero_for_entire_actual_first100K():
    assert [ramp(t) for t in (0,99999,100000,125000,150000,250000)]==[0.,0.,0.,.5,1.,1.]


def test_factor_adds_only_terminal_residual_with_no_rng_parameters_or_optimizer_changes():
    torch.set_num_threads(1);torch.manual_seed(2034)
    cfg=json.loads(PLAN.read_text());source=json.loads((PROJECT/cfg['source_plan']).read_text())
    spec=source['arms'][0]
    old=AblationLearner(spec,device='cpu');old.set_transitions(100000)
    new=deepcopy(old);opt=new.actor_opt
    state=torch.random.get_rng_state();attach(new,True,cfg['source_checkpoint_sha256'])
    assert torch.equal(state,torch.random.get_rng_state()) and new.actor_opt is opt
    obs=torch.randn(12,25);noise=torch.randn(4,12,3)
    a=old.actor(obs,geometry=old.geometry,return_reference=True,noise_increments=noise)
    b=new.actor(obs,geometry=new.geometry,return_reference=True,noise_increments=noise)
    assert all(torch.equal(l,r) for l,r in zip(a[:3],b[:3]))
    assert torch.equal(old.actor_regularizer(a[3]),new.actor_regularizer(b[3]))
    new.set_transitions(125000)
    with torch.no_grad():
        for head in new.margin_geometry.heads:head[-1].bias.add_(1.)
    _,_,_,trace=new.actor(obs,geometry=new.geometry,return_reference=True,noise_increments=noise)
    base=outward_loss(trace['velocities'],trace['normals'],trace['weights'],.25)
    margin=new.margin_geometry(trace['endpoint_observation'],trace['endpoint_action']).mean(0)
    expected=base+.5*endpoint_residual(margin)
    loss=new.actor_regularizer(trace);torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    loss.backward();assert all(p.grad is None for p in new.margin_geometry.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.actor.parameters())


@pytest.mark.skipif(not Path('/home/carla/4T/SAFEFLOW2027_SIMULATION_V12/runs-samehardware/A-full-P5-2034/full_100000.pt').exists(),reason='Actual4080 native prefix required')
def test_actual_prefix_load_keeps_real_counts_optimizer_and_every_original_node(tmp_path,monkeypatch):
    import scripts.paper_crax_payload.train_late_endpoint_v12 as module
    torch.set_num_threads(1)
    cfg=json.loads(PLAN.read_text());arm=cfg['arms'][1];learner=build(cfg,arm,2034,'cpu')
    output=tmp_path/'new-fork';output.mkdir();monkeypatch.setattr(module,'ROOT',tmp_path)
    args=SimpleNamespace(fork_from=Path(cfg['source_checkpoint']),output=output,seed=2034,device='cuda:0')
    opt=learner.actor_opt
    saved,provenance=initialize_fork(args,cfg,arm,learner)
    assert learner.actor_opt is opt and len(opt.state)>0
    assert saved['steps']==learner.transitions==100000
    assert provenance['prefix_counts']==dict(training_violations=295,safe_successes=2606,safe_timeouts=356)
    assert len((output/'progress.jsonl').read_text().splitlines())==100
    assert len(list(output.glob('evaluation_*.json')))==4
    assert saved['evaluations']==[25000,50000,75000,100000]
    assert saved['elapsed_seconds']==0.
    assert learner.late_endpoint['enabled'] and ramp(learner.transitions)==0.
