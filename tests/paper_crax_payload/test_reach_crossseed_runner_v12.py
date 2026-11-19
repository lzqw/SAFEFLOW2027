from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.paper_crax_payload.native_p5_adaptation import build_native
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import TEMPLATE,RADIUS_PLAN,bind
from scripts.paper_crax_payload.train_piper_comparison_v12 import attach as original_attach
from scripts.paper_crax_payload.train_reach_crossseed_v12 import attach,evaluate


def plan():
    template=json.loads(TEMPLATE.read_text());radius=json.loads(RADIUS_PLAN.read_text())
    rows=[dict(seed=r['seed'],parent_steps=300000,total_steps=400000,new_training_steps=100000,
        saved_cuda_rng_index=0,downstream_training_started=False,parent_checkpoint_sha256=r['checkpoint_sha256'],
        checkpoint_sha256=str(i)*64) for i,r in enumerate(radius['parents'],1)]
    summary=dict(status='complete_independent_radius200K_summary',completed_runs=2,
        new_completed_training_transitions=200000,training_seeds=[41028,41029],development_episodes=180,
        independent_test_used=False,downstream_training_started=False,
        provenance=[dict(seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],source_commit='10ba914c') for r in rows])
    return bind(template,summary,rows,41028,Path('/data/full_400000.pt'))


@pytest.mark.parametrize('arm',['P5-fixed','P5-activation2mm','Metric-only'])
def test_reuses_exact_existing_weights_optimizers_and_loss_math(arm):
    torch.set_num_threads(1);p=plan();line=p['lines']['piper_reach']
    native=json.loads(Path('configs/paper_crax_payload/piper_sim_v1.json').read_text())
    torch.manual_seed(41028);old=build_native(native,line,'cpu');old.set_transitions(400000)
    original_attach(old,p,'piper_reach',arm)
    old_rng=torch.random.get_rng_state()
    torch.manual_seed(41028);new=build_native(native,line,'cpu');new.set_transitions(400000)
    opt=new.actor_opt;attach(new,p,'piper_reach',arm)
    assert torch.equal(old_rng,torch.random.get_rng_state()) and new.actor_opt is opt
    obs=torch.randn(8,34)
    torch.manual_seed(700);a=old.actor(obs,geometry=old.geometry,return_reference=True)
    torch.manual_seed(700);b=new.actor(obs,geometry=new.geometry,return_reference=True)
    assert all(torch.equal(l,r) for l,r in zip(a[:3],b[:3]))
    left,right=old.actor_regularizer(a[3]),new.actor_regularizer(b[3])
    if isinstance(left,float):assert left==right==0.
    else:torch.testing.assert_close(left,right,rtol=0,atol=0)
    for step in (500000,550000):
        old.set_transitions(step);new.set_transitions(step)
        assert old.actor_opt.param_groups[0]['lr']==new.actor_opt.param_groups[0]['lr']==.00003
    broken=deepcopy(new.state_dict());broken.pop('reach_crossseed_v12')
    with pytest.raises(ValueError,match='Retain'):
        new.load_state_dict(broken)


def test_all_arms_receive_both_modes_and_all_extra_physics_is_counted(monkeypatch):
    import scripts.paper_crax_payload.train_reach_crossseed_v12 as module
    learner=SimpleNamespace(transitions=500000,adaptation=dict(parent_steps=400000),updates=1,
        actor_updates=1,safety_multiplier=0.,actor=torch.nn.Linear(1,1),target_actor=torch.nn.Linear(1,1),
        crossseed_identity=dict(seed=41028))
    learner.action=lambda obs,deterministic=False:deterministic
    calls=[]
    def fake_evaluate(model,config,assets,task,episodes,seed_start):
        calls.append(model.action(None))
        return dict(conditions=[dict(condition='nominal',records=[dict(length=3)])],
            new_training_transitions=0,physical_robot_interactions=0)
    monkeypatch.setattr(module,'evaluate_native',fake_evaluate)
    result=evaluate(learner,{},None,'Reach',100,12020000)
    assert calls==[False,True]
    assert result['new_simulation_controls']==6
    assert [g['deployment_mode'] for g in result['conditions']]==['sampled','zero_noise']
    assert result['independent_test_used'] is False and learner.action(None) is False


def admission_fixture(tmp_path,monkeypatch):
    import scripts.paper_crax_payload.train_reach_crossseed_v12 as module
    from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256
    p=plan();path=tmp_path/'plan.json';path.write_text(json.dumps(p))
    host=module.platform.node();output=tmp_path/'P5-fixed'
    jobs=[dict(seed=seed,arm=arm,host=host if seed==41028 else 'other-test-host',
        output=str(tmp_path/arm) if seed==41028 else '/other/'+arm,visible_devices='0')
        for seed in (41028,41029) for arm in p['arms']]
    admission=dict(status='admitted_fixed_crossseed900K_development',jobs=jobs,
        template_sha256=p['template_sha256'],runner_source_commit='fixture-not-scientific-run',
        runner_sha256=sha256(Path(module.__file__)),new_training_steps=900000,
        independent_test_used=False,scientific_early_stopping=False,automatic_retry=False,
        data_roots_by_host={host:str(tmp_path)},plan_sha256_by_seed={'41028':sha256(path)})
    registration=tmp_path/'admission.json';registration.write_text(json.dumps(admission))
    args=SimpleNamespace(execution_registration=registration,data_root=tmp_path,plan=path,resume=None,
        line='piper_reach',arm='P5-fixed',parent=Path('/data/full_400000.pt'),output=output,
        source_commit='fixture-not-scientific-run',device='cuda:0',cuda_memory_fraction=.025)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=100*1024**3))
    return module,p,args,admission


def test_execution_admission_pins_output_host_and_one_gpu(tmp_path,monkeypatch):
    module,p,args,admission=admission_fixture(tmp_path,monkeypatch)
    module.validate(p,args)  # Identity/resource check only;no model loading or launch.
    args.output=tmp_path/'unregistered-output'
    with pytest.raises(ValueError,match='admission'):
        module.validate(p,args)
    args.output=tmp_path/'P5-fixed';monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(ValueError,match='admission'):
        module.validate(p,args)


def test_execution_admission_reserves_growth_and_same_seed_hardware(tmp_path,monkeypatch):
    module,p,args,admission=admission_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=51*1024**3))
    with pytest.raises(OSError,match='headroom'):
        module.validate(p,args)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=100*1024**3))
    admission['jobs'][1]['host']='other-test-host'
    args.execution_registration.write_text(json.dumps(admission))
    with pytest.raises(ValueError,match='admission'):
        module.validate(p,args)


def test_explicit_storage_budget_reserves_only_unwritten_growth_and_keeps40GiB(tmp_path,monkeypatch):
    module,p,args,admission=admission_fixture(tmp_path,monkeypatch)
    admission.update(storage_policy=module.STORAGE_POLICY,projected_output_gib_per_local_arm=2)
    args.execution_registration.write_text(json.dumps(admission))
    # Three absent outputs reserve the entire6GiB,including pending arms.
    jobs=[j for j in admission['jobs'] if j['host']==module.platform.node()]
    assert module.remaining_output_reserve(tmp_path,jobs)==6*1024**3
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=51*1024**3))
    module.registered_storage_check(tmp_path,args)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=46*1024**3-1))
    with pytest.raises(OSError,match='40GiB'):module.registered_storage_check(tmp_path,args)
    directory=Path(jobs[0]['output']);directory.mkdir();(directory/'full_new_20000.pt').write_bytes(b'x'*4096)
    assert module.remaining_output_reserve(tmp_path,jobs)<6*1024**3
    module.registered_storage_check(tmp_path,args)  # Past writes are not reserved twice.
    reserve=module.remaining_output_reserve(tmp_path,jobs)
    (directory/'latest_full.pt.tmp').write_bytes(b'x'*4096)
    assert module.remaining_output_reserve(tmp_path,jobs)==reserve  # Never credit mutable/atomic writes.
    monkeypatch.setattr(module.shutil,'disk_usage',lambda root:SimpleNamespace(free=39*1024**3))
    with pytest.raises(OSError,match='40GiB'):module.registered_storage_check(tmp_path,args)
    (directory/'external-link').symlink_to(args.plan)
    with pytest.raises(ValueError,match='symlink'):module.remaining_output_reserve(tmp_path,jobs)
