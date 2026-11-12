"""Prospective admission metadata only: no GPU imports, physics or training."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.paper_crax_payload.formal_admission import (
    ROOT, admit_run, algorithm_file_hashes, phase_fields, require_resume_phase)


PLAN = ROOT/'configs/paper_crax_payload/pointgoal_formal_plan_v1.json'
PARAMETERS = dict(checkpoint_fractions=[.2,.4,.6,.8,1.], maximum_checkpoints_per_seed=5,
                  pointgoal=dict(env_id='safe_goal_point',difficulty=1,task_mode='cumulative',
                                 episodic_cost_budget=25.,environment_overrides={}),
                  fixture='CPU admission unit test, not a selected scientific configuration')


def frozen_fixture(tmp_path, budget=5000000):
    plan=json.loads(PLAN.read_text())
    plan.update(status='frozen',common_steps_per_seed=budget,algorithm_file_sha256=algorithm_file_hashes(),
        freeze_record=dict(frozen_at_utc='2026-11-12T00:00:00+00:00',
            decision_evidence='unit-test-only fixture; no real formal freeze',
            all_methods_common_budget_confirmed=True))
    for spec in plan['method_configurations'].values():
        spec['runtime_parameters']=deepcopy(PARAMETERS)
    path=tmp_path/'unit-formal-plan.json'
    path.write_text(json.dumps(plan))
    return path,plan


def request(path, **changes):
    args=dict(phase='formal',plan_path=path,method='stable_iso',seed=51027,steps=5000000,
              runtime_parameters=PARAMETERS,source_commit='a'*40)
    args.update(changes)
    return admit_run(**args)


@pytest.mark.parametrize('budget',[5000000,10000000])
def test_full_registered_matrix_and_common_budgets_are_admitted_only_when_frozen(tmp_path,budget):
    path,plan=frozen_fixture(tmp_path,budget)
    for method in ['sac_lag_native','stable_iso','ea']:
        for seed in [51027,51028,51029]:
            record=request(path,method=method,seed=seed,steps=budget)
            assert record['common_steps_per_seed']==budget
            assert record['nominal_checkpoint_steps']==[int(budget*f) for f in [.2,.4,.6,.8,1.]]
    assert json.loads(PLAN.read_text())['status']=='prepared_not_frozen'
    with pytest.raises(ValueError,match='not frozen'):
        request(PLAN)
    assert phase_fields(None)=={}
    # Defaults preserve existing development seeds/configurations and settings.
    assert admit_run(phase='development',plan_path=None,method='local',seed=41027,steps=500000,
                     runtime_parameters={},source_commit='legacy') is None


def test_reject_unregistered_seed_budget_config_grid_and_flags(tmp_path):
    path,plan=frozen_fixture(tmp_path)
    for changes in [dict(seed=41027),dict(method='local'),dict(steps=10000000),
                    dict(runtime_parameters=PARAMETERS|{'normalize_observations':True}),
                    dict(forbidden_flags={'profile_only':True})]:
        with pytest.raises(ValueError):request(path,**changes)
    for field,value in [('training_seeds',[51027]),('checkpoint_fractions',[.5,1.]),
                        ('runs',plan['runs'][:-1]),('algorithm_file_sha256',{})]:
        modified=deepcopy(plan);modified[field]=value;path.write_text(json.dumps(modified))
        with pytest.raises(ValueError):request(path)
    modified=deepcopy(plan)
    modified['method_configurations']['ea']['runtime_parameters']={'different_shared_config':True}
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError,match='shared configuration'):request(path)


def test_formal_resume_requires_identical_plan_without_breaking_old_development(tmp_path):
    path,_=frozen_fixture(tmp_path)
    record=request(path)
    state=dict(phase='formal',formal_admission=record)
    require_resume_phase(record,state)
    # A metadata/docs commit or relocation is not a self-referential code hash.
    require_resume_phase(record|dict(runtime_source_commit='b'*40,plan_path='/other/host/plan.json'),state)
    require_resume_phase(None,{'settings':{'unchanged_legacy_native_settings':True}})
    with pytest.raises(ValueError):require_resume_phase(record,{'settings':{}})
    with pytest.raises(ValueError):require_resume_phase(None,state)
    with pytest.raises(ValueError):require_resume_phase(record|dict(plan_sha256='c'*64),state)


def test_even_matching_frozen_flow_config_cannot_override_native_task_budget(tmp_path):
    path,plan=frozen_fixture(tmp_path)
    for key,value in [('episodic_cost_budget',50.),('environment_overrides',{'hazards':0})]:
        modified=deepcopy(plan)
        params=deepcopy(PARAMETERS);params['pointgoal'][key]=value
        for method in ['stable_iso','ea']:
            modified['method_configurations'][method]['runtime_parameters']=deepcopy(params)
        path.write_text(json.dumps(modified))
        # Both plan and CLI agree on the bad value; task cross-consistency must
        # still reject it, rather than relying only on a config mismatch.
        with pytest.raises(ValueError,match='native task/budget'):
            request(path,runtime_parameters=params)


@pytest.mark.parametrize('module',['train_crax','native_sac_checkpoint'])
def test_prepared_plan_rejected_before_any_gpu_library_import(tmp_path,module):
    output=tmp_path/'must-not-be-created'
    code='''
import importlib.abc,runpy,sys
class BlockGPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','jax','mujoco','crax'):
   raise RuntimeError('GPU_IMPORT_BEFORE_ADMISSION:'+fullname)
sys.meta_path.insert(0,BlockGPU())
sys.argv=ARGS
runpy.run_module(MODULE,run_name='__main__')
'''.replace('ARGS',repr(['formal-admission-test','--phase','formal','--formal-plan',str(PLAN),
    '--steps','5000000','--seed','51027','--source-commit','a'*40,'--output',str(output)]))
    code=code.replace('MODULE',repr('scripts.paper_crax_payload.'+module))
    process=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True,timeout=20)
    assert process.returncode!=0 and 'not frozen' in process.stderr
    assert 'GPU_IMPORT_BEFORE_ADMISSION' not in process.stderr
    assert not output.exists()
