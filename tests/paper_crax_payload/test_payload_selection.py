from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.paper_crax_payload.payload_select import (
    candidates, digest, select, summarize_validation, verify_test_authorization, validation_manifest)
from scripts.paper_crax_payload.payload_route_qualification import qualify_offset_route


def plan():
    return json.loads(Path('configs/paper_crax_payload/payload_selection_v1.json').read_text())


def evidence(p, candidate, stress_failures=0):
    conditions = []
    for index, condition in enumerate(p['validation']['conditions']):
        records = []
        for episode in range(50):
            violation = index==4 and episode < stress_failures
            records.append(dict(layout_seed=61027+episode, safe_success=not violation,
                violation=violation, true_horizon_terminal=False,
                episode_return=1., time_to_goal_capped_seconds=4. if violation else 1.))
        conditions.append(dict(condition=condition['name'], episodes=50, records=records))
    return dict(status='completed', split='validation', cohort='development', candidate=candidate,
        candidate_id=candidate['candidate_id'], plan_digest=digest(p), policy_sha256=candidate['candidate_id'],
        training_config_digest='same-native-config', frozen_unchanged=True, result=dict(conditions=conditions))


def test_equal_prospective_development_and_confirmation_opportunities():
    p = plan()
    dev, formal = candidates(p, 'development'), candidates(p, 'confirmation')
    assert len(dev)==30 and len(formal)==45
    assert sum(c['fixed_endpoint'] for c in dev)==6
    assert sum(c['fixed_endpoint'] for c in formal)==9
    for method in p['methods']:
        assert sum(c['method']==method for c in dev)==10
        assert sum(c['method']==method for c in formal)==15
    manifest = validation_manifest(p, 'development', 'plan.json', 'train.json', '/runs', '/eval', 'python')
    assert len(manifest['validation_commands'])==30
    assert all('independent_test' not in c['command'] for c in manifest['validation_commands'])


def test_each_stress_risk_gate_rejects_low_mean_hiding_bad_offset():
    p = plan(); d = evidence(p, candidates(p, 'development')[0], stress_failures=10)
    score = summarize_validation(d, p)
    assert score['stress_mean_violation_diagnostic_only']==.05
    assert not score['risk_qualified']
    documents = [evidence(p, c, stress_failures=10) for c in candidates(p, 'development')]
    result = select(p, 'development', documents)
    assert all(r['status']=='pareto_tradeoff_not_risk_qualified' and not r['eligible_for_deployment'] for r in result['methods'].values())


def test_complete_selection_is_frozen_and_return_not_a_tie_priority():
    p=plan(); docs=[evidence(p,c) for c in candidates(p,'development')]
    result=select(p,'development',docs)
    selected=result['methods']['sac_lag']['selected']
    assert selected['candidate']['candidate_id']==sorted(c['candidate_id'] for c in candidates(p,'development') if c['method']=='sac_lag')[0]
    for r in docs[1]['result']['conditions'][0]['records']:
        r['episode_return']=100000.
    again=select(p,'development',docs)
    assert again['methods']['sac_lag']['selected']['candidate']==selected['candidate']
    verify_test_authorization(p,'development',selected['candidate'],selected['policy_sha256'],result)
    with pytest.raises(ValueError):
        verify_test_authorization(p,'development',selected['candidate'],'different-policy',result)
    altered=deepcopy(result);altered['methods']['sac_lag']['risk_qualified']=False
    with pytest.raises(ValueError):
        verify_test_authorization(p,'development',selected['candidate'],selected['policy_sha256'],altered)
    with pytest.raises(ValueError):
        select(p,'development',docs[:-1])
    bad=deepcopy(docs);bad[0]['split']='independent_test'
    with pytest.raises(ValueError):
        select(p,'development',bad)


def test_failed_candidate_retained_but_unexplained_missing_cannot_disappear():
    p=plan();docs=[evidence(p,c) for c in candidates(p,'development')]
    missing=deepcopy(docs[0]);missing.update(status='unavailable_verified_terminal',evidence={})
    docs[0]=missing
    with pytest.raises(ValueError):select(p,'development',docs)
    missing['evidence']=dict(native_process_confirmed_terminal=True,retained_failure_receipt='actual-incident.json')
    result=select(p,'development',docs)
    assert result['candidate_evidence'][missing['candidate_id']]['unavailable']
    assert result['methods']['sac_lag']['allowed_candidate_count']==10
    assert result['methods']['sac_lag']['evaluated_candidate_count']==9


def test_route_claim_requires_truth_substeps_and_actual_blocking_not_offset_name():
    p=plan()['route_qualification'];hashes={'sac_lag':'a','s_iso':'b','s_ea':'c'}
    records=[dict(method=m,layout_seed=s,source_safe_success=True,sector=4,
                 shifted_minimum_distance=.02,native_interval_distance_labels=True,
                 initial_unsafe_after_declared_offset=False) for m in hashes for s in range(62027,62037)]
    d=dict(source_split='route_calibration_validation_only',oracle_used_for_action_selection=False,
        native_interval_geometry_covers_all_physical_substeps=True,replay_mode='real_open_loop_native_physics',
        selection_freeze_digest='frozen',selected_policy_sha256=hashes,
        obstacle_offset=p['fixed_obstacle_offset'],safety_distance_m=.015,nominal_trajectories=records,
        predeclared_alternative_rollouts=[dict(layout_seed=s,safe_success=True,initial_unsafe=False) for s in range(62027,62037)],
        alternative_controller_changed_after_results=False)
    assert not qualify_offset_route(p,d,hashes,'frozen')['alternative_route_claim_verified']
    for r in records:r['shifted_minimum_distance']=.01
    assert qualify_offset_route(p,d,hashes,'frozen')['alternative_route_claim_verified']
    d['oracle_used_for_action_selection']=True
    with pytest.raises(ValueError):qualify_offset_route(p,d,hashes,'frozen')


def test_frozen_fingerprint_catches_parameter_and_dual_changes_without_physics():
    import torch
    from types import SimpleNamespace
    from scripts.paper_crax_payload.payload_frozen import frozen_fingerprint
    actor=torch.nn.Linear(2,3)
    dummy=SimpleNamespace(actor=actor,log_alpha=torch.tensor(-1.),safety_multiplier=0.,
        transitions=10,updates=1,actor_updates=1)
    before=frozen_fingerprint(dummy)
    dummy.safety_multiplier=1.
    assert frozen_fingerprint(dummy)!=before
    dummy.safety_multiplier=0.
    with torch.no_grad():actor.weight.add_(1.)
    assert frozen_fingerprint(dummy)!=before


def test_openloop_uses_executed_actions_and_keeps_offset_after_observed_restore(tmp_path,monkeypatch):
    import sys
    from dataclasses import dataclass,asdict
    from types import SimpleNamespace
    import numpy as np
    from scripts.paper_crax_payload.payload_rollouts import replay_open_loop_witness
    @dataclass
    class Config:
        safety_distance:float=.015
        substeps:int=20
    class FakeEnv:
        def __init__(self):
            self.config=Config();self.obstacle_body=1;self.body=2;self.payload_body=3;self.seen=[]
            self.goal=np.zeros(3)
            self.obstacle_position=np.array([.3,0.,.2]);self.body_position=self.obstacle_position.copy()
            self.observation=np.zeros(45);self.observation[20:23]=self.obstacle_position;self.observation[31]=1.
            self.client=SimpleNamespace(performCollisionDetection=lambda:None,
                getJointState=lambda body,joint:(0.,0.),
                getBaseVelocity=lambda body:([0.,0.,0.],[0.,0.,0.]),
                getBasePositionAndOrientation=lambda body:([0.,0.,0.] if body==3 else self.body_position,[0.,0.,0.,1.]))
            self.sim=SimpleNamespace(set_base_pose=self.set_body)
        def set_body(self,name,position,quaternion):self.body_position=np.array(position)
        def _obs(self):
            value=self.observation.copy();value[20:23]=self.obstacle_position;return value
        def _distance(self):return .1,False
        def step(self,action):
            self.seen.append(action)
            violation=len(self.seen)==2
            return self._obs(),0.,violation,False,dict(minimum_distance=.01 if violation else .05,
                violation=violation,safe_success=False)
    def restore(env,state):
        env.observation=np.array(state['observation']);env.obstacle_position=np.array(state['obstacle'])
        env.body_position=env.obstacle_position.copy()
        return env._obs()
    monkeypatch.setitem(sys.modules,'scripts.paper_crax_payload.payload_environment',
        SimpleNamespace(restore_payload_environment=restore))
    env=FakeEnv()
    header=dict(kind='initial_state',condition=dict(name='nominal'),layout_seed=62027,
        task_configuration=asdict(env.config),initial_state=dict(observation=env._obs().tolist(),
        obstacle=env.obstacle_position.tolist(),joints=list(range(7))+[9,10],positions=[0.]*9,
        velocities=[0.]*9,payload_position=[0.,0.,0.],payload_orientation=[0.,0.,0.,1.],
        payload_linear_velocity=[0.,0.,0.],payload_angular_velocity=[0.,0.,0.],goal=[0.,0.,0.]))
    source=[header,dict(kind='executed_interval',step=1,raw_action=[2,0,0],executed_action=[1,0,0]),
        dict(kind='executed_interval',step=2,raw_action=[-2,0,0],executed_action=[-1,0,0])]
    path=tmp_path/'witness.jsonl';path.write_text('\n'.join(json.dumps(r) for r in source)+'\n')
    result=replay_open_loop_witness(path,env,[.045,0.,.04])
    assert env.seen==[[1,0,0],[-1,0,0]]
    np.testing.assert_allclose(env.body_position,[.345,0.,.24])
    assert result['unchanged_initial_robot_gripper_payload_goal_asserted']
    assert result['obstacle_translation_applied_after_full_restore']
    assert result['shifted_violation'] and result['new_control_intervals']==2
    assert result['new_interval_physics_substeps']==40
    assert result['oracle_used_for_action_selection'] is False
