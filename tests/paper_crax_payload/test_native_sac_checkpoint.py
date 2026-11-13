"""Focused host instrumentation/state contracts, no JAX or physics executed."""
import ast
from collections import defaultdict,deque
from copy import deepcopy
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.paper_crax_payload.native_sac_checkpoint import (
    CORE_FUNCTIONS, FullCheckpointManager, instrument_source)


def _official_source():
    path=Path('/home/lzqw/SAFEFLOW2027_delivery/CRAX-official/crax/training/agents/sac_lag/train.py')
    if not path.exists():pytest.skip('Pinned official source checkout is required for this AST identity check')
    return path.read_text()


def test_instrumentation_changes_only_host_schedule_and_snapshots():
    source=_official_source()
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='train')
    transformed=instrument_source(source).body[0]
    original_core={n.name:ast.dump(n,include_attributes=False) for n in original.body
                   if isinstance(n,ast.FunctionDef) and n.name in CORE_FUNCTIONS}
    transformed_core={n.name:ast.dump(n,include_attributes=False) for n in transformed.body
                      if isinstance(n,ast.FunctionDef) and n.name in CORE_FUNCTIONS}
    assert original_core==transformed_core
    assert len(original_core)==6
    assert ast.dump(original.args)==ast.dump(transformed.args)
    text=ast.unparse(transformed)
    assert 'if not _paper_checkpoint.resume_available:' in text
    assert 'range(_paper_start_epoch, num_evals_after_init)' in text
    assert '_paper_checkpoint.save_epoch(' in text
    assert '_paper_checkpoint.restore(' in text
    compile(ast.fix_missing_locations(ast.Module(body=[transformed],type_ignores=[])),'host-only','exec')
    with pytest.raises(ValueError,match='unpinned'):
        instrument_source(source+'\n# changed upstream\n')


class FakeBackend:
    @staticmethod
    def effects_barrier():pass
    @staticmethod
    def device_get(value):return deepcopy(value)
    @staticmethod
    def device_put(value):return deepcopy(value)


def _toy_states():
    training=SimpleNamespace(normalizer_params=dict(mean=np.array([2.,3.])),policy_params=np.array([.1,.2]),
        policy_optimizer_state=dict(momentum=np.array([.03])),q_params=np.array([4.]),
        q_optimizer_state=dict(momentum=np.array([.04])),qc_params=np.array([50.]),
        qc_optimizer_state=dict(momentum=np.array([.07])),target_qc_params=np.array([49.]),
        target_q_params=np.array([3.]),alpha_params=np.array([-.7]),
        alpha_optimizer_state=dict(momentum=np.array([.01])),lambda_lagr=np.array([.4]),
        gradient_steps=SimpleNamespace(hi=np.array([0]),lo=np.array([32])),
        env_steps=SimpleNamespace(hi=np.array([0]),lo=np.array([384])))
    environment=SimpleNamespace(info=dict(paper_total_cost=np.array([[30.,40.]]),
        paper_total_goals=np.array([[2.,3.]]),paper_reset_keys=np.array([[[1,2],[3,4]]]),
        partial_episode_cost=np.array([[1.,2.]])),qpos=np.array([[[.1,.2],[.3,.4]]]))
    replay=SimpleNamespace(data=np.arange(12).reshape(1,3,4),insert_position=np.array([3]),
        sample_position=np.array([0]),key=np.array([[5,6]],dtype=np.uint32))
    evaluator=SimpleNamespace(_key=np.array([8,9],dtype=np.uint32),_eval_walltime=3.)
    logger=SimpleNamespace(_buffer_size=10,_metrics_buffer=defaultdict(lambda:deque(maxlen=10),
        {'episodic/cost':deque([3.,4.],maxlen=10)}),_num_steps=384,_last_log_steps=256,
        _log_count=2,_episodic_metrics_updated={'episodic/cost'},_sps_last_steps=256,_sps_last_time=1000.)
    return training,environment,replay,np.array([10,11],dtype=np.uint32),evaluator,logger,SimpleNamespace(_size=3)


def test_full_host_checkpoint_restores_all_state_and_rng(tmp_path):
    settings=dict(seed=41,num_envs=8,normalize_observations=False)
    manager=FullCheckpointManager(tmp_path/'first',settings,backend=FakeBackend())
    stride,epochs=manager.epoch_plan(32,16,4096,128,8,1,1)
    assert (stride,epochs)==(32,16)
    assert len(set(manager.plan['candidate_epochs']))==5
    state=_toy_states()
    manager.save_epoch(*state[:4],*state[4:],5.,1,384,{'training/actor_loss':np.array(2.)})
    restored=FullCheckpointManager(tmp_path/'restored',settings,
        resume=tmp_path/'first/full_latest.pkl',backend=FakeBackend())
    restored.epoch_plan(999,999,4096,128,8,1,1)
    new_state=_toy_states()
    new_state[4]._key=np.array([100,100]);new_state[6]._size=0
    values=restored.restore(*new_state)
    np.testing.assert_array_equal(values[0].qc_optimizer_state['momentum'],[.07])
    np.testing.assert_array_equal(values[0].target_qc_params,[49.])
    np.testing.assert_array_equal(values[0].lambda_lagr,[.4])
    np.testing.assert_array_equal(values[1].info['partial_episode_cost'],[[1.,2.]])
    np.testing.assert_array_equal(values[2].key,[[5,6]])
    np.testing.assert_array_equal(values[3],[10,11])
    np.testing.assert_array_equal(new_state[4]._key,[8,9])
    assert new_state[5]._sps_last_time is None
    assert new_state[6]._size==3
    assert values[4:7]==(5.,1,384)
    assert restored.plan['actor_epoch_steps']==32
    assert restored.initial_steps==384
    status=__import__('json').loads((tmp_path/'first/checkpoint_status.json').read_text())
    assert status['training_cumulative_cost']==70
    assert status['new_training_transitions']==384
    with pytest.raises(ValueError,match='identical'):
        FullCheckpointManager(tmp_path/'bad',settings|dict(normalize_observations=True),
            resume=tmp_path/'first/full_latest.pkl',backend=FakeBackend())


def test_policy_only_is_not_resume_and_target_extension_is_explicit(tmp_path):
    path=tmp_path/'old_policy.pkl'
    with path.open('wb') as stream:pickle.dump(({},{}),stream)
    with pytest.raises(ValueError,match='policy-only'):
        FullCheckpointManager(tmp_path/'bad',{},resume=path,backend=FakeBackend())
    manager=FullCheckpointManager(tmp_path/'first',{},backend=FakeBackend())
    manager.epoch_plan(32,16,4096,128,8,1,1)
    manager.save_epoch(*_toy_states(),5.,1,384,{})
    resumed=FullCheckpointManager(tmp_path/'next',{},resume=tmp_path/'first/full_latest.pkl',backend=FakeBackend())
    with pytest.raises(ValueError,match='explicit'):
        resumed.epoch_plan(777,999,8192,128,8,1,1)
    extended=FullCheckpointManager(tmp_path/'extend',{},resume=tmp_path/'first/full_latest.pkl',
        allow_target_extension=True,backend=FakeBackend())
    stride,epochs=extended.epoch_plan(777,999,8192,128,8,1,1)
    assert stride==32
    assert extended.plan['actual_rounded_final_steps']>=8192
    with pytest.raises(ValueError,match='one native device'):
        FullCheckpointManager(tmp_path/'multi',{}).epoch_plan(32,16,4096,128,8,2,1)


def test_prospective_history_preserves_per_epoch_cost_without_settings_changes(tmp_path):
    import json
    settings=dict(seed=41,num_envs=8,normalize_observations=False)
    manager=FullCheckpointManager(tmp_path/'history',settings,backend=FakeBackend(),
                                  invocation_source_commit='a'*40)
    manager.epoch_plan(32,16,4096,128,8,1,1)
    first=_toy_states()
    manager.save_epoch(*first,5.,1,384,{})
    second=_toy_states()
    second[1].info['paper_total_cost']=np.array([[40.,60.]])
    manager.save_epoch(*second,6.,2,640,{})
    history=[json.loads(line) for line in (tmp_path/'history/checkpoint_history.jsonl').read_text().splitlines()]
    assert [r['steps'] for r in history]==[384,640]
    assert [r['training_cumulative_cost'] for r in history]==[70.,100.]
    assert history[1]['cost_per_1000_transitions']==156.25
    assert history[1]['new_training_transitions']==640
    assert history[0]['completed_epoch']==1 and history[1]['completed_epoch']==2
    assert all(r['evaluation_calls']==0 and r['phase']=='development' for r in history)
    with (tmp_path/'history/full_latest.pkl').open('rb') as stream:payload=pickle.load(stream)
    assert payload['settings']==settings and 'phase' not in payload
    restored=FullCheckpointManager(tmp_path/'history-resume',settings,
        resume=tmp_path/'history/full_latest.pkl',backend=FakeBackend())
    assert restored.initial_steps==640 and restored.settings==settings
