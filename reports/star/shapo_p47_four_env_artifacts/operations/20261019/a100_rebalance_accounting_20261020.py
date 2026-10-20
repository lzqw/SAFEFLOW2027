"""Keep physical tails of the two5080-to-A100 short-Point continuations visible."""
import hashlib
import json
import math
from pathlib import Path


def verify_A100_return_balances(base,completed_only=False):
    base=Path(base);records=[]
    specs=[('point-button-2030',2030,0,'GPU-7b6a829b-7b56-af3e-cf56-1af815c62dbe'),
           ('point-button-2031',2031,1,'GPU-164dfe85-4b4b-0f44-fcd8-99ca6d1e8b78')]
    for job,seed,index,uuid in specs:
        path=base/(job+'_a100_return_balance_20261020_1051_state.json')
        if not path.exists():continue
        state=json.loads(path.read_text());plan=state['plan']
        if state.get('phase')!='completed_first_target_epoch_source_retired':
            if completed_only:continue
            raise ValueError('A100 return transition/failed-attempt accounting unresolved: '+job)
        assert state['job']==plan['job']==job and state['seed']==plan['seed']==seed
        assert state['source_commit']==plan['source_commit']=='62ca7399357c10645a955dcbead7ed2153a82c69'
        assert state['total_budget_steps']==plan['total_budget_steps']==10000000
        assert state['scientific_parameters_changed'] is False and state['other_projects_paused'] is False
        assert state['source_retired'] is True and state['source_paused'] is False
        assert state['source_canonical_path_vacated'] is True
        assert state['full_replay_rng_optimizers_finite_model'] is True
        controller=base/'rebalance_short_points_5080_to_a100_20261020_1051.py'
        assert hashlib.sha256(controller.read_bytes()).hexdigest()==state['controller_sha256']
        assert plan['two_ordinary_frames_qualified'] is True
        earlier=json.loads((base/plan['previous_target_observation']).read_text())
        current=json.loads((base/plan['source_ordinary_observation']).read_text())
        assert current['observed_at_unix']-earlier['observed_at_unix']>=1800
        for observation in (earlier,current):
            gpu=next(g for g in observation['hosts']['tjucta-a100']['gpus'] if g['index']==index)
            assert gpu['uuid']==uuid and not gpu['foreign_pids']
        assert next(g for g in current['hosts']['tjucta-a100']['gpus'] if g['index']==index)['status']=='available_for_rebalance'
        for capacity in (plan['target_capacity_before_pause'],state['target_capacity_at_launch']):
            assert capacity['gpu_uuid']==uuid and capacity['physical_gpu_index']==index
            assert not capacity['foreign_gpu_pids'] and capacity['free_mib']>=16384
            assert capacity['project_free_bytes']>=20*1024**3
            assert capacity['cpu_quota_hierarchy']=='hybrid_v1_cpu'
            assert capacity['cpu_quota_scope']=='observer_tmux_existing_GPU_workers_and_all_ancestors'
            assert capacity['cpu_quota_records']
            quota=capacity['minimum_cpu_quota_cores_or_null']
            assert quota is None or quota>=8
        assert state['target_capacity_at_launch']['copy_complete'] is True
        assert state['runtime_change']['from_torch']=='2.7.1+cu128'
        assert state['runtime_change']['to_torch']=='2.7.1+cu118'
        assert state['runtime_change']['bitwise_equivalence_claimed'] is False
        steps=state['last_logged_source_steps']-state['resume_steps']
        cost=state['last_logged_source_cost']-state['resume_cumulative_training_cost']
        assert 0<=steps<50000 and steps==state['known_repeated_steps_lower_bound']
        assert math.isfinite(cost) and cost>=0 and cost==state['known_extra_training_cost_lower_bound']
        assert state['unlogged_source_tail_unknown_not_zero'] is True
        source,target=state['source_identity'],state['target_identity']
        assert source['boot_id']=='5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f'
        assert target['boot_id']=='6928f333-1036-473c-92d5-4c482d5fef59'
        assert source['device']==target['device']=='cuda:0'
        assert source['cuda_visible_devices']=='0' and target['cuda_visible_devices']==str(index)
        for identity in (source,target):
            args=identity['args']
            for flag,value in {'--seed':str(seed),'--steps':'10000000','--vector-envs':'1',
                               '--batch-size':'256','--replay-size':'400000','--steps-per-epoch':'5000'}.items():
                assert args[args.index(flag)+1]==value
        proof=state['target_GPU_identity_proof']
        assert proof['verified'] is True and proof['pid']==target['pid'] and proof['gpu_uuid']==uuid
        first=state['first_target_epoch']
        assert first['steps']>=state['resume_steps']+5000 and first['cost']>=state['resume_cumulative_training_cost']
        native=json.loads((base/state['first_target_native_observation']).read_text())
        workers=[w for w in native['training'] if w['job']==job];assert len(workers)==1
        worker=workers[0]
        assert worker['pid']==target['pid'] and worker['start_ticks']==target['start_ticks']
        assert worker['boot_id']==target['boot_id'] and worker['steps']>=first['steps']
        assert worker['progress_csv']==first['csv']
        gpu=next(g for g in native['gpus'] if g['uuid']==uuid)
        assert any(p['pid']==target['pid'] for p in gpu['processes'])
        records.append({'job':job,'runtime_placement':'short_Point_5080_to_spare_A100_GPU',
                        'completion_state':state,'first_target_native_observation':native,
                        'known_additional_physical_steps':steps,
                        'known_additional_physical_training_cost':cost,
                        'unknown_unlogged_tail_not_zero':True})
    return records
