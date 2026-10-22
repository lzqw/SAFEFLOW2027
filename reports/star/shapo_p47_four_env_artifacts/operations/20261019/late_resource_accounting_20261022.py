"""Account for the observed 22 October resource moves, without rerunning science."""
import hashlib
import json
import math
from pathlib import Path

def resource_records(base):
    base=Path(base);records=[];files=set()
    def read(name):
        path=base/name
        value=json.loads(path.read_text())
        files.add(name)
        return value
    storage=base/'a100_backup_replica_storage_relocation_20261023.json'
    if storage.exists():
        s=read(storage.name)
        assert s['phase']=='completed' and s['backup_service_restarted'] is True
        assert s['trainers_signalled']==s['additional_scientific_rollouts']==0
        assert s['other_projects_paused'] is False
        r=s['remote']
        assert r['phase']=='completed_owned_backup_replica_relocated_original_path_preserved'
        assert r['logical_path']=='/home/lzqw/SAFEFLOW2027_A100/recovery_backups/from_4080'
        assert r['physical_destination']=='/storage_0/SAFEFLOW2027_lzqw_uid1007/recovery_backups/from_4080'
        assert r['trainers_signalled']==r['additional_scientific_rollouts']==r['backup_payloads_lost']==0
        assert r['scientific_parameters_changed'] is False and r['foreign_projects_modified'] is False
        assert r['redundant_home_copy_removed'] is True and r['original_retained_at'] is None
        rows=r['source_manifest']
        assert len({row['path'] for row in rows})==len(rows)
        for row in rows:
            p=Path(row['path'])
            assert not p.is_absolute() and '..' not in p.parts
            assert row['kind'] in ('file','directory')
            if row['kind']=='file':assert len(row['sha256'])==64 and row['bytes']>=0
        assert sum(row.get('bytes',0) for row in rows)==r['logical_bytes']
        assert hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()==r['manifest_sha256']
        helper=base/'relocate_a100_backup_replica_20261023.py'
        assert hashlib.sha256(helper.read_bytes()).hexdigest()==s['helper_sha256']
        # Storage-only proof accompanies the ledger without adding scientific
        # interactions or inventing a physical-work recovery record.
    accounting=base/'duplicate_branch_accounting_20261022.json'
    if accounting.exists():
        a=read(accounting.name)
        p=read('duplicate_branch_selection_20261022.json')
        assert a['format']=='safeflow-resource-accounting-v1'
        assert p['phase']=='three_duplicates_retired_and_archived'
        assert p['scientific_parameters_changed'] is False and p['budget_reduced'] is False
        assert p['selected_host']=='4080'
        expected={'ant-2028','walker2d-2028','walker2d-2029'}
        assert {r['job'] for r in a['entries']}==expected and len(a['entries'])==3
        for r in a['entries']:
            source=p['hosts']['tjucta-a100']['jobs'][r['job']]
            target=p['hosts']['4080']['jobs'][r['job']]
            assert source['resume_sha256']==target['resume_sha256']
            assert source['config_sha256']==target['config_sha256']
            assert target['counters']['env_steps']>source['counters']['env_steps']
            assert r['known_additional_physical_steps']==r['retired_source_last_logged_steps']-r['shared_prefix_steps']
            assert r['known_additional_physical_training_cost']==r['retired_source_last_logged_cost']-r['shared_prefix_cumulative_training_cost']
            assert r['retired_source_last_logged_steps']>=source['counters']['env_steps']
            assert r['retired_source_last_logged_cost']>=source['counters']['cumulative_training_cost']
            assert r['unlogged_source_tail_unknown_not_zero'] is True
            records.append(dict(r,kind='excluded_duplicate_branch',selection_file='duplicate_branch_selection_20261022.json'))
        assert a['known_additional_physical_steps']==sum(r['known_additional_physical_steps'] for r in a['entries'])
        assert a['known_additional_physical_training_cost']==sum(r['known_additional_physical_training_cost'] for r in a['entries'])
    for path in sorted(base.glob('a100_to_4080_point*_20261022_state.json')):
        s=read(path.name)
        assert s['phase']=='completed_first_target_epoch_source_retired', 'Unfinished handoff must not be omitted'
        assert s['job'] in {'point-button-2028','point-button-2030'}
        assert s['source_host']=='tjucta-a100' and s['target_host']=='4080'
        assert s['source_commit']=='62ca7399357c10645a955dcbead7ed2153a82c69'
        assert s['scientific_parameters_changed'] is False and s['other_projects_paused'] is False
        assert s['source_retired'] is True and s['source_paused'] is False
        assert s['total_budget_steps']==10000000 and s['unlogged_source_tail_unknown_not_zero'] is True
        assert s['counters']['env_steps']==s['resume_steps']
        assert s['counters']['cumulative_training_cost']==s['resume_cumulative_training_cost']
        args=s['target_identity']['args']
        for flag,value in {'--seed':s['job'].rsplit('-',1)[1],'--steps':'10000000',
                '--device':'cuda:0','--resume':s['target_checkpoint']}.items():
            assert args[args.index(flag)+1]==value
        assert s['first_target_worker']['cuda_execution_backend']=='mps_verified'
        assert s['first_target_worker']['pid']==s['target_pid']==s['target_identity']['pid']
        assert s['first_target_epoch']['steps']>=s['resume_steps']+5000
        steps=s['last_logged_source_steps']-s['resume_steps']
        cost=s['last_logged_source_cost']-s['resume_cumulative_training_cost']
        assert steps==s['known_repeated_steps_lower_bound'] and cost==s['known_extra_training_cost_lower_bound']
        records.append(dict(job=s['job'],kind='full_checkpoint_resource_handoff',proof_file=path.name,
            source_host=s['source_host'],target_host=s['target_host'],resume_steps=s['resume_steps'],
            checkpoint_sha256=s['checkpoint_sha256'],known_additional_physical_steps=steps,
            known_additional_physical_training_cost=cost,unknown_unlogged_tail_not_zero=True))
    for path in sorted(base.glob('short_to_idle_a100_*_20261022_state.json')):
        s=read(path.name)
        assert s['phase']=='completed_first_target_epoch_source_retired', 'Do not omit an unfinished short-cell handoff'
        allowed={'point-goal-2027':('5080',10000000),'walker2d-2028':('4080',5000000)}
        assert s['job'] in allowed
        source,budget=allowed[s['job']]
        assert s['source_host']==source and s['target_host']=='tjucta-a100'
        assert s['total_budget_steps']==budget and s['new_experiment_cells']==0
        assert s['source_commit']=='62ca7399357c10645a955dcbead7ed2153a82c69'
        assert s['scientific_parameters_changed'] is False and s['other_projects_paused'] is False
        assert s['additional_evaluation_rollouts']==0
        assert s['source_retired'] is True and s['source_paused'] is False
        assert s['full_replay_rng_optimizer_present'] is True
        assert s['unlogged_source_tail_unknown_not_zero'] is True
        assert s['counters']['env_steps']==s['resume_steps']
        assert s['counters']['cumulative_training_cost']==s['resume_cumulative_training_cost']
        args=s['target_identity']['args']
        for flag,value in {'--seed':s['job'].rsplit('-',1)[1],'--steps':str(budget),
                '--device':'cuda:0','--resume':s['target_checkpoint'],'--batch-size':'256',
                '--replay-size':'400000','--vector-envs':'1','--steps-per-epoch':'5000'}.items():
            assert args[args.index(flag)+1]==value
        worker=s['first_target_worker']
        assert worker['pid']==s['target_pid']==s['target_identity']['pid']
        assert worker['cuda_execution_backend']=='mps_verified'
        assert worker['mps_gpu_uuid_verified']==s['target_capacity']['uuid']
        assert s['first_target_epoch']['steps']>=s['resume_steps']+5000
        assert s['archive_resume_input'].startswith('/home/carla/4T/SAFEFLOW2027_4080/')
        steps=s['last_logged_source_steps']-s['resume_steps']
        cost=s['last_logged_source_cost']-s['resume_cumulative_training_cost']
        assert 0<=steps<=5000 and steps==s['known_repeated_steps_lower_bound']
        assert cost==s['known_extra_training_cost_lower_bound']
        records.append(dict(job=s['job'],kind='short_cell_full_checkpoint_resource_handoff',proof_file=path.name,
            source_host=source,target_host='tjucta-a100',resume_steps=s['resume_steps'],
            checkpoint_sha256=s['checkpoint_sha256'],known_additional_physical_steps=steps,
            known_additional_physical_training_cost=cost,unknown_unlogged_tail_not_zero=True))
    cold_jobs={
        'cold_point2027_5080_to_a100_20261023_state.json':'point-goal-2027',
        'cold_button2027_gpu1_5080_to_a100_20261023_state.json':'point-button-2027',
        'cold_goal2031_gpu0_5080_to_a100_20261023_state.json':'point-goal-2031',
        'cold_button2031_gpu0_5080_to_a100_20261023_state.json':'point-button-2031',
    }
    for filename,job in cold_jobs.items():
        cold=base/filename
        if not cold.exists():continue
        s=read(cold.name)
        assert s['phase']=='completed_first_new_epoch_on_A100_source_cold_archive_preserved'
        assert s['job']==job and s['total_budget_steps']==10000000
        assert s['source_host']=='5080' and s['target_host']=='tjucta-a100'
        assert s['source_commit']=='62ca7399357c10645a955dcbead7ed2153a82c69'
        assert s['new_experiment_cells']==0 and s['scientific_parameters_changed'] is False
        assert s['source_was_already_stopped'] is True and s['source_archive_preserved'] is True
        assert s['source_signals_sent']==s['additional_evaluation_rollouts']==0
        assert s['training_launched_on_5080'] is False and s['other_projects_paused'] is False
        inventory=read(s['inventory_file'])
        original=next(r for r in inventory['rows'] if r['job']==s['job'])
        assert original['full_replay_rng_optimizers_present'] is True
        assert s['checkpoint_sha256']==original['checkpoint_sha256']
        assert s['counters']==original['counters'] and s['resume_steps']==s['counters']['env_steps']
        args=s['target_identity']['args']
        for flag,value in {'--seed':job.rsplit('-',1)[1],'--steps':'10000000','--device':'cuda:0',
                '--resume':s['target_checkpoint'],'--vector-envs':'1','--batch-size':'256',
                '--replay-size':'400000','--steps-per-epoch':'5000'}.items():
            assert args[args.index(flag)+1]==value
        worker=s['first_target_worker']
        assert worker['pid']==s['target_pid']==s['target_identity']['pid']
        assert worker['cuda_execution_backend']=='mps_verified'
        assert worker['mps_gpu_uuid_verified']==s['target_capacity']['gpu_uuid']
        assert s['first_target_epoch']['steps']>=s['resume_steps']+5000
        assert s['archive_resume_input'].startswith('/home/carla/4T/SAFEFLOW2027_4080/')
        steps=s['last_logged_source_steps']-s['resume_steps']
        cost=s['last_logged_source_cost']-s['resume_cumulative_training_cost']
        assert steps==s['known_repeated_steps_lower_bound']==original['known_repeated_steps_if_resumed']
        assert cost==s['known_extra_training_cost_lower_bound']==original['known_additional_training_cost_if_resumed']
        assert s['unlogged_source_tail_unknown_not_zero'] is True
        records.append(dict(job=s['job'],kind='cold_full_checkpoint_recovery_without_source_GPU_restart',
            proof_file=cold.name,source_host='5080',target_host='tjucta-a100',resume_steps=s['resume_steps'],
            checkpoint_sha256=s['checkpoint_sha256'],known_additional_physical_steps=steps,
            known_additional_physical_training_cost=cost,unknown_unlogged_tail_not_zero=True))
    hold=base/'hold_5080_8h_20261022.json'
    if hold.exists():
        snapshot=read(hold.name)
        assert snapshot['stop_preexisted_the_requested_reservation'] is True
        assert snapshot['resume_after']-snapshot['prepared_at']>=8*3600-0.001
        expected={f'point-goal-{s}' for s in range(2027,2032)}|{'point-button-2027','point-button-2031'}
        assert len(snapshot['lanes'])==7 and {r['lane'] for r in snapshot['lanes']}==expected
        for row in snapshot['lanes']:
            steps=int(float(row['last_logged_steps']))-row['counters']['env_steps']
            cost=row['last_logged_source_cost']-row['counters']['cumulative_training_cost']
            assert steps==row['known_repeated_steps_lower_bound'] and 0<=steps<=50000
            assert cost==row['known_extra_training_cost_lower_bound']
            assert row['unlogged_source_tail_unknown_not_zero'] is True
            records.append(dict(job=row['lane'],kind='preexisting_5080_stop_tail',
                source_host='5080',historical_snapshot_file=hold.name,
                snapshot_is_not_current_liveness_evidence=True,
                stop_preexisted_requested_reservation=True,
                resume_steps=row['counters']['env_steps'],
                known_additional_physical_steps=steps,
                known_additional_physical_training_cost=cost,
                unknown_unlogged_tail_not_zero=True))
    restoration=base/'historical_segment_restoration_20261022.json'
    if restoration.exists():
        restored=read(restoration.name)
        assert restored['phase']=='both_earlier_histories_restored_without_overwrite'
        assert restored['scientific_parameters_changed'] is False
        assert restored['existing_files_overwritten'] is False
        assert restored['new_scientific_evaluations']==0
        assert len(restored['jobs'])==2 and {r['job'] for r in restored['jobs']}=={'point-button-2028','point-button-2030'}
        for row in restored['jobs']:
            assert len(row['shared_checkpoint_sha256'])==64
            assert row['restored_bytes']==sum(r['bytes'] for r in row['restored_files'])
            for item in row['restored_files']:
                p=Path(item['relative_path'])
                assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='training'
                assert len(item['sha256'])==64 and item['bytes']>=0
    ant_restoration=base/'ant2029_historical_restoration_20261022.json'
    if ant_restoration.exists():
        restored=read(ant_restoration.name)
        assert restored['phase']=='ant2029_earlier_history_restored_without_overwrite'
        assert restored['job']=='ant-2029'
        assert restored['source_host']=='4080' and restored['target_host']=='tjucta-a100'
        assert restored['scientific_parameters_changed'] is False
        assert restored['existing_training_files_overwritten'] is False
        assert restored['learner_signals_sent']==restored['new_scientific_evaluations']==0
        assert restored['restored_bytes']==sum(r['bytes'] for r in restored['restored_files'])
        assert len(restored['shared_checkpoint_sha256'])==64
        relocation=restored['artifact_relocation']
        assert relocation['original_host']=='5080' and relocation['transfer_via_host']=='4080'
        assert relocation['target_host']=='tjucta-a100'
    resumed=base/'hold_5080_8h_20261022_resumed.json'
    if resumed.exists():
        dispatch=read(resumed.name)
        proof=read('resume_confirmation_5080_20261022.json')
        assert dispatch['status']=='resumed'
        assert proof['phase']=='seven_resume_workers_and_new_epochs_observed'
        assert proof['dispatched_at']==dispatch['dispatched_at']>=dispatch['resume_after']
        assert dispatch['resume_after']-dispatch['prepared_at']>=8*3600-0.001
        assert proof['budget_per_lane']==10000000 and proof['additional_evaluation_rollouts']==0
        assert proof['scientific_parameters_changed'] is False
        expected={r['lane']:r for r in dispatch['lanes']}
        assert len(expected)==len(proof['confirmed_workers'])==7
        assert {r['job'] for r in proof['confirmed_workers']}==set(expected)
        for row in proof['confirmed_workers']:
            original=expected[row['job']]
            assert row['pid']==original['pid']
            assert row['checkpoint_steps']==original['counters']['env_steps']
            assert row['observed_steps']>=row['checkpoint_steps']+5000
            assert row['cuda_execution_backend']=='mps_verified'
    preserved=base/'resume_inputs_5080_20261022_manifest.json'
    if preserved.exists():
        inputs=read(preserved.name)
        assert inputs['schema']=='safeflow-resume-input-preservation-v1'
        assert inputs['phase']=='seven_named_resume_inputs_preserved_on_separate_4T_filesystem'
        assert inputs['archive_on_separate_4T_filesystem'] is True
        assert inputs['source_payloads_moved_or_deleted'] is False
        assert inputs['scientific_parameters_changed'] is False
        assert inputs['learner_signals_sent']==inputs['additional_evaluation_rollouts']==inputs['model_or_tensor_loads']==0
        assert inputs['historical_2027_early_parent_gaps_repaired'] is False
        original=read('hold_5080_8h_20261022_resumed.json')
        expected={r['lane']:r for r in original['lanes']}
        assert len(inputs['rows'])==7 and {r['job'] for r in inputs['rows']}==set(expected)
        assert inputs['total_bytes']==sum(r['bytes'] for r in inputs['rows'])
        for row in inputs['rows']:
            assert row['source_checkpoint']==expected[row['job']]['checkpoint']
            assert row['counters_from_pre_resume_snapshot']==expected[row['job']]['counters']
            assert len(row['sha256'])==64 and row['bytes']>0
            assert Path(row['archive_path'])==Path(inputs['archive_root'])/row['job']/'latest_full.pt'
    for r in records:
        for key in ('known_additional_physical_steps','known_additional_physical_training_cost'):
            assert isinstance(r[key],(int,float)) and math.isfinite(r[key]) and r[key]>=0
    return records,files
