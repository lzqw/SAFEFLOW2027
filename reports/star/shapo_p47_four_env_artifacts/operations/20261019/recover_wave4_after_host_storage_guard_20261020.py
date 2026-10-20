#!/usr/bin/env python3
"""Explicit continuation after pre-launch storage abort; never replay failed main.

Original458e implementation stays immutable. Only its transaction namespace is
changed, transparently recorded here; same jobs/config/50k-input/5k-proof guards.
"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

from host_full_input_cache_20261020 import ensure_cache

BASE=Path(__file__).resolve().parent
ORIGINAL=BASE/'handoff_point2031_MPS_wave4_5080_20261020.py'
ORIGINAL_SHA='458e802571d17d47acb70586ce9ed4e82334e08381a756497ce7402f5b29de84'
SIDE='mps_wave4_recovery_5080_20261020'
STATE=BASE/'point2031_MPS_wave4_recovery_5080_20261020_controller_state.json'
RECEIPT=BASE/'mps_wave4_prelaunch_storage_abort_resolution_20261020.json'


def load_module(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    assert not STATE.exists() and not RECEIPT.exists(),'Inspect existing continuation; never duplicate it'
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()==ORIGINAL_SHA
    ensure_cache(BASE)
    m=load_module(ORIGINAL,'immutable_wave4_transaction_implementation')
    for file,fn in [('mps_recovery_accounting_20261020.py','verify_existing_MPS_transitions'),
                    ('mps_wave2_recovery_accounting_20261020.py','verify_wave2_MPS_transitions'),
                    ('mps_wave3_recovery_accounting_20261020.py','verify_wave3_MPS_transitions')]:
        assert len(getattr(load_module(BASE/file,'prior_'+fn),fn)(BASE))==2
    failed_path=BASE/'point-button-2031_mps_wave4_5080_20261020_state.json'
    failed=json.loads(failed_path.read_text())
    assert failed['phase']=='needs_inspection_after_MPS_handoff_error'
    assert failed['target_launch_requested'] is False and failed['target_absence_confirmed'] is True
    assert failed['source_paused'] is False and failed['original_source_resumed_in_memory'] is True
    readiness=json.loads((BASE/'mps_point2031_wave4_5080_20261020_preparation.json').read_text())
    expected={w['job']:w for w in readiness['workers_at_ordinary_observation']}
    assert set(expected)=={'point-goal-2031','point-button-2031'}
    source_proofs=[]
    for job in ('point-button-2031','point-goal-2031'):
        h=m.helpers(job);worker,status=m.identity(h,expected[job])
        assert len(status['workers'])==1 and worker['state'] not in ('T','t')
        source_proofs.append({'job':job,'source_identity':worker,'progress':status['progress']})
    helper_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt={'kind':'resolved_prelaunch_storage_abort_no_new_target_or_discarded_rollouts',
        'failed_original_state':failed_path.name,
        'failed_original_state_sha256':hashlib.sha256(failed_path.read_bytes()).hexdigest(),
        'original_transaction_implementation_sha256':ORIGINAL_SHA,
        'recovery_helper':Path(__file__).name,'recovery_helper_sha256':helper_sha,
        'transaction_namespace_override':SIDE,'source_continuity_proofs':source_proofs,
        'prelaunch_failed_attempt_additional_physical_steps':0,
        'prelaunch_failed_attempt_additional_training_cost':0,
        'no_original_state_log_or_input_deleted':True,
        'host_storage_offload_receipt':'host_full_input_cache_20261020_offload_receipt.json',
        'scientific_parameters_changed':False,'new_experiment_cells':0}
    m.save(RECEIPT,receipt)
    # Original failed state/log remain untouched. Every new transaction has its
    # own immutable input, output, target marker and first-epoch observation.
    m.SIDE=SIDE;m.STATE=STATE
    h=m.helpers('point-button-2031');attached=m.capacity(h)
    controller={'phase':'waiting_for_fresh_recovery_full_checkpoints',
        'private_daemon_started':True,'existing_formal_server_attached_not_created':True,
        'existing_formal_server_proofs':attached['existing_formal_MPS_client_proofs'],
        'completed_jobs':[],'recovery_helper_sha256':helper_sha,
        'abort_resolution_receipt':RECEIPT.name,
        'scientific_parameters_changed':False,'other_projects_modified':False}
    m.save(STATE,controller)
    # Button's earlier snapshot expires first; ordering is engineering freshness,
    # not evaluation performance or new scientific experiment selection.
    pending=[expected['point-button-2031'],expected['point-goal-2031']]
    while pending:
        for worker in pending[:]:
            if m.transition(m.helpers(worker['job']),worker,controller):
                pending.remove(worker);controller['completed_jobs'].append(worker['job'])
                m.save(STATE,controller)
        if pending:time.sleep(30)
    controller['phase']='two_existing_Point2031_cells_running_MPS_after_storage_recovery'
    m.save(STATE,controller)


if __name__=='__main__':
    main()
