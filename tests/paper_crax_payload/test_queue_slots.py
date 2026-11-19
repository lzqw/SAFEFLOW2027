import pytest
from scripts.paper_crax_payload.run_queue import next_slot,validate_slot_environments

def test_next_job_uses_released_slot_not_pending_jobs_preassigned_gpu():
    assert next_slot([],2)==0
    assert next_slot([dict(slot_id=0)],2)==1
    assert next_slot([dict(slot_id=1)],2)==0
    with pytest.raises(ValueError):next_slot([dict(slot_id=0),dict(slot_id=1)],2)

def test_optional_slot_binding_preserves_old_plans_and_rejects_conflicts():
    validate_slot_environments(dict(slots=2,jobs=[dict(environment={'CUDA_VISIBLE_DEVICES':'0'})]))
    plan=dict(slots=2,slot_environments=[{'CUDA_VISIBLE_DEVICES':'0'},{'CUDA_VISIBLE_DEVICES':'1'}],jobs=[{}])
    validate_slot_environments(plan)
    plan['jobs'][0]['environment']={'CUDA_VISIBLE_DEVICES':'0'}
    with pytest.raises(ValueError):validate_slot_environments(plan)
    plan['jobs']=[{}];plan['slot_environments']=[]
    with pytest.raises(ValueError):validate_slot_environments(plan)


def test_third_slot_is_only_for_complete_registered_three_arm_single_seed():
    from scripts.paper_crax_payload.run_queue import validate_shared_slots
    jobs=[dict(module='scripts.paper_crax_payload.train_reach_crossseed_v12',
        arguments=['--arm',arm,'--execution-registration','admission.json','--plan','seed41028.json',
                   '--line','piper_reach'],require_completed_receipt=True,required_steps=550000)
        for arm in ('P5-fixed','P5-activation2mm','Metric-only')]
    plan=dict(slots=3,crossseed_three_arm_single_seed=41028,jobs=jobs)
    validate_shared_slots(plan)
    assert next_slot([dict(slot_id=0),dict(slot_id=1)],3)==2
    plan['slots']=4
    with pytest.raises(ValueError):validate_shared_slots(plan)
    plan['slots']=3;plan['jobs'][2]['arguments'][1]='P5-fixed'
    with pytest.raises(ValueError):validate_shared_slots(plan)
    with pytest.raises(ValueError):validate_shared_slots(dict(slots=3,jobs=[]))
