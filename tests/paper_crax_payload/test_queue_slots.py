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
