import pytest

from scripts.paper_crax_payload.collect_component_campaign import placements,require_completion,FILES


def test_fixed_placements_never_use5080_or_copy_checkpoints_or_raw_traces():
    items=placements()
    assert len(items)==6 and len({r['arm'] for r in items})==6
    assert {r['host'] for r in items}=={'tjucta-a100','4080'}
    assert {r['arm'] for r in items if r['host']=='4080'}=={'iso','metric-only'}
    assert all(name.endswith(('.json','.jsonl')) for name in FILES)
    assert all(name not in FILES for name in ('latest_full.pt','trajectories.jsonl.gz'))


def test_completion_gate_rejects_wrong_budget_seed_source_and_smoke():
    receipt=dict(status='completed',steps=300000,training_seed=2027,
        source_commit='b31878c0',arm=dict(name='full'),engineering_only=False)
    require_completion(receipt,'full')
    for key,value in [('status','running'),('steps',100000),('training_seed',2028),
                      ('source_commit','another-experiment'),('engineering_only',True)]:
        with pytest.raises(ValueError):require_completion(receipt|{key:value},'full')
    with pytest.raises(ValueError):require_completion(receipt,'iso')
