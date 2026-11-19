from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import TEMPLATE,RADIUS_PLAN,bind


def fixture():
    template=json.loads(TEMPLATE.read_text());radius=json.loads(RADIUS_PLAN.read_text())
    rows=[dict(seed=r['seed'],parent_steps=300000,total_steps=400000,new_training_steps=100000,
        saved_cuda_rng_index=0,downstream_training_started=False,parent_checkpoint_sha256=r['checkpoint_sha256'],
        checkpoint_sha256=str(i)*64) for i,r in enumerate(radius['parents'],1)]
    summary=dict(status='complete_independent_radius200K_summary',completed_runs=2,
        new_completed_training_transitions=200000,training_seeds=[41028,41029],development_episodes=180,
        independent_test_used=False,downstream_training_started=False,
        provenance=[dict(seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],source_commit='10ba914c') for r in rows])
    return template,summary,rows


def test_fixed_pair_three_arms_fair_budget_and_actual_seed_parent_rng0():
    template,summary,rows=fixture()
    for seed in (41028,41029):
        plan=bind(template,summary,rows,seed,Path('/data')/str(seed)/'full_400000.pt')
        assert plan['lines']['piper_reach']['seed']==seed
        assert plan['lines']['piper_reach']['parent_rng_cuda_index']==0
        assert plan['new_training_steps_this_seed']==3*plan['additional_steps_per_run']==450000
        assert plan['milestones'][-2:]==[100000,150000]
        assert plan['same_mode_final_deployment_review']['applies_to_all_arms']
        assert all(v['ramp_steps']==0 for v in plan['candidates'].values())
        assert plan['preparation_only'] and plan['training_launches']==0


@pytest.mark.parametrize('damage',['missing_seed','partial','rng1','wrong_parent','sha_provenance','tuned_recipe'])
def test_rejects_partial_best_seed_legacy_rng_or_unregistered_tuning(damage):
    template,summary,rows=deepcopy(fixture())
    if damage=='missing_seed':rows.pop()
    elif damage=='partial':summary['completed_runs']=1
    elif damage=='rng1':rows[0]['saved_cuda_rng_index']=1
    elif damage=='wrong_parent':rows[0]['parent_checkpoint_sha256']='0'*64
    elif damage=='sha_provenance':rows[0]['checkpoint_sha256']='3'*64
    else:template['candidates']['P5-fixed']['activation_offset_m']=.004
    with pytest.raises(ValueError):
        bind(template,summary,rows,41028,Path('/data/full_400000.pt'))
