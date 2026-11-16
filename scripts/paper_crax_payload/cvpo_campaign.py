"""Close the finite CVPO campaign: fixed endpoints, validation freeze, one test.

No replacement of the existing three-method results and no additional RL.
The independent split is opened only after all15 validation candidates are frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean,stdev
import time

from scripts.paper_crax_payload.pointgoal_select import digest
from scripts.paper_crax_payload.train_crax import atomic_json


METRICS=('mean_native_return','mean_episode_cost','mean_goals_per_episode',
         'mean_capped_first_goal_seconds','budget_compliant_fraction')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_registration(config):
    if (config['training_seeds']!=[51027,51028,51029] or config['steps_per_seed']!=10000000
            or config['nominal_total_budget']!=30000000 or config['num_envs']!=256
            or config['milestones']!=[2000000,4000000,6000000,8000000,10000000]
            or config['development_evaluation_seeds']!=[61027,61028,61029,61030,61031]
            or config['evaluation_episodes_per_seed']!=50 or config['cost_limit']!=25
            or config['independent_evaluation_seeds']!=[71027,71028,71029,71030,71031]
            or config['independent_episodes_per_seed']!=100
            or config['scientific_early_stop'] or config['automatic_failed_restart']):
        raise ValueError('Keep the registered three-seed10M, five-candidate, common-split protocol')
    return [(step+255)//256*256 for step in config['milestones']]


def collect_candidates(root,config):
    milestones=validate_registration(config);candidates=[];receipts=[]
    for seed in config['training_seeds']:
        directory=Path(root)/f'cvpo-{seed}-10m'
        manifest=json.loads((directory/'run_config.json').read_text())
        receipt=json.loads((directory/'receipt.json').read_text())
        if (receipt['status']!='completed' or receipt['steps']!=milestones[-1]
                or receipt['training_seed']!=seed or manifest['configuration']!=config
                or manifest['training_seed']!=seed or receipt.get('engineering_only')
                or receipt['evaluated_milestones']!=milestones):
            raise ValueError('Require genuine full-budget native training, not a prefix')
        receipts.append(receipt)
        for step in milestones:
            path=directory/f'validation_{step}.json'
            record=json.loads(path.read_text());results=record['results']
            if (record['steps']!=step or record['training_seed']!=seed
                    or record['split']!='common_validation' or record['independent_test_used'] is not False
                    or [r['seed'] for r in results]!=config['development_evaluation_seeds']
                    or any(r['episodes']!=50 or len(r['records'])!=50 for r in results)):
                raise ValueError('Missing or wrong common validation rows')
            policy=directory/f'policy_{step}.pt';policy_hash=sha256(policy)
            if record['policy_sha256']!=policy_hash:raise ValueError('Validation policy bytes changed')
            candidates.append(dict(method='cvpo',run=directory.name,training_seed=seed,
                checkpoint_steps=step,policy_path=str(policy),policy_sha256=policy_hash,
                validation_sha256=sha256(path),
                **{metric:mean(r[metric] for r in results) for metric in METRICS}))
    return candidates,receipts


def freeze_selection(candidates,source_commit):
    if len(candidates)!=15 or len({(r['training_seed'],r['checkpoint_steps']) for r in candidates})!=15:
        raise ValueError('All three training seeds and five candidates are required')
    eligible=[r for r in candidates if r['mean_episode_cost']<=25]
    front=[r for r in candidates if not any(
        other['mean_episode_cost']<=r['mean_episode_cost'] and
        other['mean_goals_per_episode']>=r['mean_goals_per_episode'] and
        (other['mean_episode_cost']<r['mean_episode_cost'] or
         other['mean_goals_per_episode']>r['mean_goals_per_episode']) for other in candidates)]
    selected=min(eligible or front,key=lambda r:(-r['mean_goals_per_episode'],r['mean_episode_cost'],
        r['mean_capped_first_goal_seconds'],r['run'],r['checkpoint_steps']))
    result=dict(schema='official_cvpo_common_validation_freeze_v4',source_commit=source_commit,
        independent_test_used=False,candidates=candidates,selected=selected,
        status='risk_qualified' if eligible else 'pareto_tradeoff_not_risk_qualified',
        risk_qualified_candidates=len(eligible),common_validation_seeds=[61027,61028,61029,61030,61031],
        episodes_per_validation_seed=50,
        ranking='same_as_existing_PointGoal;cost<=25_then_max_goals,min_cost,min_capped_time',
        independent_evaluation_seeds=[71027,71028,71029,71030,71031],independent_episodes_per_seed=100)
    result['freeze_digest']=digest(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();config=json.loads(args.config.read_text());started=time.monotonic()
    if args.output.exists():raise FileExistsError('Preserve the first freeze and test; no automatic repeat')
    candidates,receipts=collect_candidates(args.run_root,config)
    selection=freeze_selection(candidates,args.source_commit)
    args.output.mkdir(parents=True)
    atomic_json(args.output/'selection.json',selection)  # Written before test env creation.
    fixed=[r for r in candidates if r['checkpoint_steps']==10000128]
    atomic_json(args.output/'fixed_10m.json',dict(training_seed_count=3,by_training_seed=fixed,
        statistics_unit='training_seed;not_evaluation_episode',
        mean_and_sample_SD={key:dict(mean=mean(r[key] for r in fixed),sd=stdev(r[key] for r in fixed))
                            for key in METRICS},receipts=receipts))
    import torch
    from scripts.paper_crax_payload.cvpo_bridge import build_policy,restore_full_policy_state
    from scripts.paper_crax_payload.crax_env import PointGoalTorch
    from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer
    from scripts.paper_crax_payload.train_cvpo import evaluate
    torch.set_num_threads(1);device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(config['torch_cuda_allocator_fraction'],device)
    chosen=selection['selected'];path=Path(chosen['policy_path'])
    if sha256(path)!=chosen['policy_sha256']:raise ValueError('Frozen chosen policy changed')
    state=torch.load(path,map_location=device,weights_only=False)
    if (state['configuration']!=config or state['steps']!=chosen['checkpoint_steps']
            or state['training_seed']!=chosen['training_seed']):raise ValueError('Wrong selected native policy')
    env=PointGoalTorch(100,71027,args.device)
    policy=build_policy(env.obs_dim,env.action_dim,env.native.episode_length,args.device,chosen['training_seed'],config)
    restore_full_policy_state(policy,state['policy'])
    normalizer=ObservationNormalizer(env.obs_dim,device,config['observation_normalization'])
    normalizer.load_state_dict(state['normalizer'])
    results=[]
    for seed in config['independent_evaluation_seeds']:
        result=evaluate(policy,normalizer,env,seed);results.append(result)
        atomic_json(args.output/f'independent_{seed}.json',result)
    aggregate={key:mean(r[key] for r in results) for key in METRICS}
    atomic_json(args.output/'independent.json',dict(selection_freeze_digest=selection['freeze_digest'],
        selected=chosen,episodes=500,per_seed=results,**aggregate,
        independent_test_used_for_selection=False,training_seed_count_for_selected_policy=1,
        fixed_endpoint_summary_is_separate=True))
    atomic_json(args.output/'receipt.json',dict(status='completed',training_transitions=0,
        independent_episodes=500,native_controls=sum(r['native_controls'] for r in results),
        source_commit=args.source_commit,selection_freeze_digest=selection['freeze_digest'],
        elapsed_seconds=time.monotonic()-started,physical_robot_interactions=0,
        existing_three_method_results_not_changed=True,automatic_failed_restart=False))


if __name__=='__main__':main()
