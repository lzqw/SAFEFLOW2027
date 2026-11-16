"""Fixed adapted Reach endpoints: common validation, freeze, independent test.

No checkpoint sweep or extra training. Iso/EA each use their sole400K endpoint
after matched100K radius adaptation. The original selected Iso-Low is retained
as a stagnant reference without matched adaptation, not a fair adaptation arm.
The original15-candidate release is read-only and never overwritten.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from deployment.piper.bundle import FrozenPiperPolicy
from deployment.piper.replay_policy import benchmark,read_observations,replay_bundle
from scripts.paper_crax_payload.piper_reach_release import summarize,reach_schedule
from scripts.paper_crax_payload.piper_train import load_frozen,evaluate_piper,export_command
from scripts.star.arm_safe_entropy.train import atomic_json


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): value.update(block)
    return value.hexdigest()


def freeze_record(method,steps,score):
    status=('risk_qualified' if score['task_progress_observed'] else 'risk_qualified_but_stagnant') if score['risk_qualified'] else 'pareto_tradeoff_not_risk_qualified'
    return dict(robot_family='Piper',task='Reach',method=method,
        selected_checkpoint_steps=steps,selection_split='validation',
        independent_test_used_for_selection=False,risk_qualified=score['risk_qualified'],
        status=status,candidate_count=1,score=score,
        checkpoint_rule='sole fixed400K adaptation endpoint, never intermediate reselection',
        physical_deployment_eligible=False,hardware_calibration_pending=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--base-config',type=Path,required=True)
    parser.add_argument('--assets-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Never overwrite/retry an admitted release')
    old=args.root/'reach-release'
    old_selection=json.loads((old/'selection.json').read_text())
    if json.loads((old/'receipt.json').read_text())['status']!='completed_offline_preparation':
        raise ValueError('Require the preserved completed original release')
    candidates=[]
    for method in ('Iso','EA'):
        run=args.root/'runs'/f'reach-radius-{method.lower()}-100k'
        receipt=json.loads((run/'receipt.json').read_text())
        if (receipt['status']!='completed' or receipt['training_steps']!=400000 or
                receipt['newly_collected_transitions']!=100000 or receipt['method']!=method or
                receipt['geometry_adaptation']['radius_values']!=[.045,.05,.055]):
            raise ValueError('Require the completed matched100K geometry adaptation, not another checkpoint')
        candidates.append(dict(method=method,checkpoint=str(run/'full_400000.pt'),
            config=str(args.root/'configs'/f'reach-radius-{method.lower()}-100k.json'),
            training_receipt=str(run/'training_receipt_400000.json'),steps=400000,
            cohort='matched_radius_adaptation_endpoint',new_adaptation_transitions=100000))
    low=old_selection['methods']['Iso-Low']
    candidates.append(dict(method='Iso-Low',checkpoint=low['candidate']['checkpoint'],
        config=str(args.base_config),training_receipt=low['candidate']['receipt'],
        steps=low['candidate']['steps'],cohort='original_selected_stagnant_reference',
        new_adaptation_transitions=0))
    base=json.loads(args.base_config.read_text())
    plan=base['evaluation']
    if (plan['validation_layout_seed_start']!=61027 or plan['independent_test_layout_seed_start']!=71027
            or plan['validation_episodes_per_condition']!=50 or plan['independent_test_episodes_per_condition']!=100):
        raise ValueError('Preserve registered validation/independent families and episode counts')
    for candidate in candidates:
        path=Path(candidate['checkpoint'])
        if not path.is_file(): raise FileNotFoundError(path)
        candidate['full_checkpoint_sha256']=digest(path)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    atomic_json(args.output/'plan.json',dict(candidates=candidates,
        validation_new_candidates=2,validation_episodes_per_condition=50,
        independent_test_episodes_per_condition=100,conditions=plan['conditions'],
        layout_seed_starts=dict(validation=61027,independent_test=71027),
        original_release_selection_sha256=digest(old/'selection.json'),
        original2250_validation_episodes_not_repeated=True,
        no_training_or_checkpoint_sweep=True,
        geometry_scope='55mm is in adaptation training family; independent layouts, not unseen radius',
        Iso_Low_is_not_matched_adaptation=True,physical_motion_enabled=False))
    rows=[]; controls=0;substeps=0;queries=0;selected={}
    for candidate in candidates:
        method=candidate['method']
        atomic_json(args.output/'status.json',dict(state='validation_and_freeze',method=method,completed_controls=controls))
        common=dict(config=Path(candidate['config']),assets_receipt=args.assets_receipt,
            checkpoint=Path(candidate['checkpoint']),device='cpu')
        config,assets,saved,learner=load_frozen(SimpleNamespace(**common))
        if saved['method']!=method or saved['task']!='Reach' or saved['steps']!=candidate['steps']:
            raise ValueError('Native endpoint identity mismatch')
        if method=='Iso-Low':
            selection=low['selection']
        else:
            evaluation=evaluate_piper(learner,config,assets,'Reach',50,61027)
            evaluation.update(split='validation',method=method,checkpoint_steps=candidate['steps'],
                fixed_endpoint=True,training_seed=41027)
            atomic_json(args.output/f'validation-{method.lower()}.json',evaluation)
            controls+=evaluation['new_simulation_control_intervals']
            substeps+=evaluation['new_physics_substeps'];queries+=evaluation['distance_queries']
            selection=freeze_record(method,candidate['steps'],summarize(evaluation))
            for condition in evaluation['conditions']:
                rows.append(dict(method=method,steps=candidate['steps'],cohort=candidate['cohort'],split='validation',
                    **{k:condition[k] for k in ('condition','episodes','safe_success','violation','time_to_goal_capped_seconds')}))
        selection_path=args.output/f'selection-{method.lower()}.json'
        atomic_json(selection_path,selection)
        export_command(SimpleNamespace(**common,selection_record=selection_path,
            training_receipt=Path(candidate['training_receipt']),output=args.output/'bundles'/method.lower()))
        selected[method]=dict(candidate=candidate,selection=selection,
            bundle_policy_sha256=json.loads((args.output/'bundles'/method.lower()/'manifest.json').read_text())['policy_sha256'])
        del learner,saved
    # Freeze every identity and decision BEFORE observing any independent label.
    freeze=dict(methods=selected,independent_test_used_for_selection=False,
        fixed_endpoint_rule=True,old_original_release_not_modified=True)
    atomic_json(args.output/'selection.json',freeze)
    freeze_sha=digest(args.output/'selection.json')
    for candidate in candidates:
        method=candidate['method']
        atomic_json(args.output/'status.json',dict(state='independent_test',method=method,completed_controls=controls,
            selection_sha256=freeze_sha))
        if digest(Path(candidate['checkpoint']))!=candidate['full_checkpoint_sha256']:
            raise ValueError('Frozen full checkpoint changed before independent test')
        config,assets,saved,learner=load_frozen(SimpleNamespace(config=Path(candidate['config']),
            assets_receipt=args.assets_receipt,checkpoint=Path(candidate['checkpoint']),device='cpu'))
        evaluation=evaluate_piper(learner,config,assets,'Reach',100,71027)
        evaluation.update(split='independent_test',method=method,checkpoint_steps=candidate['steps'],
            training_seed=41027,selection_sha256=freeze_sha,policy_reselected=False)
        atomic_json(args.output/f'independent-{method.lower()}.json',evaluation)
        controls+=evaluation['new_simulation_control_intervals']
        substeps+=evaluation['new_physics_substeps'];queries+=evaluation['distance_queries']
        for condition in evaluation['conditions']:
            rows.append(dict(method=method,steps=candidate['steps'],cohort=candidate['cohort'],split='independent_test',
                **{k:condition[k] for k in ('condition','episodes','safe_success','violation','time_to_goal_capped_seconds')}))
        del learner,saved
    _,observations=read_observations(old/'observations.jsonl')
    timings={};replays={}
    for method in ('Iso','Iso-Low','EA'):
        bundle=args.output/'bundles'/method.lower()
        timings[method]=benchmark(FrozenPiperPolicy(bundle),observations)
        replays[method]=replay_bundle(bundle,old/'observations.jsonl',args.output/f'offline-{method.lower()}.jsonl')
    if digest(args.output/'selection.json')!=freeze_sha:
        raise ValueError('Selection must remain frozen after tests')
    with (args.output/'results.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(args.output/'latency.json',timings)
    atomic_json(args.output/'planned_onsite_reach90.json',reach_schedule())
    atomic_json(args.output/'receipt.json',dict(status='completed_offline_adapted_release',
        new_validation_episodes=300,independent_test_episodes=900,new_training_transitions=0,
        new_simulation_control_intervals=controls,new_physics_substeps=substeps,distance_queries=queries,
        selection_sha256=freeze_sha,selection_frozen_before_independent_test=True,
        offline_replays=replays,physical_robot_interactions=0,physical_motion_enabled=False,
        real_deployment_machine_latency_pending=True,original_release_not_modified=True))
    atomic_json(args.output/'status.json',dict(state='completed_offline_adapted_release',completed_controls=controls))


if __name__=='__main__':
    main()
