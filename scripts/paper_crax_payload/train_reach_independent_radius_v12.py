"""Explicit fixed100K radius coverage of both true independent native parents.

No new method candidate;reuse native strict distribution-fork implementation.
Original parent checkpoints, configs, seeds, results and RNG indices are retained.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.paper_crax_payload.piper_train import FORMAT,train
from scripts.paper_crax_payload.train_reach_fresh_parent_v12 import configuration as parent_configuration
from scripts.paper_crax_payload.train_native_p5_adaptation import sha256,storage_check
from scripts.star.arm_safe_entropy.train import atomic_json


PROJECT=Path(__file__).resolve().parents[2]
PLAN=PROJECT/'configs/paper_crax_payload/simulation_v12_reach_independent_radius.json'
ROOT=Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12')


def validate_parent(saved,receipt,plan,seed,digest):
    if plan!=json.loads(PLAN.read_text()) or seed not in plan['training_seeds']:
        raise ValueError('Both predeclared independent native seeds only;no implicit new branch')
    entry=next(r for r in plan['parents'] if r['seed']==seed)
    parent_plan=json.loads((PROJECT/plan['parent_plan']).read_text())
    expected=parent_configuration(parent_plan,seed)
    if (digest!=entry['checkpoint_sha256'] or saved['format']!=FORMAT
            or saved['robot_family']!='Piper' or saved['task']!='Reach' or saved['method']!='EA'
            or saved['steps']!=300000 or saved['training_seed']!=seed or saved['engineering']
            or saved['source_commit']!='22d4966f' or saved['configuration']!=expected
            or saved.get('geometry_adaptation') is not None or saved.get('task_transfer') is not None
            or receipt['status']!='completed' or receipt['training_steps']!=300000
            or receipt['training_seed']!=seed or receipt['initial_steps']!=0
            or receipt['newly_collected_transitions']!=300000 or receipt['engineering']
            or receipt['training_source_commit']!='22d4966f'
            or saved['replay']['data']['obs'].shape[-1]!=34 or len(saved['rng']['cuda'])!=1):
        raise ValueError('Require this seed actual fixed300K scientific parent,not an old or best-node substitute')
    for field,native in [('training_violation_count','violations'),('training_safe_successes','safe_successes'),
                         ('training_safe_timeouts','safe_timeouts'),('training_episodes','episodes')]:
        if receipt[field]!=saved[native]:
            raise ValueError('Native parent checkpoint counters disagree with complete receipt')
    config=deepcopy(expected)
    config['training_radius_values']=plan['training_radius_values']
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--assets-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    if plan!=json.loads(PLAN.read_text()) or args.seed not in plan['training_seeds']:
        raise ValueError('Use registered fixed200K total radius coverage stage only')
    root=ROOT.resolve(strict=True)
    if not args.output.resolve().is_relative_to(root) or args.output.exists():
        raise ValueError('Distinct new data-disk directory;never overwrite,rename or silently resume')
    if not Path('/storage_0').is_mount() or ROOT.stat().st_dev!=Path('/storage_0').stat().st_dev:
        raise OSError('The admitted data filesystem must be mounted;no home fallback')
    storage_check(ROOT,growth_gib=plan['projected_total_output_gib'])
    device=torch.device(args.device)
    if device.type!='cuda' or torch.cuda.device_count()!=1:
        raise ValueError('Bind exactly one shared GPU;its saved generator index is0')
    torch.cuda.set_per_process_memory_fraction(.025,device)
    entry=next(r for r in plan['parents'] if r['seed']==args.seed)
    checkpoint=ROOT/entry['relative_checkpoint']
    receipt_path=checkpoint.parent/'receipt.json';receipt=json.loads(receipt_path.read_text())
    digest=sha256(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=validate_parent(saved,receipt,plan,args.seed,digest)
    del saved
    directory=ROOT/'independent-radius-configurations';directory.mkdir(exist_ok=True)
    config_path=directory/f'reach-EA-radius-{args.seed}.json'
    if config_path.exists():
        raise FileExistsError('Preserve existing seed/config registration;never automatic retry')
    atomic_json(config_path,config)
    atomic_json(directory/f'reach-EA-radius-{args.seed}-registration.json',dict(plan=plan,
        seed=args.seed,source_commit=args.source_commit,parent_checkpoint_sha256=digest,
        parent_receipt_sha256=sha256(receipt_path),config_sha256=sha256(config_path),
        saved_cuda_rng_index=0,configuration_change='training_radius_values_only',
        fork_rng=plan['fork_rng'],new_training_steps=100000,independent_test_used=False,
        downstream_training_started=False,automatic_retry=False))
    native=SimpleNamespace(config=config_path,assets_receipt=args.assets_receipt,task='Reach',method='EA',
        seed=args.seed,steps=400000,source_commit=args.source_commit,device=args.device,output=args.output,
        engineering=False,resume=None,init_from=None,task_transfer=None,adapt_from=checkpoint)
    train(native)


if __name__=='__main__':
    main()
