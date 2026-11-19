"""Fresh native Reach300K from two true seeds, without weakening old loaders."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.paper_crax_payload.piper_train import train
from scripts.paper_crax_payload.train_native_p5_adaptation import storage_check
from scripts.star.arm_safe_entropy.train import atomic_json


PROJECT = Path(__file__).resolve().parents[2]
PLAN = PROJECT/'configs/paper_crax_payload/simulation_v12_reach_fresh_parents.json'
ROOT = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12')


def configuration(plan, seed):
    if plan != json.loads(PLAN.read_text()) or seed not in plan['training_seeds']:
        raise ValueError('Only the registered two independent native training seeds')
    base = PROJECT/plan['base_config']
    if hashlib.sha256(base.read_bytes()).hexdigest() != plan['base_config_sha256']:
        raise ValueError('Do not change the original native300K scientific recipe')
    config = deepcopy(json.loads(base.read_text()))
    config['development_seed'] = seed
    config['evaluation']['development_layout_seed_start'] = plan['development_layout_seed_start']
    config['evaluation']['action_sampling_seed'] = plan['action_sampling_seed']
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--assets-receipt',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True); p.add_argument('--device',default='cuda:0')
    args = p.parse_args(); plan = json.loads(args.plan.read_text()); config = configuration(plan,args.seed)
    if not args.output.resolve().is_relative_to(ROOT.resolve(strict=True)) or args.output.exists():
        raise ValueError('New distinct data-disk directory only;never overwrite or silently resume')
    storage_check(ROOT,growth_gib=2)
    device = torch.device(args.device)
    if device.type=='cuda':
        if torch.cuda.device_count()!=1:
            raise ValueError('One explicit visible shared GPU required;record actual RNG index0')
        torch.cuda.set_per_process_memory_fraction(.025,device)
    configs = ROOT/'fresh-parent-configurations'; configs.mkdir(exist_ok=True)
    path = configs/f'reach-EA-{args.seed}.json'
    if path.exists():
        raise FileExistsError('Preserve previous exact seed registration')
    atomic_json(path,config)
    atomic_json(configs/f'reach-EA-{args.seed}-registration.json',dict(plan=plan,training_seed=args.seed,
        source_commit=args.source_commit,configuration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        initialization='fresh_networks_optimizers_replay_dual_environment_rng_no_parent_import',
        saved_cuda_rng_index=0,output=str(args.output),independent_test_used=False,
        automatic_retry=False,downstream_training_started=False))
    native = SimpleNamespace(config=path,assets_receipt=args.assets_receipt,task='Reach',method='EA',
        seed=args.seed,steps=300000,source_commit=args.source_commit,device=args.device,output=args.output,
        engineering=False,resume=None,init_from=None,task_transfer=None,adapt_from=None)
    train(native)


if __name__=='__main__': main()
