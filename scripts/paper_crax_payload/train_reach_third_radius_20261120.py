"""Explicit third-parent300K ->400K fork; requires actual completed parent hashes."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace

import torch

from scripts.paper_crax_payload.piper_train import FORMAT, train
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256
from scripts.paper_crax_payload.train_reach_third_parent_20261120 import (
    PLAN as PARENT_PLAN, configuration as parent_configuration)
from scripts.star.arm_safe_entropy.train import atomic_json


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path('/home/carla/4T/SAFEFLOW2027_SIMULATION_V12')
PARENT_SOURCE = '58bf769dceb1eabd2eec07c627decf68d810c2aa'


def validate_parent(saved, receipt, manifest):
    expected = parent_configuration(json.loads(PARENT_PLAN.read_text()))
    if (saved['format'] != FORMAT or saved['robot_family'] != 'Piper'
            or saved['task'] != 'Reach' or saved['method'] != 'EA'
            or saved['training_seed'] != 41030 or saved['steps'] != 300000
            or saved['source_commit'] != PARENT_SOURCE or saved['engineering'] is not False
            or saved['configuration'] != expected or saved.get('geometry_adaptation') is not None
            or saved.get('task_transfer') is not None
            or receipt['status'] != 'completed' or receipt['training_seed'] != 41030
            or receipt['training_steps'] != 300000 or receipt['initial_steps'] != 0
            or receipt['newly_collected_transitions'] != 300000
            or receipt['training_source_commit'] != PARENT_SOURCE or receipt['engineering'] is not False
            or manifest['source_commit'] != PARENT_SOURCE or manifest['training_seed'] != 41030
            or manifest['initial_steps'] != 0 or manifest['total_transition_budget'] != 300000
            or manifest['configuration'] != expected or manifest['parent_checkpoint'] is not None):
        raise ValueError('Only the actual fixed300K independent third parent; never a best node or replacement seed')
    for receipt_key, saved_key in [('training_violation_count','violations'),
            ('training_safe_successes','safe_successes'), ('training_safe_timeouts','safe_timeouts'),
            ('training_episodes','episodes')]:
        if receipt[receipt_key] != saved[saved_key]:
            raise ValueError('Retain exact inherited native counters, including failed episodes')
    if saved['episodes'] != saved['violations']+saved['safe_successes']+saved['safe_timeouts']:
        raise ValueError('All inherited terminal classes required')
    config = deepcopy(expected)
    config['training_radius_values'] = [.045, .05, .055]
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--parent-receipt-sha256', required=True)
    parser.add_argument('--parent-manifest-sha256', required=True)
    parser.add_argument('--assets-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = ROOT.resolve(strict=True)
    parent = root/'runs-third-parent-20261120/Reach-EA-41030/full_300000.pt'
    directory = parent.parent
    output = root/'runs-third-radius-20261120/Reach-EA-41030'
    if (platform.node() != '447' or not Path('/home/carla/4T').is_mount()
            or root.stat().st_dev != Path('/home/carla/4T').stat().st_dev
            or args.output.resolve() != output or args.output.exists()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or args.device != 'cuda:0'):
        raise ValueError('Exact new4080 data-disk output and one shared GPU required')
    for path, digest in [(parent,args.parent_sha256), (directory/'receipt.json',args.parent_receipt_sha256),
                         (directory/'run_config.json',args.parent_manifest_sha256)]:
        if len(digest) != 64 or sha256(path) != digest:
            raise ValueError('Bind actual completed parent/checkpoint/receipt/config hashes before launch')
    manifest = json.loads((directory/'run_config.json').read_text())
    proc = Path('/proc', str(manifest['native_pid']), 'cmdline')
    try:
        command = proc.read_bytes()
    except FileNotFoundError:
        command = b''
    if str(directory).encode() in command:
        raise ValueError('Parent still live; do not launch the downstream fork')
    if sorted(p.name for p in directory.glob('evaluation_*.json')) != [
            'evaluation_120000.json','evaluation_180000.json','evaluation_240000.json',
            'evaluation_300000.json','evaluation_60000.json']:
        raise ValueError('Keep all five original development nodes including regressions')
    receipt = json.loads((directory/'receipt.json').read_text())
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    config = validate_parent(saved, receipt, manifest)
    if saved['replay']['data']['obs'].shape[-1] != 34 or len(saved['rng']['cuda']) != 1:
        raise ValueError('Keep native34D replay and the actual single-GPU generator index0')
    del saved
    if torch.cuda.device_count() != 1:
        raise ValueError('One visible shared GPU required')
    torch.cuda.set_per_process_memory_fraction(.025, torch.device(args.device))

    def storage_guard():
        # Reserve full2GiB for this fork and2GiB for the concurrent Payload
        # parent at each checkpoint; no credit for mutable growing outputs.
        if shutil.disk_usage(root).free < 44*1024**3:
            raise OSError('Keep40GiB floor plus both complete prospective output budgets')

    storage_guard()
    registration = root/'third-radius-configuration-20261120'
    registration.mkdir(exist_ok=False)
    path = registration/'configuration.json'
    atomic_json(path, config)
    atomic_json(registration/'registration.json', dict(training_seed=41030,
        source_commit=args.source_commit, parent_checkpoint=str(parent),
        parent_sha256=args.parent_sha256, parent_receipt_sha256=args.parent_receipt_sha256,
        parent_manifest_sha256=args.parent_manifest_sha256, config_sha256=sha256(path),
        initial_steps=300000, target_steps=400000, new_training_budget=100000,
        changed_scientific_fields=['training_radius_values'], training_radius_values=[.045,.05,.055],
        inherited_counts={key:receipt[key] for key in ('training_violation_count','training_safe_successes',
            'training_safe_timeouts','training_episodes')},
        fork_rng='native_distribution_fork_reseeded_by_original_seed_not_exact_resume',
        independent_test_used=False, scientific_early_stopping=False, automatic_retry=False,
        downstream_method_confirmation_started=False, physical_robot_interactions=0))
    native = SimpleNamespace(config=path, assets_receipt=args.assets_receipt, task='Reach', method='EA',
        seed=41030, steps=400000, source_commit=args.source_commit, device=args.device, output=output,
        engineering=False, resume=None, init_from=None, task_transfer=None, adapt_from=parent)
    train(native, storage_guard=storage_guard)


if __name__ == '__main__':
    main()
