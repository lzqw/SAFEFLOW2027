"""Explicit disk-interruption recovery on4080; retain the abandoned150K–175K segment."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace

import torch

from scripts.paper_crax_payload.piper_train import FORMAT, train
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256
from scripts.paper_crax_payload.train_payload_fresh_parents_20261120 import PLAN, configuration
from scripts.star.arm_safe_entropy.train import atomic_json


ROOT = Path('/home/carla/4T/SAFEFLOW2027_SIMULATION_V12')
CHECKPOINT_SHA = '6839e61e35f7b5ead69b1cf9f1c4b9d27b24141cb3f1795a33aed538fa4f0579'
ORIGINAL_SOURCE = '870e7e81ce29ab0b06a24fe546544d1f848ad666'


def validate(saved, manifest):
    expected = configuration(json.loads(PLAN.read_text()), 42028)
    if (saved['format'] != FORMAT or saved['task'] != 'Payload' or saved['method'] != 'EA'
            or saved['training_seed'] != 42028 or saved['steps'] != 150000
            or saved['source_commit'] != ORIGINAL_SOURCE or saved['configuration'] != expected
            or saved['engineering'] is not False or saved.get('geometry_adaptation') is not None
            or saved.get('task_transfer') is not None or len(saved['rng']['cuda']) != 1
            or manifest['host'] != 'CTAlab' or manifest['native_pid'] != 452062
            or manifest['source_commit'] != ORIGINAL_SOURCE or manifest['configuration'] != expected
            or manifest['initial_steps'] != 0 or manifest['total_transition_budget'] != 300000
            or saved['episodes'] != saved['violations']+saved['safe_successes']+saved['safe_timeouts']):
        raise ValueError('Only the exact original150K full native state and failed A100 run')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    root = ROOT.resolve(strict=True)
    inputs = root/'payload42028-recovery-input-20261121'
    parent = inputs/'latest_full.pt'
    if (platform.node() != '447' or not Path('/home/carla/4T').is_mount()
            or args.output.resolve() != root/'runs-payload-recovery-20261121/Payload-EA-42028'
            or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or sha256(parent) != CHECKPOINT_SHA):
        raise ValueError('New4080 data-disk continuation from the exact verified checkpoint only')
    manifest = json.loads((inputs/'run_config.json').read_text())
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    config = validate(saved, manifest)
    registration = root/'payload42028-recovery-registration-20261121'
    registration.mkdir(exist_ok=False)
    path = registration/'configuration.json'
    atomic_json(path, config)
    atomic_json(registration/'registration.json', dict(source_commit=args.source_commit,
        original_source_commit=ORIGINAL_SOURCE, checkpoint_sha256=CHECKPOINT_SHA,
        original_manifest_sha256=sha256(inputs/'run_config.json'),
        original_progress_sha256=sha256(inputs/'progress.jsonl'),
        restored_steps=150000, original_observed_steps=175000, target_steps=300000,
        abandoned_segment_steps=25000, reason='A100 data-disk40GiB guard;original exit1 retained',
        restore='networks_optimizers_replay_dual_alpha_environments_partial_episodes_CPU_and_single_CUDA_RNG',
        cross_hardware_bitwise_equivalence_claimed=False, fresh_initialization=False,
        independent_test_used=False, automatic_retry=False, original_evidence_retained=True))
    del saved
    if torch.cuda.device_count() != 1:
        raise ValueError('One shared GPU; retain original saved generator index0')
    torch.cuda.set_per_process_memory_fraction(.025, torch.device('cuda:0'))
    def storage_guard():
        # New recovery2GiB plus all three forthcoming Reach arms6GiB.
        if shutil.disk_usage(root).free < 48*1024**3:
            raise OSError('Keep40GiB floor plus full8GiB registered recovery/confirmation output growth')
    storage_guard()
    native = SimpleNamespace(config=path,
        assets_receipt=Path('/home/carla/4T/SAFEFLOW2027_PAPER/analysis/piper/assets-ac41fcb.json'),
        task='Payload', method='EA', seed=42028, steps=300000, source_commit=args.source_commit,
        device='cuda:0', output=args.output, engineering=False, resume=parent,
        init_from=None, task_transfer=None, adapt_from=None)
    train(native, storage_guard=storage_guard)


if __name__ == '__main__':
    main()
