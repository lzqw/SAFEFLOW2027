"""Bounded disposable learner-update memory probe; never collect transitions."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from scripts.paper_crax_payload.payload_models import build_payload_learner
from scripts.star.arm_safe_entropy.train import Replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--parents', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Retain previous measured probe')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.125, 0)
    config = json.loads(args.config.read_text())
    results = []
    for method, name in [('s_iso', 'payload-iso-41028-v1/full_120000.pt'),
                         ('sac_lag', 'payload-sac-41028-v1/full_300000.pt')]:
        path = args.parents / name
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['method'] != method or saved['training_seed'] != 41028:
            raise ValueError('Require the actual registered native parent')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        learner = build_payload_learner(config, method, 'cuda:0')
        learner.load_state_dict(saved['learner'])
        replay = Replay(config['shared_learner']['replay_capacity'], 45)
        replay.load_state_dict(saved['replay'])
        before = (learner.updates, learner.actor_updates)
        for _ in range(4):
            learner.update(replay.sample(config['shared_learner']['batch_size'], learner.device, recent=True))
        torch.cuda.synchronize()
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                       '--format=csv,noheader,nounits'], text=True)
        own = next((int(r.split(',')[1]) for r in apps.splitlines()
                    if int(r.split(',')[0]) == os.getpid()), None)
        results.append(dict(method=method, parent=str(path), native_parent_steps=saved['steps'],
            batch_size=config['shared_learner']['batch_size'], disposable_update_rounds=learner.updates-before[0],
            disposable_actor_updates=learner.actor_updates-before[1],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), observed_process_gpu_mib=own))
        del learner, replay, saved
        gc.collect()
    result = dict(probe='bounded_disposable_native_parent_updates', native_pid=os.getpid(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cuda_allocator_fraction_cap=.125, updates_are_discarded=True,
        new_environment_transitions=0, scientific_checkpoints_changed=False,
        result_cannot_bound_foreign_memory_growth=True, methods=results)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
