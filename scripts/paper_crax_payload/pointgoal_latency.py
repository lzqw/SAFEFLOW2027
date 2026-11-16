"""Full batch1 timing of the final selected native PointGoal Flow policies.

Uses genuine recorded observations from the existing4M common pool ONLY as a
fixed timing fixture. No physical simulation or final-model mechanism claim is
made from that old pool. The selected Actor/EMA/normalizer remain unchanged.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner
from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--observation-pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the measured latency result')
    torch.set_num_threads(1)
    selection = json.loads(args.selection.read_text())
    pool = json.loads(args.observation_pool.read_text())
    raw = np.array([r['observation'] for r in pool['records']], dtype=np.float32)
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Need the genuine existing fixed observation fixture')
    results = []
    for method in ('stable_iso','ea'):
        candidate = selection['methods'][method]['selected']
        path = args.run_root/candidate['run']/f"policy_{candidate['checkpoint_steps']}.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != candidate['policy_sha256']:
            raise ValueError('Require the exact frozen validation-selected policy')
        saved = torch.load(path, map_location=args.device, weights_only=False)
        state, settings = saved['learner'], saved['learner']['settings']
        permitted = set(inspect.signature(CumulativeFlowLearner.__init__).parameters)-{'self','device'}
        constructor = {k:v for k,v in settings.items() if k in permitted}
        if 'lagrangian_base_lr' in settings:
            constructor['lagrangian_lr'] = settings['lagrangian_base_lr']
        learner = CumulativeFlowLearner(**constructor, device=args.device)
        learner.load_state_dict(state)
        learner.actor.eval().requires_grad_(False); learner.margin_geometry.eval().requires_grad_(False)
        normalizer = ObservationNormalizer(settings['obs_dim'],learner.device,saved['normalizer']['mode'])
        normalizer.load_state_dict(saved['normalizer'])
        def synchronize():
            if learner.device.type=='cuda':
                torch.cuda.synchronize(learner.device)
        def action(observation):
            obs = torch.as_tensor(observation[None], device=learner.device)
            with torch.no_grad():
                value = learner.actor(normalizer.transform(obs),geometry=learner.geometry)[0]
            return value.cpu().numpy()
        torch.manual_seed(695000)
        timings = []
        for index in range(220):
            synchronize(); start=time.perf_counter()
            action(raw[index % len(raw)])
            synchronize()
            if index>=20:
                timings.append(time.perf_counter()-start)
        count = dict(velocity_network_forwards=0, margin_ensemble_forwards=0)
        def velocity_hook(*_): count['velocity_network_forwards']+=1
        def margin_hook(*_): count['margin_ensemble_forwards']+=1
        hooks=[learner.actor.net.register_forward_hook(velocity_hook),
               learner.margin_geometry.register_forward_hook(margin_hook)]
        try:
            action(raw[0])
        finally:
            for hook in hooks: hook.remove()
        count['batched_head_VJP_traversals']=count['margin_ensemble_forwards']
        count['per_head_action_gradients']=2*count['margin_ensemble_forwards']
        results.append(dict(method=method, selected_run=candidate['run'], selected_steps=candidate['checkpoint_steps'],
            policy_sha256=digest, device=str(learner.device), host=platform.node(),
            processor=platform.processor(), torch_version=str(torch.__version__), threads=1,
            batch_size=1,warmup=20, measured_actions=200,
            p50_s=float(np.quantile(timings,.5)),p95_s=float(np.quantile(timings,.95)),
            mean_s=float(np.mean(timings)),base_integration_steps=learner.actor.num_steps,
            forward_counts=count, physical_controls=0, training_transitions=0,
            includes='observation conversion/normalization, sampler, geometry/tail/input gradients, host return',
            measurement_host_is_not_a_claim_of_deployment_machine=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(results=results,selection_freeze_digest=selection['freeze_digest'],
        fixture_path=str(args.observation_pool), fixture_sha256=hashlib.sha256(args.observation_pool.read_bytes()).hexdigest(),
        fixture_role='existing recorded physical states for timing only; not final64x32 mechanism',
        SAC_latency_not_substituted_with_another_robot_policy=True),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
