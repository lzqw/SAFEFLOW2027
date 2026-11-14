"""CRAX data bridge to pinned, unmodified official FSRL CVPO; no replacement RL."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer


UPSTREAM_COMMIT='e056fc9498d5d037869533da7cf976acf462f918'


def transition_batch(observation, action, reward, info):
    """Store terminal observations, never auto-reset policy observations."""
    from tianshou.data import Batch
    cpu=lambda x:x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    terminated=cpu(info['terminated']).astype(bool);truncated=cpu(info['truncated']).astype(bool)
    return Batch(obs=cpu(observation),act=cpu(action),rew=cpu(reward),
        terminated=terminated,truncated=truncated,done=terminated|truncated,
        obs_next=cpu(info['terminal_observation']),info=Batch(cost=cpu(info['cost'])))


class NormalizedReplayView:
    """Normalize only sampled obs at current frozen statistics; raw replay intact.

    All reward/cost/end-flag arrays and Tianshou per-env next indices delegate
    unchanged, so official multi-step return code still owns the target rules.
    """
    def __init__(self, raw, normalizer):
        self.raw=raw;self.normalizer=normalizer

    def __getattr__(self,name):return getattr(self.raw,name)

    def __getitem__(self,index):
        from tianshou.data import Batch
        value=self.raw[index]
        out=Batch(**{k:v for k,v in value.items()})
        for field in ('obs','obs_next'):
            device=self.normalizer.mean.device
            out[field]=self.normalizer.transform(torch.as_tensor(value[field],device=device,dtype=torch.float32))
        return out

    def sample(self,count):
        indices=self.raw.sample_indices(count)
        return self[indices],indices


def build_policy(obs_dim, action_dim, horizon, device='cpu', seed=51027):
    import gymnasium as gym
    from fsrl.agent import CVPOAgent
    env=SimpleNamespace(observation_space=gym.spaces.Box(-np.inf,np.inf,(obs_dim,),dtype=np.float32),
        action_space=gym.spaces.Box(-1.,1.,(action_dim,),dtype=np.float32),
        spec=SimpleNamespace(max_episode_steps=horizon))
    # Standard official Gaussian/trust-region method, not a Flow architecture.
    # Shared gamma, cost25 and Acme normalization are explicit benchmark choices.
    return CVPOAgent(env,cost_limit=25.,device=device,thread=1,seed=seed,
        gamma=.99,n_step=2,hidden_sizes=(128,128),actor_lr=5e-4,critic_lr=1e-3,
        tau=.05,sample_act_num=16,deterministic_eval=False).policy


def export_full_policy_state(policy):
    # Official get_extra_state is empty; state_dict alone is NOT recoverable.
    return dict(policy=policy.state_dict(),actor_optimizer=policy.actor_optim.state_dict(),
        critic_optimizer=policy.critics_optim.state_dict(),estep_dual=policy.estep_dual.detach().clone(),
        estep_optimizer=policy.estep_optim.state_dict(),
        mstep_dual_mu=policy.mstep_dual_mu.detach().clone(),
        mstep_dual_std=policy.mstep_dual_std.detach().clone(),mstep_optimizer=policy.mstep_optim.state_dict(),
        official_source_commit=UPSTREAM_COMMIT)


def synthetic_smoke(upstream):
    """Real official updates on synthetic fixtures, ZERO native interactions."""
    from fsrl.policy.cvpo import CVPO
    from tianshou.data import VectorReplayBuffer
    import tianshou
    upstream=Path(upstream)
    source=Path(__import__('inspect').getfile(CVPO)).resolve()
    if not source.is_relative_to(upstream.resolve()):raise ValueError('Wrong official source import')
    policy=build_policy(5,2,1000)
    norm=ObservationNormalizer(5,'cpu','running_acme')
    norm.update(torch.arange(20,dtype=torch.float32).reshape(4,5))
    raw=VectorReplayBuffer(128,2);view=NormalizedReplayView(raw,norm)
    indices=None
    for i in range(6):
        obs=torch.full((2,5),float(i))
        # Both episode boundary types retained; terminal obs is deliberately
        # distinct from the separate next policy reset observation.
        info=dict(terminated=torch.tensor([i==2,False]),truncated=torch.tensor([False,i==2]),
            terminal_observation=obs+10,cost=torch.tensor([0.,1.]))
        batch=transition_batch(obs,torch.zeros(2,2),torch.tensor([1.,2.]),info)
        added=raw.add(batch,buffer_ids=np.arange(2))[0]
        if i==2:indices=added
    assert np.array_equal(raw[indices].obs_next,np.full((2,5),12.,np.float32))
    before=raw[indices].obs.copy();sample=view[indices]
    assert torch.equal(sample.obs,norm.transform(torch.as_tensor(before)))
    assert np.array_equal(before,raw[indices].obs)
    # Actual official one-step semantics: true termination zeroes bootstrap;
    # time-limit truncation retains it from the terminal, not reset, state.
    checked=policy.compute_nstep_returns(view[indices],view,indices,
        lambda buffer,idx:[torch.full((len(idx),1),3.),torch.full((len(idx),1),4.)],n_step=1)
    expected=torch.tensor([[1.,0.],[2.+.99*3.,1.+.99*4.]])
    assert torch.allclose(checked.rets.reshape(2,2),expected,atol=1e-5)
    actor_before=[p.detach().clone() for p in policy.actor.parameters()]
    policy.pre_update_fn()
    for _ in range(2):policy.update(8,view)
    policy.post_update_fn()
    assert any(not torch.equal(a,b) for a,b in zip(actor_before,policy.actor.parameters()))
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    full=export_full_policy_state(policy)
    assert full['estep_optimizer']['state'] and full['mstep_optimizer']['state']
    return dict(status='official_synthetic_bridge_smoke_passed',official_source_commit=UPSTREAM_COMMIT,
        source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch_version=str(torch.__version__),numpy_version=np.__version__,tianshou_version=tianshou.__version__,
        terminal_and_truncation_bootstrap_checked=True,raw_replay_and_current_normalization_checked=True,
        official_updates=2,native_training_transitions=0,synthetic_fixture_transitions=12,
        official_dual_optimizer_recovery_state_explicit=True,cost_limit=25.,
        discounted_qc_threshold=policy.qc_thres[0],physical_robot_interactions=0,
        real_environment_compatibility_proven=False,formal_baseline_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic-smoke',action='store_true',required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous evidence')
    torch.set_num_threads(1)
    result=synthetic_smoke(args.upstream)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
