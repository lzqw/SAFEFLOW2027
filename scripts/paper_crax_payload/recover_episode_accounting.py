"""Explicit offline recovery only when the entire genuine prefix is in replay.

No environment step, learner update, model/optimizer/RNG or multiplier change.
Only native-per-step dual permits this repair: its episode EMA is diagnostic.
"""
import numpy as np
import torch

def recover_episode_accounting(state):
    settings=state['learner']['settings'];task=state['environment']['config']
    if settings.get('dual_rule')!='native_per_step':
        raise ValueError('Episode-dual training cannot be silently corrected after the fact')
    replay=state['replay'];E=state['profile']['E'];steps=state['steps']
    if replay['size']!=steps or steps>replay['capacity'] or steps%E:
        raise ValueError('Full chronological prefix required;never fabricate missing history')
    if task['reward_distance_scale']!=0 or task['reward_goal']!=1:
        raise ValueError('This recovery requires the recorded native reward=goals configuration')
    data={k:replay['data'][k].detach().cpu().numpy().reshape(-1,E)
          for k in ('reward','cost','terminated','truncated')}
    partial={k:np.zeros(E,dtype=np.float64) for k in ('return','cost','goals','length')}
    totals=np.zeros(6);rows=[];ema=0.;count=0
    for index in range(steps//E):
        partial['return']+=data['reward'][index];partial['goals']+=data['reward'][index]
        partial['cost']+=data['cost'][index];partial['length']+=1
        done=(data['terminated'][index]!=0)|(data['truncated'][index]!=0)
        if done.any():
            episode_cost=partial['cost'][done]
            ema=float(episode_cost.mean()) if count==0 else .95*ema+.05*float(episode_cost.mean())
            count+=int(done.sum())
            totals[1:]+= [done.sum(),partial['return'][done].sum(),episode_cost.sum(),
                         partial['goals'][done].sum(),(partial['goals'][done]>=1).sum()]
            rows.extend([int((index+1)*E),float(partial['return'][j]),float(partial['cost'][j]),
                          float(partial['goals'][j]),int(partial['length'][j])] for j in np.flatnonzero(done))
            for value in partial.values():value[done]=0
    before=dict(episodes=float(state['totals'][1]),partial_lengths=state['partial']['length'][:4].tolist())
    for key,values in partial.items():state['partial'][key].copy_(torch.as_tensor(values,device=state['partial'][key].device))
    # Preserve the already recorded raw accumulated cost, including FP summation.
    state['totals'][1:].copy_(torch.as_tensor(totals[1:],device=state['totals'].device))
    state['learner']['episode_cost_ema'].fill_(ema)
    state['learner']['dual_episodes'].fill_(count)
    return dict(before=before,recovered_complete_episodes=count,
        recovered_partial_lengths=partial['length'][:4].tolist(),recovered_episode_cost_ema=ema,
        original_raw_training_cost_preserved=float(state['totals'][0]),
        models_optimizers_multiplier_replay_environment_rng_changed=False,
        new_training_transitions=0,nontraining_simulator_controls=0),rows
