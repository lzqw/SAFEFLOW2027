"""One algebra/sampling check of the two requested mechanism controls, not RL."""
import argparse
import json
from pathlib import Path

import torch

from scripts.star.arm_safe_entropy.models import ArmFlowActor, ArmLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    torch.manual_seed(1337)
    obs = torch.zeros(4096, 25)
    def known_geometry(s, a):
        normal = torch.zeros_like(a)
        normal[..., 0] = 1.
        return normal, torch.ones(a.shape[:-1])
    flow = ArmFlowActor(25, reference_control='random_aniso')
    velocity = torch.tensor([.2,.3,.4])
    flow.velocity = lambda s,a,t: velocity.expand_as(a)
    with torch.no_grad():
        actions, energy, clip, trace = flow(obs, geometry=known_geometry, return_reference=True)
    normals = trace['normals']
    assert torch.allclose(normals, normals[:, :1].expand_as(normals))
    assert torch.allclose(normals.norm(dim=-1), torch.ones(4096,4), atol=1e-6)
    assert not torch.allclose(normals[0],normals[1])
    assert abs(float(normals[:,0,0].mean())) < .05
    assert torch.allclose(trace['vn'] + 2 * trace['vt'], torch.full((4096,4),.12),atol=1e-7)
    vnormal = (velocity*normals[:,0]).sum(-1)
    expected = .5*(vnormal.square()/.01+(velocity.square().sum()-vnormal.square())/.055)
    assert torch.allclose(energy,expected,atol=1e-5)
    # Illustrative variance for algebra only; NOT a chosen experimental setting.
    low = ArmLearner(25,'flow_iso_low',device='cpu',variance=.015)
    low.actor.velocity = lambda s,a,t: velocity.expand_as(a)
    with torch.no_grad():
        _, low_energy, _, low_trace = low.actor(obs[:4],return_reference=True)
    assert torch.allclose(low_trace['vn'],torch.full((4,4),.015))
    assert torch.allclose(low_trace['vt'],torch.full((4,4),.015))
    assert torch.allclose(low_energy,torch.full((4,),float(.5*velocity.square().sum()/.015)))
    assert abs(float(low.log_alpha.exp())-.015*.02)<1e-8
    assert abs(low.energy_target-.75/.015)<1e-8
    payload = dict(kind='mechanism_control_algebra_only_not_trained_efficacy',
        random_direction_fixed_across_action_generation=True, random_directions_independent_of_safety=True,
        random_aniso_reference_trace=.12, reference_noise_and_inverse_energy_paired=True,
        iso_low_normal_variance_not_total_trace_matching=True, iso_low_energy_rescaling_checked=True,
        iso_low_variance_used_for_algebra_only=.015, experiment_variance_not_selected=True,
        new_training_transitions=0, trained_control_results=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
