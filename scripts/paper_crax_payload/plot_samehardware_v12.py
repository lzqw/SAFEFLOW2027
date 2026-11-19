"""Plot the four complete4080 controls, including every early and final DEV node."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.paper_crax_payload.plot_simulation_v12 import read


ARMS = {
    'A-full-P5': ('P5 + late LR', '#0072B2', '-'),
    'A-Iso-matchedLR': ('Iso + late LR', '#555555', '--'),
    'A-no-lateLR': ('P5 constant LR', '#D55E00', '-.'),
    'A-Metric-matchedLR': ('Metric + late LR', '#009E73', ':'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); receipt = json.loads((args.analysis/'receipt.json').read_text())
    if (receipt['status'] != 'complete_offline_review' or receipt['runs'] != 4
            or not receipt['same_hardware_confirmation'] or receipt['development_nodes'] != 84
            or receipt['actual_completed_training_transitions'] != 1000000):
        raise ValueError('Use the strict complete four-arm samehardware summary')
    training = read(args.analysis/'training.csv'); dev = read(args.analysis/'development_nodes.csv')
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'figure.dpi':150,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(13,7.4),constrained_layout=True)
    panels = [('training_violations','Cumulative training violations'),
              ('training_cumulative_return','Cumulative sampled return'),
              ('safe_timeouts','Cumulative training timeouts')]
    for ax,(metric,title) in zip(axes[0],panels):
        for arm,(label,color,style) in ARMS.items():
            rows = sorted((r for r in training if r['arm']==arm),key=lambda r:int(r['steps']))
            if [int(r['steps']) for r in rows] != list(range(1000,250001,1000)):
                raise ValueError('Plot every1K training node')
            ax.plot([int(r['steps'])/1000 for r in rows],[float(r[metric]) for r in rows],
                    label=label,color=color,linestyle=style,linewidth=1.7)
        ax.axvspan(150,200,color='#999999',alpha=.09)
        ax.set_title(title); ax.set_xlabel('Training transitions (thousands)'); ax.grid(alpha=.2)
        ax.set_xlim(0,250)
    axes[0,0].legend(fontsize=8)
    for ax,condition,title in zip(axes[1],('nominal','radius065','execution085'),
                                 ('Nominal: safe success','Radius65mm: safe success','Execution0.85: safe success')):
        for arm,(label,color,style) in ARMS.items():
            rows = sorted((r for r in dev if r['arm']==arm and r['condition']==condition),
                          key=lambda r:int(r['transitions']))
            if [int(r['transitions']) for r in rows] != [25000,50000,75000,100000,150000,200000,250000]:
                raise ValueError('Plot all seven original DEV nodes, never best-only')
            ax.plot([int(r['transitions'])/1000 for r in rows],
                    [100*int(r['safe_successes'])/int(r['episodes']) for r in rows],
                    marker='o',markersize=3,label=label,color=color,linestyle=style,linewidth=1.7)
        ax.axvspan(150,200,color='#999999',alpha=.09)
        ax.set_title(title); ax.set_xlabel('Training transitions (thousands)')
        ax.set_ylabel('DEV safe success (%)'); ax.set_ylim(0,103); ax.set_xlim(0,250); ax.grid(alpha=.2)
    fig.suptitle('SafeReach complete250K: same RTX4080, one training seed2034;30 DEV layouts/condition',fontsize=12)
    fig.supxlabel('Shading: registered150-200K actor LR decay. Lines connect saved nodes; no independent test or seed uncertainty.',fontsize=9)
    for suffix in ('png','pdf'):
        fig.savefig(args.output/f'samehardware_learning_curves.{suffix}')
    plt.close(fig)
    (args.output/'provenance.json').write_text(json.dumps(dict(status='complete_existing_data_plot',
        input_receipt_sha256=hashlib.sha256((args.analysis/'receipt.json').read_bytes()).hexdigest(),
        source_commits=[r['source_commit'] for r in receipt['provenance']],
        training_seed_count=1,complete_native_runs=4,training_nodes=1000,DEV_nodes=84,
        all_registered_nodes_included=True,new_training_steps=0,new_simulation_controls=0,
        independent_test_used=False,best_checkpoint_selected=False,
        uncertainty='No seed uncertainty or significance estimate;connected saved nodes are not continuous-time dominance'),indent=2)+'\n')


if __name__=='__main__': main()
