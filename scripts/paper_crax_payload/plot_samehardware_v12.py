"""Plot the four complete4080 controls, including every early and final DEV node."""
import argparse
import hashlib
import json
from pathlib import Path
import time

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
    group=p.add_mutually_exclusive_group()
    group.add_argument('--endpoint-residual',action='store_true')
    group.add_argument('--late-endpoint',action='store_true')
    p.add_argument('--wait-checks',type=int,default=0,help='Optional finite half-hour wait for the strict complete analysis')
    args = p.parse_args()
    if not 0<=args.wait_checks<=48:
        raise ValueError('Bound the figure observer to at most24h')
    for check in range(args.wait_checks):
        if (args.analysis/'receipt.json').exists():
            break
        print(json.dumps(dict(wait_check=check+1,interval_seconds=1800,
            waiting_for='complete_offline_review',training_launches=0)),flush=True)
        if check+1<args.wait_checks:
            time.sleep(1800)
    receipt = json.loads((args.analysis/'receipt.json').read_text())
    expected_runs = 8 if args.late_endpoint else 6 if args.endpoint_residual else 4
    if args.late_endpoint:
        valid_scope=(receipt.get('new_campaign_completed_training_transitions')==300000
            and receipt.get('shared_original_prefix_steps_not_new')==100000
            and receipt.get('historical_comparison_training_transitions')==1500000
            and receipt.get('inherited_first100K_not_new_confirmation') is True
            and receipt.get('training_seed_count')==1)
    else:
        valid_scope=(receipt['development_nodes']==21*expected_runs
            and receipt['actual_completed_training_transitions']==250000*expected_runs
            and receipt.get('endpoint_residual_screen' if args.endpoint_residual else 'same_hardware_confirmation'))
    if (receipt['status'] != 'complete_offline_review' or receipt['runs'] != expected_runs
            or not receipt.get('all_registered_nodes_retained')
            or receipt['independent_test_used'] is not False
            or not valid_scope):
        raise ValueError('Use every complete registered native run, not partial/best endpoints')
    arms = dict(ARMS)
    if args.endpoint_residual or args.late_endpoint:
        arms = {k:(label+' (historical)',color,style) for k,(label,color,style) in arms.items()}
        arms.update({'E-control-P5':('P5 (current control)','#56B4E9','-.'),
                     'E-endpoint-residual':('P5 + endpoint residual','#CC79A7','-')})
    if args.late_endpoint:
        arms={k:(label+' (prior)' if k.startswith('E-') else label,color,style)
              for k,(label,color,style) in arms.items()}
        arms.update({'L-control-continuation':('P5 matched100K continuation','#E69F00','--'),
                     'L-endpoint-ramp100K150K':('P5 + post100K residual','#882255','-')})
    training = read(args.analysis/'training.csv'); dev = read(args.analysis/'development_nodes.csv')
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'figure.dpi':150,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
    panels = [('training_violations','Cumulative training violations'),
              ('training_cumulative_return','Cumulative sampled return'),
              ('safe_timeouts','Cumulative training timeouts'),
              ('training_stall_episodes','Cumulative stalled training episodes')]
    for ax,(metric,title) in zip(axes[0],panels):
        for arm,(label,color,style) in arms.items():
            rows = sorted((r for r in training if r['arm']==arm),key=lambda r:int(r['steps']))
            if [int(r['steps']) for r in rows] != list(range(1000,250001,1000)):
                raise ValueError('Plot every1K training node')
            ax.plot([int(r['steps'])/1000 for r in rows],[float(r[metric]) for r in rows],
                    label=label,color=color,linestyle=style,linewidth=1.7)
        ax.axvspan(150,200,color='#999999',alpha=.09)
        ax.set_title(title); ax.set_xlabel('Training transitions (thousands)'); ax.grid(alpha=.2)
        ax.set_xlim(0,250)
    axes[0,0].legend(fontsize=8)
    for ax,condition,title in zip(axes[1,:3],('nominal','radius065','execution085'),
                                 ('Nominal: safe success','Radius65mm: safe success','Execution0.85: safe success')):
        for arm,(label,color,style) in arms.items():
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
    ax = axes[1,3]
    for arm,(label,color,style) in arms.items():
        rows = sorted((r for r in training if r['arm']==arm),key=lambda r:int(r['steps']))
        ax.plot([int(r['steps'])/1000 for r in rows],[int(r['safe_successes']) for r in rows],
                label=label,color=color,linestyle=style,linewidth=1.7)
    ax.axvspan(150,200,color='#999999',alpha=.09)
    ax.set_title('Cumulative safe training successes');ax.set_xlabel('Training transitions (thousands)')
    ax.set_xlim(0,250);ax.grid(alpha=.2)
    if args.late_endpoint:
        for ax in axes.flat:
            ax.axvline(100,color='#777777',linewidth=.8,linestyle=':')
            ax.axvspan(100,150,color='#E69F00',alpha=.08)
    scope = ('matched100K prefix + late residual pair + prior references' if args.late_endpoint else
             'endpoint residual pair + historical references' if args.endpoint_residual else 'four original controls')
    fig.suptitle('SafeReach complete250K: '+scope+';one training seed2034,30 DEV layouts/condition',fontsize=12)
    annotation=('Late pair inherits identical observed0-100K prefix;gold100-150K residual ramp;gray150-200K LR decay.'
                if args.late_endpoint else 'Shading: registered150-200K actor LR decay.')
    fig.supxlabel(annotation+' Saved nodes connected;no independent test or seed uncertainty.',fontsize=9)
    basename = ('late_endpoint_learning_curves' if args.late_endpoint else
                'endpoint_residual_learning_curves' if args.endpoint_residual else 'samehardware_learning_curves')
    for suffix in ('png','pdf'):
        fig.savefig(args.output/(basename+'.'+suffix))
    plt.close(fig)
    (args.output/'provenance.json').write_text(json.dumps(dict(status='complete_existing_data_plot',
        input_receipt_sha256=hashlib.sha256((args.analysis/'receipt.json').read_bytes()).hexdigest(),
        source_commits=[r['source_commit'] for r in receipt['provenance']],
        training_seed_count=1,complete_native_runs=expected_runs,training_nodes=250*expected_runs,DEV_nodes=21*expected_runs,
        contemporary_baselines=receipt.get('contemporary_baselines'),
        historical_samehardware_baselines=receipt.get('historical_samehardware_baselines'),
        shared_original_prefix_steps_not_new=receipt.get('shared_original_prefix_steps_not_new'),
        new_campaign_completed_training_transitions=receipt.get('new_campaign_completed_training_transitions'),
        all_registered_nodes_included=True,new_training_steps=0,new_simulation_controls=0,
        independent_test_used=False,best_checkpoint_selected=False,
        uncertainty='No seed uncertainty or significance estimate;connected saved nodes are not continuous-time dominance'),indent=2)+'\n')


if __name__=='__main__': main()
