"""Compact plots/CSV of the first prospectively fixed selected-policy examples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    receipt=json.loads((args.input/'receipt.json').read_text())
    if receipt['status']!='completed' or receipt['actual_native_controls']>2000:
        raise ValueError('Require the completed finite native examples')
    args.output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    figure,axes=plt.subplots(1,2,figsize=(11,5.6))
    rows=[];geometry={};hashes={}
    for ax,method,label in zip(axes,('stable_iso','ea'),('Selected Iso','Selected EA')):
        source=args.input/f'{method}.json'
        value=json.loads(source.read_text());summary=value['summary'];initial=value['initial']
        hashes[method]=hashlib.sha256(source.read_bytes()).hexdigest()
        geometry[method]=initial
        records=value['records']
        positions=np.array([initial['agent_xy']]+[r['position_xy'] for r in records])
        hazards=initial['hazards']
        for index,(position,kind,size) in enumerate(zip(hazards['positions'],hazards['types'],hazards['sizes'])):
            position=np.asarray(position)[:2]
            if kind=='rect':
                half=np.asarray(hazards['half_extents'][index])[:2]
                patch=Rectangle(position-half,*(2*half),facecolor='#d4ad9c',edgecolor='#99634b',alpha=.65)
            else:
                patch=Circle(position,float(size),facecolor='#d4ad9c',edgecolor='#99634b',alpha=.65)
            ax.add_patch(patch)
        ax.plot(positions[:,0],positions[:,1],color='#236c9d',linewidth=1.3,label='Native trajectory')
        cost=np.array([r['native_cost'] for r in records])>0
        goal=np.array([r['native_goals'] for r in records])>0
        ax.scatter(positions[1:][cost,0],positions[1:][cost,1],s=13,c='#bd3434',label='Positive native cost',zorder=3)
        ax.scatter(positions[1:][goal,0],positions[1:][goal,1],s=20,c='#2b8c59',marker='D',label='Goal completion',zorder=4)
        ax.scatter(*positions[0],c='black',marker='x',s=45,label='Start',zorder=5)
        ax.scatter(*positions[-1],c='#236c9d',marker='s',s=28,label='End',zorder=5)
        ax.set_title(f"{label} — {summary['selected_steps']/1e6:.2f}M\n"
            f"Goals {summary['goals']:.0f}; cost {summary['cost']:.2f}; {summary['controls']} controls",fontsize=11)
        ax.set(xlabel='Native x (m)',ylabel='Native y (m)',aspect='equal')
        ax.autoscale_view();ax.margins(.06);ax.grid(alpha=.15)
        for record in records:
            rows.append(dict(method=method,step=record['step'],x=record['position_xy'][0],y=record['position_xy'][1],
                native_reward=record['native_reward'],native_cost=record['native_cost'],native_goals=record['native_goals'],
                action_json=json.dumps(record['action']),native_goals_after_json=json.dumps(record['goals_after'])))
    handles,labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=5,fontsize=8)
    figure.suptitle('One predeclared layout (61027), frozen selected policies, native horizon1000')
    figure.text(.5,.018,'Illustrative development trajectories only; no independent-test, training-seed-average or physical deployment claim.',
        ha='center',fontsize=9)
    figure.tight_layout(rect=(0,.13,1,.92))
    for suffix in ('png','pdf'):figure.savefig(args.output/f'pointgoal_selected_trajectories.{suffix}',dpi=170)
    plt.close(figure)
    with (args.output/'trajectory_controls.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'native_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    (args.output/'raw_source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for name in ('receipt.json','plan.json'):shutil.copy2(args.input/name,args.output/name)
    print(json.dumps(dict(native_control_rows=len(rows),new_simulation=0,new_training=0)))


if __name__=='__main__':
    main()
