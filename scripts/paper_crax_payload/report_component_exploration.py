"""Plot existing distributed trajectory summaries. Zero simulator/RL work."""
import argparse
import csv
import json
from pathlib import Path


def csv_write(path,rows):
    if not rows:return
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(value,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve earlier exploration report')
    results=value['results']
    names=['iso','noise-only','metric-only','full','full-rho010','full-rho050']
    if len(results)!=6 or {r['arm'] for r in results}!=set(names) or any(
            r['steps']!=300000 or not r['completed_receipt'] or not r['explicit_trace_end'] for r in results):
        raise ValueError('Final exploration report requires the full six-arm cohort')
    output.mkdir(parents=True)
    counts=[dict(arm=r['arm'],steps=r['steps'],safe_successes=r['successes'],
        training_violations=r['violations'],safe_timeouts=r['timeouts'],
        censored_episodes=r['censored_episodes']) for r in results]
    csv_write(output/'training_totals.csv',counts)
    matching=[dict(window_start=w['start'],window_end=w['end'],**r)
              for w in value['matched_success_coverage'] for r in w['arms']]
    csv_write(output/'matched_success_coverage.csv',matching)
    frontier=[dict(maximum_training_violations=w['maximum_training_violations'],**r)
              for w in value['equal_violation_budget'] for r in w['arms']]
    csv_write(output/'actual_violation_budget.csv',frontier)
    csv_write(output/'all_episode_training_curve.csv',[
        dict(arm=result['arm'],**point) for result in results for point in result['curve']])
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for result in results:
        points=[dict(steps=0,successes=0,violations=0),*result['curve']]
        axes[0].plot([r['steps']/1000 for r in points],[r['successes'] for r in points],lw=1,label=result['arm'])
        axes[1].plot([r['steps']/1000 for r in points],[r['violations'] for r in points],lw=1)
        axes[2].plot([r['violations'] for r in points],[r['successes'] for r in points],lw=1)
    axes[0].set(xlabel='Native transitions (K)',ylabel='Cumulative safe successes')
    axes[1].set(xlabel='Native transitions (K)',ylabel='Cumulative violations')
    axes[2].set(xlabel='Actual cumulative violations',ylabel='Cumulative safe successes')
    axes[0].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('All completed training episodes; censored episodes retained separately; one development seed')
    fig.tight_layout();fig.savefig(output/'task_discovery_and_risk.png',dpi=170)
    fig.savefig(output/'task_discovery_and_risk.pdf');plt.close(fig)
    core=['iso','noise-only','metric-only','full']
    fig,axes=plt.subplots(3,4,figsize=(12,8.5),squeeze=False)
    for col,name in enumerate(core):
        result=next(r for r in results if r['arm']==name)
        for row,window in enumerate(result['windows']):
            ax=axes[row,col]
            for example in window['first_success_examples']:
                points=example['tcp_xyz'];obstacle=example['obstacle']
                ax.plot([p[0]-obstacle[0] for p in points],[p[1]-obstacle[1] for p in points],lw=1)
            if not window['first_success_examples']:ax.text(.5,.5,'No safe success',ha='center',transform=ax.transAxes)
            ax.scatter([0],[0],s=15,color='black');ax.set_aspect('equal',adjustable='datalim')
            ax.set(title=f'{name}: {window["start"]//1000}-{window["end"]//1000}K',
                xlabel='TCP x - obstacle x (m)',ylabel='TCP y - obstacle y (m)');ax.grid(alpha=.15)
    fig.suptitle('First three chronological safe successes / window (not selected by route)\n'
                 'TCP projections only, not obstacle contours or full-arm legal-route proofs',fontsize=10)
    fig.tight_layout();fig.savefig(output/'first_chronological_training_trajectories.png',dpi=170)
    fig.savefig(output/'first_chronological_training_trajectories.pdf');plt.close(fig)
    full=next(r for r in results if r['arm']=='full');iso=next(r for r in results if r['arm']=='iso')
    last=value['equal_violation_budget'][-1]
    matched={r['arm']:r for r in last['arms']}
    text=(f'# 新训练期探索结果\n\n六臂均300K；同一个已知开发seed2027，不是跨seed统计。\n\n'
        f'同交互预算：Full安全成功{full["successes"]}、违规{full["violations"]}；'
        f'Iso安全成功{iso["successes"]}、违规{iso["violations"]}。\n\n'
        f'六臂共同累计违规上限{last["maximum_training_violations"]}时，'
        f'Full成功{matched["full"]["successes"]}（实际违规{matched["full"]["violations"]}），'
        f'Iso成功{matched["iso"]["successes"]}（实际违规{matched["iso"]["violations"]}）。'
        '使用真实日志节点，没有插值制造精确相同的风险。\n\n'
        '覆盖表每个窗口取四臂成功数的共同最小值，按时间顺序前N条；'
        '世界网格20mm、弧长重采样5mm。N=0时不宣称覆盖优势。'
        '所有成功数、违规、超时与预算末未完回合同时保留。\n\n'
        '轨迹图仅展示每窗口最早三个安全成功，不用于推断整体成功率。'
        'TCP投影／网格占用不证明全臂可行路线，也不证明拓扑多样性或泛化因果。'
        '冻结终点任务指标请另看组件／参数独立测试表；训练发现与执行效率不混用。\n')
    (output/'INTERPRETATION_ZH.md').write_text(text)
    receipt=dict(status='completed_from_existing_summaries',new_training_transitions=0,
        new_simulation_controls=0,physical_robot_interactions=0,all_six_arms_retained=True,
        training_seed_count=1,examples='first_three_chronological_successes_per_fixed_window',
        raw_trajectories_not_copied=True,legal_route_or_homotopy_proof=False)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(json.dumps(report(json.loads(args.comparison.read_text()),args.output)))


if __name__=='__main__':main()
