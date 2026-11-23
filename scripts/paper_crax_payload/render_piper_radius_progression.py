"""Run and film a fixed, paired exploratory sweep without retraining or layout substitution."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from scripts.paper_crax_payload.evaluate_selected_seed_holdout_20261123 import PLAN, load_case
from scripts.paper_crax_payload.evaluate_piper_deployment_modes_v12 import deployment_mode
from scripts.paper_crax_payload.native_p5_adaptation import target_deployment
from scripts.paper_crax_payload.piper_train import make_piper_env
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256
from scripts.paper_crax_payload.render_piper_holdout_examples_20261123 import Scene, panel
from scripts.paper_crax_payload.render_piper_reach_difficulty_20261123 import elevation
from scripts.paper_crax_payload.summarize_simulation_v12 import output
from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng


# Replay the same examples shown earlier; the batch is the first 20 previously
# registered evaluation layout seeds excluding the demonstration.
SETTINGS = {
    'Reach': (12310075, (45, 50, 55, 60, 65, 70, 75)),
    'Payload': (12320005, (45, 50, 55, 57.5, 60, 65, 70)),
}
TRAINED = (45, 50, 55)
NAME = 'piper_reach_then_payload_radius_progression'
SIZE = (1280, 720)
BG, INK, MUTED = '#edf3f8', '#132d46', '#44647f'
GREEN, BLUE, AMBER = '#08785b', '#126a9b', '#935c20'


@lru_cache
def font(size, bold=False):
    suffix = 'Bold' if bold else 'Regular'
    return ImageFont.truetype(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{suffix}.ttc', size, index=2)


def text(draw, xy, s, size=22, *, fill=INK, bold=False, anchor=None):
    draw.text(xy, s, font=font(size,bold), fill=fill, anchor=anchor)


def signature(env):
    return np.r_[env.task_position(), env.goal, env.obstacle_position,
                 [env.b('getJointState', env.robot, j)[0] for j in env.arm_joints]]


def sweep(learner, cfg, assets, task, radius, seeds, demo, reference, action_seed):
    env = make_piper_env(cfg, assets, task, {'obstacle_radius_m': radius/1000})
    scene = None
    results, clip = {}, None
    try:
        with isolated_rng(), target_deployment(learner), deployment_mode(learner, 'zero_noise'):
            for index, seed in enumerate(seeds):
                torch.manual_seed(action_seed+index)
                torch.cuda.manual_seed_all(action_seed+index)
                try:
                    obs, info = env.reset(seed=seed)
                except RuntimeError as exc:
                    if not str(exc).startswith('No safe reachable Piper start/goal'):
                        raise
                    results[seed] = dict(outcome='reset_rejected', reason='no_admissible_reset')
                    continue
                state = signature(env)
                if seed not in reference or not np.allclose(state, reference[seed], rtol=0, atol=1e-8):
                    results[seed] = dict(outcome='reset_rejected', reason='original_layout_not_admitted')
                    continue
                minimum = info['minimum_distance']
                if seed==demo:
                    scene = Scene(env,assets)
                    clip = dict(frames=[scene.frame(env)], points=[env.task_position().tolist()],
                        distances=[minimum], goal=env.goal.tolist(), obstacle=env.obstacle_position.tolist(),
                        radius=radius/1000, dt=env.control_dt)
                for step in range(1,env.config.horizon+1):
                    obs,_,done,truncated,info = env.step(np.clip(learner.action(obs[None])[0],-1.,1.))
                    minimum = min(minimum,info['minimum_distance'])
                    if seed==demo:
                        clip['frames'].append(scene.frame(env))
                        clip['points'].append(env.task_position().tolist())
                        clip['distances'].append(minimum)
                    if done or truncated:
                        break
                else:
                    raise RuntimeError('Native horizon did not terminate the rollout')
                success,violation = bool(info['safe_success']),bool(info['violation'])
                outcome = 'safe_success' if success else 'safety_stop' if violation else 'timeout'
                results[seed] = dict(outcome=outcome, reason='', minimum_clearance_mm=1000*minimum,
                    collision=minimum<=0, steps=step, terminal_time_s=step*env.control_dt)
                if seed==demo:
                    clip.update(success=success, violation=violation)
                    scene.close()
                    scene = None
    finally:
        if scene is not None:
            scene.close()
        env.close()
    return results, clip


def batch_layouts(task, spec):
    start = spec['layout_seed_start']
    demo = SETTINGS[task][0]
    return [seed for seed in range(start, start+21) if seed!=demo][:20]


def evaluate_task(root, assets, task, spec):
    demo, radii = SETTINGS[task]
    batch = batch_layouts(task,spec)
    seeds = batch+[demo]
    reference, records, clips, identities = {}, [], {}, []
    for ci, case in enumerate(spec['cases']):
        learner,cfg,identity = load_case(root, case, task, spec['training_seed'])
        line_name = 'piper_reach' if task=='Reach' else 'piper_payload'
        # Derive the coverage from the checkpoint's original completed run.
        native = json.loads((root/case['relative_run']/'run_config.json').read_text())
        assert tuple(round(x*1000) for x in native['plan']['lines'][line_name]['training_radius_values'])==TRAINED
        identities.append(dict(case=case['name'], checkpoint_sha256=identity['checkpoint_sha256']))
        if ci==0:
            fixture = make_piper_env(cfg,assets,task,{'obstacle_radius_m':.045})
            try:
                for seed in seeds:
                    fixture.reset(seed=seed)
                    reference[seed] = signature(fixture)
            finally:
                fixture.close()
            physics = cfg['simulation']
        elif cfg['simulation']!=physics:
            raise ValueError('Paired cases have different native physics')
        method = 'SAFEFlow' if ci==0 else 'Metric-only'
        for radius in radii:
            result,clip = sweep(learner,cfg,assets,task,radius,seeds,demo,reference,
                                spec['action_seed_start']+2000)
            clips[radius,method] = clip
            records.extend(dict(task=task,method=method,radius_mm=radius,layout_seed=seed,
                                shown_in_video=seed==demo,**row) for seed,row in result.items())
            print(task,method,radius,result[demo]['outcome'],flush=True)
        del learner
        torch.cuda.empty_cache()
    return records,clips,identities


def aggregate(records):
    results = {}
    for task,(_,radii) in SETTINGS.items():
        for radius in radii:
            masks=[]
            for method in ('SAFEFlow','Metric-only'):
                take=[r for r in records if r['task']==task and r['radius_mm']==radius
                      and r['method']==method and not r['shown_in_video']]
                if len(take)!=20 or len({r['layout_seed'] for r in take})!=20:
                    raise ValueError('The full fixed batch is required')
                counts={k:sum(r['outcome']==k for r in take)
                        for k in ('safe_success','safety_stop','timeout','reset_rejected')}
                if sum(counts.values())!=20:
                    raise ValueError('Unexpected terminal class')
                masks.append({r['layout_seed'] for r in take if r['outcome']=='reset_rejected'})
                results[task,radius,method] = dict(task=task,radius_mm=radius,method=method,
                    attempted_layouts=20,admitted_layouts=20-counts['reset_rejected'],**counts)
            if masks[0]!=masks[1]:
                raise ValueError('Initialization admission must be policy independent')
    return results


def base():
    im=Image.new('RGB',SIZE,BG)
    return im,ImageDraw.Draw(im)


def training_card(task):
    im,d=base()
    d.rectangle((0,0,1280,12),fill=BLUE)
    ordinal='01' if task=='Reach' else '02'
    subtitle='末端到达目标' if task=='Reach' else '夹持负载到达目标'
    text(d,(58,54),f'{ordinal} / PIPER {task.upper()}',25,fill=BLUE,bold=True)
    text(d,(58,102),f'{task}：{subtitle}',51,bold=True)
    text(d,(58,181),'几何覆盖训练使用三档球形障碍半径；最大训练半径为 55 mm。',26,fill=MUTED)
    for i,radius in enumerate(TRAINED):
        x=64+390*i
        d.rounded_rectangle((x,256,x+366,472),radius=17,fill='white',outline='#c9dbe8')
        rr=radius*.94
        d.ellipse((x+183-rr,351-rr,x+183+rr,351+rr),fill='#edb6ba',outline='#b6434a',width=3)
        text(d,(x+183,435),f'{radius} mm',29,bold=True,anchor='mm')
    radii=' → '.join(str(x) for x in SETTINGS[task][1])
    text(d,(58,511),f'评测顺序（半径 mm）：{radii}',25,bold=True)
    text(d,(58,562),'冻结模型 · 同起点、目标和障碍中心 · 同为无噪声部署 · 安全阈值 15 mm',22,fill=MUTED)
    text(d,(58,609),'随后展示一个原有布局的回放；每档另有 20 个固定布局，初始化拒绝单列。',22,fill=MUTED)
    return im


def round_frame(task, radius, demo, data, stats, tick):
    im,d=base()
    radii=SETTINGS[task][1]
    scope='训练覆盖内' if radius in TRAINED else '超出训练最大半径'
    text(d,(15,7),f'Piper {task}  /  障碍半径 {radius:g} mm',31,bold=True)
    text(d,(15,49),f'第 {radii.index(radius)+1}/{len(radii)} 档 · {scope} · 回放布局 {demo} · 半速播放',18,fill=MUTED)
    for i,(x,label,clip) in enumerate(zip((12,644),('SAFEFlow','Metric-only'),data)):
        if clip is None:
            v=Image.new('RGB',(624,480),'white')
            a=ImageDraw.Draw(v)
            text(a,(16,10),label,25,bold=True)
            text(a,(312,210),'原布局未通过初始化检查',26,fill=AMBER,bold=True,anchor='mm')
            text(a,(312,265),'未运行策略；不算途中失败',18,fill=MUTED,anchor='mm')
        else:
            v=panel(clip,label,tick)
            elevation(v,clip,min(tick,len(clip['points'])-1))
        im.paste(v,(x,76))
        r=stats[task,radius,label]
        text(d,(x+6,570),f'固定 20 布局：安全到达 {r["safe_success"]}  |  安全停止 {r["safety_stop"]}',19,bold=True)
        text(d,(x+6,609),f'超时 {r["timeout"]}  |  原布局初始化拒绝 {r["reset_rejected"]}',18,fill=MUTED)
    text(d,(15,673),'红：障碍   绿：目标   蓝：末端／负载轨迹   黄：负载（Payload）   ·   安全停止不等于碰撞',17,fill=MUTED)
    return im


def recap(task, stats):
    im,d=base()
    text(d,(48,30),f'{task}  /  固定 20 布局汇总',37,bold=True)
    text(d,(48,90),'两方法使用同一批布局。拒绝初始化的布局保留在 20 个尝试布局里。',21,fill=MUTED)
    for x,label in zip((65,420,730,1053),('障碍半径','SAFEFlow 到达','Metric-only 到达','初始化拒绝')):
        text(d,(x,159),label,21,fill=BLUE,bold=True)
    for i,r in enumerate(SETTINGS[task][1]):
        y=206+i*56
        d.rounded_rectangle((48,y,1232,y+48),radius=8,fill='white' if r in TRAINED else '#dceaf5')
        l,b=(stats[task,r,m] for m in ('SAFEFlow','Metric-only'))
        for x,s in zip((65,420,730,1053),(f'{r:g} mm',f'{l["safe_success"]} / 20',
                       f'{b["safe_success"]} / 20',str(l['reset_rejected']))):
            text(d,(x,y+9),s,22,bold=True)
    text(d,(48,638),'浅蓝：未训练半径。探索性小样本、每任务一个已选训练 seed，不等于正式总体成功率。',20,fill=MUTED)
    return im


def encode(path, tasks, stats, preview):
    with subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n',
            '-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','25','-i','-','-an',
            '-threads','2','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',
            str(path)],stdin=subprocess.PIPE) as encoder:
        def emit(im, repeats):
            bits=im.tobytes()
            for _ in range(repeats):
                encoder.stdin.write(bits)
        for task,(demo,radii) in SETTINGS.items():
            emit(training_card(task),113)
            for radius in radii:
                pair=[tasks[task][radius,m] for m in ('SAFEFlow','Metric-only')]
                end=max((len(c['frames'])-1 for c in pair if c is not None),default=0)
                if end==0:
                    emit(round_frame(task,radius,demo,pair,stats,0),65)
                    continue
                for tick in range(end+1):
                    im=round_frame(task,radius,demo,pair,stats,tick)
                    if task=='Reach' and radius==70 and tick==end:
                        im.save(preview)
                    emit(im,15 if tick==0 else 40 if tick==end else 2)
            emit(recap(task,stats),150)
        encoder.stdin.close()
        if encoder.wait()!=0:
            raise RuntimeError('Video encoding failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=json.loads(PLAN.read_text())
    root=Path(spec['data_root']).resolve(strict=True)
    if (not args.output.resolve().is_relative_to(root) or args.output.resolve()==root
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1
            or shutil.disk_usage(root).free<41*1024**3):
        raise ValueError('Use the admitted data disk and one shared GPU, preserving storage headroom')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.025,torch.device('cuda:0'))
    assets=json.loads(Path('/home/carla/4T/SAFEFLOW2027_PAPER/analysis/piper/assets-ac41fcb.json').read_text())
    records, clips, identities = [], {}, {}
    for task in SETTINGS:
        rows,clips[task],identities[task]=evaluate_task(root,assets,task,spec['tasks'][task])
        records.extend(rows)
    output(args.output/'rollouts.csv',records)
    stats=aggregate(records)
    output(args.output/'summary.csv',list(stats.values()))
    encode(args.output/(NAME+'.mp4'),clips,stats,args.output/(NAME+'.png'))
    (args.output/'manifest.json').write_text(json.dumps(dict(
        scope='Exploratory paired fixed-layout radius sweep, separate from the original frozen test',
        original_checkpoint_plan_sha256=sha256(PLAN),training_radii_mm=TRAINED,
        evaluation_mode='zero_noise',selected_cases=identities,
        per_task={t:dict(radii_mm=r, demo_layout_seed=s, total_batch_layouts=20,
                         batch_layout_seeds=batch_layouts(t,spec['tasks'][t])) for t,(s,r) in SETTINGS.items()},
        reset_policy='Match start joint state, tip, goal and obstacle center to the original 45 mm scene',
        total_batch_policy_layout_attempts=sum(not r['shown_in_video'] for r in records),
        video_demo_attempts=sum(r['shown_in_video'] for r in records),
        video_sha256=sha256(args.output/(NAME+'.mp4')),new_training_steps=0,
        physical_robot_interactions=0),indent=2)+'\n')


if __name__=='__main__':
    main()
