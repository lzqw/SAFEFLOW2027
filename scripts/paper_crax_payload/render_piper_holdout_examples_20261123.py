"""Render selected, paired Piper holdout examples from the original frozen policies."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as bullet
import torch

from scripts.paper_crax_payload.evaluate_selected_seed_holdout_20261123 import PLAN, load_case
from scripts.paper_crax_payload.evaluate_piper_deployment_modes_v12 import deployment_mode
from scripts.paper_crax_payload.native_p5_adaptation import target_deployment
from scripts.paper_crax_payload.piper_train import make_piper_env
from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng


EXAMPLES = (
    ('Reach', 'radius_060', 12310075, 'reach_60mm_comparison'),
    ('Payload', 'radius_0575', 12320005, 'payload_57_5mm_comparison'),
    ('Payload', 'radius_0575', 12320025, 'payload_57_5mm_safeflow_failure'),
)
PANEL = (624, 480)
CAM_HEIGHT = 344
BG, INK, MUTED = '#f4f7fb', '#142538', '#41576c'
GREEN, RED, CYAN = '#087f5b', '#af3038', '#157eae'
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), size)


class Scene:
    """Separate display-only Bullet client; never modifies the evaluated simulator."""
    def __init__(self, env, assets):
        self.id = bullet.connect(bullet.DIRECT)
        self.robot = self.call('loadURDF', assets['resolved_urdf'], useFixedBase=True,
                               flags=bullet.URDF_USE_INERTIA_FROM_FILE)
        self.obstacle = self.shape(bullet.GEOM_SPHERE, [.85,.18,.20,1],
                                   radius=env.config.obstacle_radius_m)
        self.goal = self.shape(bullet.GEOM_SPHERE, [.05,.73,.30,1], radius=.012)
        self.payload = (self.shape(bullet.GEOM_BOX, [1.,.63,.08,1],
                                    halfExtents=env.config.payload_half_extents_m)
                        if env.payload is not None else None)
        self.shape(bullet.GEOM_BOX, [.88,.91,.95,1], position=[.20,0,-.019], halfExtents=[2,2,.015])
        self.view = bullet.computeViewMatrix([.91,-.74,.65],[.20,0,.24],[0,0,1])
        self.projection = bullet.computeProjectionMatrixFOV(41,PANEL[0]/CAM_HEIGHT,.03,4)

    def call(self, name, *a, **kw):
        return getattr(bullet,name)(*a,physicsClientId=self.id,**kw)

    def shape(self, kind, color, position=(0,0,0), **geometry):
        visual = self.call('createVisualShape',kind,rgbaColor=color,**geometry)
        return self.call('createMultiBody',baseMass=0,baseVisualShapeIndex=visual,basePosition=position)

    def frame(self, env):
        for joint in range(env.b('getNumJoints',env.robot)):
            position = env.b('getJointState',env.robot,joint)[0]
            self.call('resetJointState',self.robot,joint,position)
        for obj, pos in [(self.obstacle,env.obstacle_position),(self.goal,env.goal)]:
            self.call('resetBasePositionAndOrientation',obj,pos,[0,0,0,1])
        if self.payload is not None:
            self.call('resetBasePositionAndOrientation',self.payload,
                      *env.b('getBasePositionAndOrientation',env.payload))
        _,_,rgba,_,_=self.call('getCameraImage',PANEL[0],CAM_HEIGHT,self.view,self.projection,
                               lightDirection=[.4,-.5,1],shadow=1,renderer=bullet.ER_TINY_RENDERER)
        return Image.fromarray(np.asarray(rgba,dtype=np.uint8).reshape(CAM_HEIGHT,PANEL[0],4),'RGBA').convert('RGB')

    def close(self):
        self.call('disconnect')


def replay(learner, cfg, assets, task, condition, seed, archived, prefix=False):
    ci,c=next((i,c) for i,c in enumerate(cfg['evaluation']['conditions']) if c['name']==condition)
    env=make_piper_env(cfg,assets,task,c.get('simulation_overrides'))
    scene=Scene(env,assets)
    frames, distances, points = [], [], []
    try:
        first=cfg['evaluation']['development_layout_seed_start']
        with isolated_rng(),target_deployment(learner),deployment_mode(learner,'zero_noise'):
            for layout in range(first if prefix else seed, seed+1):
                rng=cfg['evaluation']['action_sampling_seed']+ci*1000+layout-first
                torch.manual_seed(rng)
                torch.cuda.manual_seed_all(rng)
                obs,reset=env.reset(seed=layout)
                minimum=reset['minimum_distance']
                if layout==seed:
                    points.append(env.task_position().tolist())
                    distances.append(minimum)
                    frames.append(scene.frame(env))
                for _ in range(env.config.horizon):
                    act=learner.action(obs[None])[0]
                    obs,_,done,truncated,info=env.step(np.clip(act*c.get('action_scale',1.),-1.,1.))
                    minimum=min(minimum,info['minimum_distance'])
                    if layout==seed:
                        points.append(env.task_position().tolist())
                        distances.append(minimum)
                        frames.append(scene.frame(env))
                    if done or truncated:
                        break
        same_outcome=(info['safe_success']==archived['safe_success'] and info['violation']==archived['violation'])
        ref=np.asarray(archived['trajectory'])
        error=(float(np.max(np.linalg.norm(np.asarray(points)-ref,axis=1)))
               if len(points)==len(ref) else None)
        if (not same_outcome or error is None or error>.0005
                or abs(minimum-archived['minimum_distance_m'])>.0002):
            if prefix:
                raise ValueError(f'{task} {seed}: rerender did not reproduce archived trajectory ({error=})')
            return None
        return dict(frames=frames, distances=distances, points=points,
                    goal=env.goal.tolist(), obstacle=env.obstacle_position.tolist(),
                    radius=env.config.obstacle_radius_m, dt=env.control_dt,
                    success=bool(info['safe_success']), violation=bool(info['violation']),
                    max_replay_path_error_m=error, archived_minimum_distance_m=archived['minimum_distance_m'])
    finally:
        scene.close()
        env.close()


def inset(draw, data, frame):
    x,y,w,h=408,54+CAM_HEIGHT-152,202,142
    draw.rounded_rectangle((x,y,x+w,y+h),radius=9,fill='white',outline='#c7d6e3')
    draw.text((x+9,y+6),'Top view (XY projection)',font=font(12,True),fill=INK)
    base_x=data['obstacle'][0]
    def xy(p):
        return x+w/2+p[1]*580, y+70-(p[0]-base_x)*440
    ox,oy=xy(data['obstacle'])
    rx,ry=data['radius']*580,data['radius']*440
    draw.ellipse((ox-rx,oy-ry,ox+rx,oy+ry),fill='#f5b5b6',outline=RED,width=2)
    gx,gy=xy(data['goal'])
    draw.ellipse((gx-4,gy-4,gx+4,gy+4),fill=GREEN)
    path=[xy(p) for p in data['points'][:frame+1]]
    if len(path)>1:
        draw.line(path,fill=CYAN,width=3)
    px,py=path[-1]
    draw.ellipse((px-4,py-4,px+4,py+4),fill=CYAN)
    draw.text((x+9,y+h-20),'Blue: task path   Green: goal',font=font(11),fill=MUTED)


def panel(data, label, tick):
    frame=min(tick,len(data['frames'])-1)
    end=frame==len(data['frames'])-1
    outcome='SAFE SUCCESS' if data['success'] else 'SAFETY STOP' if data['violation'] else 'TIMEOUT'
    canvas=Image.new('RGB',PANEL,'white')
    canvas.paste(data['frames'][frame],(0,54))
    dr=ImageDraw.Draw(canvas)
    dr.text((16,13),label,font=font(24,True),fill=INK)
    status=outcome if end else 'IN MOTION'
    color=(GREEN if data['success'] else RED) if end else MUTED
    dr.rounded_rectangle((438,10,608,43),radius=8,fill='#eaf0f5')
    dr.text((523,26),status,anchor='mm',font=font(15,True),fill=color)
    inset(dr,data,frame)
    clearance=data['distances'][frame]*1000
    dr.text((16,411),f'Min. clearance so far: {clearance:.2f} mm',font=font(18,True),
            fill=RED if clearance<15 else INK)
    dr.text((16,443),f'Sim time: {frame*data["dt"]:.2f} s    |    Safety threshold: 15 mm',
            font=font(15),fill=MUTED)
    return canvas


def combined(task, condition, seed, results, tick):
    canvas=Image.new('RGB',(1280,600),BG)
    dr=ImageDraw.Draw(canvas)
    radius='60' if condition=='radius_060' else '57.5'
    dr.text((12,11),f'Piper {task}  /  {radius} mm obstacle radius',font=font(27,True),fill=INK)
    dr.text((12,49),f'Paired layout {seed}  |  Frozen policies, zero noise  |  Playback 0.5x',
            font=font(16),fill=MUTED)
    canvas.paste(panel(results[0],'SAFEFlow',tick),(12,80))
    canvas.paste(panel(results[1],'Metric-only',tick),(644,80))
    dr.text((12,575),'Simulation only   |   Red: obstacle   |   Green: goal'
            + ('   |   Gold: held payload' if task=='Payload' else '')
            +'   |   Motion freezes on each policy\'s terminal frame.',
            font=font(14),fill=MUTED)
    return canvas


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    spec=json.loads(PLAN.read_text())
    root=Path(spec['data_root']).resolve(strict=True)
    if (not args.output.resolve().is_relative_to(root) or os.environ.get('CUDA_VISIBLE_DEVICES')!='0'
            or torch.cuda.device_count()!=1):
        raise ValueError('Use the admitted simulation data disk and shared GPU')
    torch.cuda.set_per_process_memory_fraction(.025,torch.device('cuda:0'))
    assets=json.loads(Path('/home/carla/4T/SAFEFLOW2027_PAPER/analysis/piper/assets-ac41fcb.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=[]
    for task,condition,seed,name in EXAMPLES:
        task_spec=spec['tasks'][task]
        results, sources=[],[]
        for case in task_spec['cases']:
            raw=root/'selected-seed-frozen-holdout-20261123'/task/(case['name']+'.json')
            groups=json.loads(raw.read_text())['conditions']
            archived=next(r for r in next(g for g in groups if g['condition']==condition)['records']
                          if r['seed']==seed)
            learner,cfg,identity=load_case(root,case,task,task_spec['training_seed'])
            cfg['evaluation']['action_sampling_seed']=task_spec['action_seed_start']
            cfg['evaluation']['development_layout_seed_start']=task_spec['layout_seed_start']
            data=replay(learner,cfg,assets,task,condition,seed,archived)
            if data is None:
                data=replay(learner,cfg,assets,task,condition,seed,archived,prefix=True)
            results.append(data)
            sources.append(dict(case=case['name'],checkpoint_sha256=identity['checkpoint_sha256'],
                terminal_safe_success=data['success'],terminal_violation=data['violation'],
                steps=len(data['frames'])-1,minimum_distance_m=data['distances'][-1],
                archived_minimum_distance_m=data['archived_minimum_distance_m'],
                max_replay_path_error_m=data['max_replay_path_error_m']))
            del learner
            torch.cuda.empty_cache()
        clip=args.output/(name+'.mp4')
        end=max(len(r['frames']) for r in results)-1
        with subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n',
                '-f','rawvideo','-pix_fmt','rgb24','-s','1280x600','-r','25','-i','-','-an',
                '-threads','2','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
                str(clip)],stdin=subprocess.PIPE) as encoder:
            for tick in range(end+1):
                img=combined(task,condition,seed,results,tick)
                copies=16 if tick==0 else 35 if tick==end else 2
                frame=img.tobytes()
                for _ in range(copies):
                    encoder.stdin.write(frame)
                if tick==end:
                    img.save(args.output/(name+'.png'))
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError(f'Video encoding failed: {clip}')
        manifest.append(dict(video=clip.name,task=task,condition=condition,layout_seed=seed,
            training_seed=task_spec['training_seed'],sources=sources))
        print(json.dumps(manifest[-1]),flush=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(
        scope='Illustrative selected paired examples rerendered from archived frozen holdout models, not new evaluation statistics',
        examples=manifest,new_training_steps=0,physical_robot_interactions=0),indent=2)+'\n')


if __name__=='__main__':
    main()
