"""Compare one frozen Reach policy on the same held-out layout at two radii."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
import numpy as np
import torch

from scripts.paper_crax_payload.evaluate_selected_seed_holdout_20261123 import PLAN, load_case
from scripts.paper_crax_payload.render_piper_holdout_examples_20261123 import (
    BG, CYAN, GREEN, INK, MUTED, RED, font, panel, replay,
)


SEED = 12310075  # Reuse the layout from the previous Reach comparison video.
CONDITIONS = (('nominal', 'Nominal  |  45 mm'), ('radius_060', 'Hard  |  60 mm'))
NAME = 'reach_nominal45_vs_hard60'


def elevation(canvas, data, tick):
    x, y, w, h = 408, 246, 202, 142
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((x,y,x+w,y+h), radius=9, fill='white', outline='#c7d6e3')
    d.text((x+9,y+6), 'Side view (YZ projection)', font=font(12,True), fill=INK)
    oy, oz = data['obstacle'][1:]

    def yz(p):
        return x+w/2+(p[1]-oy)*615, y+83-(p[2]-oz)*445

    cx, cy = yz(data['obstacle'])
    rx, ry = data['radius']*615, data['radius']*445
    d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry), fill='#f5b5b6', outline=RED, width=2)
    gx, gy = yz(data['goal'])
    d.ellipse((gx-4,gy-4,gx+4,gy+4), fill=GREEN)
    path = [yz(p) for p in data['points'][:tick+1]]
    if len(path)>1:
        d.line(path, fill=CYAN, width=3)
    px, py = path[-1]
    d.ellipse((px-4,py-4,px+4,py+4), fill=CYAN)
    peak = 1000*max(p[2] for p in data['points'][:tick+1])
    d.text((x+9,y+h-20), f'Peak tip height: {peak:.1f} mm', font=font(11), fill=MUTED)


def comparison_frame(results, tick):
    canvas = Image.new('RGB', (1280,600), BG)
    d = ImageDraw.Draw(canvas)
    d.text((12,11), 'Piper Reach  /  Nominal vs. hard obstacle', font=font(27,True), fill=INK)
    d.text((12,49), f'Same SAFEFlow checkpoint and layout {SEED}  |  Zero noise  |  Playback 0.5x',
           font=font(16), fill=MUTED)
    for x,label,data in zip((12,644),(c[1] for c in CONDITIONS),results):
        image = panel(data,label,tick)
        elevation(image,data,min(tick,len(data['points'])-1))
        canvas.paste(image,(x,80))
    d.text((12,575), 'Simulation only   |   Red: obstacle   |   Green: goal   |   Blue: end-effector path in inset',
           font=font(14), fill=MUTED)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(PLAN.read_text())
    root = Path(spec['data_root']).resolve(strict=True)
    if (not args.output.resolve().is_relative_to(root)
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1):
        raise ValueError('Use the admitted simulation data disk and one shared GPU')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.025, torch.device('cuda:0'))
    assets = json.loads(Path('/home/carla/4T/SAFEFLOW2027_PAPER/analysis/piper/assets-ac41fcb.json').read_text())
    task = spec['tasks']['Reach']
    case = task['cases'][0]
    learner, cfg, identity = load_case(root, case, 'Reach', task['training_seed'])
    cfg['evaluation']['action_sampling_seed'] = task['action_seed_start']
    cfg['evaluation']['development_layout_seed_start'] = task['layout_seed_start']
    archived = root/'selected-seed-frozen-holdout-20261123'/'Reach'/(case['name']+'.json')
    groups = {g['condition']:g['records'] for g in json.loads(archived.read_text())['conditions']}
    results = []
    for condition, _ in CONDITIONS:
        record = next(r for r in groups[condition] if r['seed']==SEED)
        data = replay(learner, cfg, assets, 'Reach', condition, SEED, record)
        if data is None:
            data = replay(learner, cfg, assets, 'Reach', condition, SEED, record, prefix=True)
        results.append(data)
    a,b = results
    if (not all(np.allclose(a[field],b[field],atol=1e-8,rtol=0) for field in ('goal','obstacle'))
            or not np.allclose(a['points'][0],b['points'][0],atol=1e-8,rtol=0)
            or (a['radius'],b['radius'])!=(.045,.060) or a['dt']!=b['dt']):
        raise ValueError('This comparison requires the same start, target and obstacle center')

    args.output.mkdir(parents=True, exist_ok=False)
    end = max(len(r['frames']) for r in results)-1
    with subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n',
            '-f','rawvideo','-pix_fmt','rgb24','-s','1280x600','-r','25','-i','-','-an',
            '-threads','2','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
            str(args.output/(NAME+'.mp4'))], stdin=subprocess.PIPE) as encoder:
        for tick in range(end+1):
            frame = comparison_frame(results,tick)
            if tick==end:
                frame.save(args.output/(NAME+'.png'))
            bits = frame.tobytes()
            for _ in range(18 if tick==0 else 44 if tick==end else 2):
                encoder.stdin.write(bits)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Video encoding failed')

    manifest = dict(task='Reach', case=case['name'], training_seed=task['training_seed'],
        layout_seed=SEED, checkpoint_sha256=identity['checkpoint_sha256'], deployment_mode='zero_noise',
        playback_speed=.5, video=NAME+'.mp4', physical_robot_interactions=0, new_training_steps=0,
        scope='One held-out example rendered from the frozen policy, not new evaluation statistics',
        conditions=[dict(condition=c[0], obstacle_radius_m=r['radius'], safe_success=r['success'],
            violation=r['violation'], steps=len(r['points'])-1, terminal_time_s=(len(r['points'])-1)*r['dt'],
            minimum_clearance_m=r['distances'][-1], peak_tip_height_m=max(p[2] for p in r['points']),
            max_archived_trajectory_error_m=r['max_replay_path_error_m']) for c,r in zip(CONDITIONS,results)])
    (args.output/(NAME+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
