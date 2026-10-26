"""Record labelled, side-by-side rollouts of a fixed checkpoint; no resets."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .evaluate import load_policy
from .env import VelocitySim
from .reset_bank import ResetBank,DIRECTIONS
from .mdp import OBS_DIM,observation,is_recovery
from .rotations import yaw_mat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    pi,ck=load_policy(a.checkpoint,'cuda:0');bank=ResetBank(a.bank,ck['joint_names'],'cuda:0')
    selected=[]
    for direction in DIRECTIONS:
        ids=np.flatnonzero((bank.stages=='low')&(bank.sources=='natural')&(bank.labels==direction))
        # Deterministic median-height representative, not selected by policy outcome.
        heights=bank.qpos[torch.tensor(ids,device='cuda:0'),2].cpu().numpy()
        selected.append(int(ids[np.argsort(heights,kind='stable')[len(ids)//2]]))
    from mjlab.viewer.viewer_config import ViewerConfig
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:font=ImageFont.truetype(font_path,18)
    except OSError:font=ImageFont.load_default()
    specs=[('walk_run',[.5,1.,2.,3.],False,12),('fast_run',[4.,5.],False,12),
        ('recovery_stand',[0.]*4,True,10),('recovery_walk',[1.]*4,True,10)]
    report=dict(checkpoint=a.checkpoint,iteration=ck['iteration'],bank_sha256=bank.sha256,bank_rows=selected,
        selection='Natural low poses; median root height within each direction',videos=[])
    for name,speeds,recovery,seconds in specs:
        viewer=ViewerConfig(width=640,height=400,distance=3.0,elevation=-18,azimuth=135,max_extra_envs=0)
        sim=VelocitySim(len(speeds),render=True,reset_mode='standing',viewer_cfg=viewer)
        path=out/(name+'.mp4');frames=0
        try:
            if list(sim.robot.joint_names)!=ck['joint_names'] or not torch.allclose(sim.default_q,ck['default_q']):raise ValueError('Robot interface mismatch')
            if abs(sim.env.step_dt-ck['dt'])>1e-8:raise ValueError('Control timestep mismatch')
            if recovery:sim.reset_to_bank(bank,torch.tensor(selected,device='cuda:0'))
            hist=torch.zeros(sim.n,pi.hist,OBS_DIM,device='cuda:0')
            with imageio.get_writer(path,fps=25,codec='libx264',quality=8,macro_block_size=1) as writer,torch.inference_mode():
                for step in range(round(seconds/sim.env.step_dt)):
                    sim.command.zero_()
                    if recovery or step*sim.env.step_dt>=2:sim.command[:,0]=torch.tensor(speeds,device='cuda:0')
                    o=observation(sim.state(),sim.a_prev,sim.default_q)
                    hist=o[:,None].expand_as(hist).clone() if step==0 else torch.cat((hist[:,1:],o[:,None]),1)
                    if step%2==0:
                        S=sim.state();vx=(yaw_mat(S[1]).transpose(-1,-2)@S[2].unsqueeze(-1)).squeeze(-1)[:,0]
                        canvas=Image.new('RGB',(1280,480 if sim.n==2 else 960),(20,24,30));draw=ImageDraw.Draw(canvas)
                        for i in range(sim.n):
                            sim.env.cfg.viewer.env_idx=i
                            # AUTO is a free camera in this runtime; explicitly track
                            # each panel's root to avoid an empty scene after running.
                            target=sim.robot.data.root_link_pos_w[i].cpu().numpy().copy()
                            target[2]=float(sim.env.scene.env_origins[i,2])+.6
                            sim.env._offline_renderer._cam.lookat[:]=target
                            frame=Image.fromarray(sim.env.render());x=(i%2)*640;y=(i//2)*480
                            canvas.paste(frame,(x,y+80))
                            title=DIRECTIONS[i] if recovery else f'Target {speeds[i]:.1f} m/s'
                            draw.text((x+12,y+5),f'SuperMimic {ck["iteration"]//1000}k | {title}',font=font,fill='white')
                            draw.text((x+12,y+30),f't={step*sim.env.step_dt:.2f}s  cmd={float(sim.command[i,0]):.1f}  vx={float(vx[i]):.2f} m/s',font=font,fill='white')
                            draw.text((x+12,y+55),f'root={float(S[0][i,2]):.2f}m  upright={not bool(is_recovery(S)[i])}  no reset',font=font,fill=(180,210,240))
                        writer.append_data(np.asarray(canvas));frames+=1
                        if step in (0,150,400):canvas.save(out/f'{name}_{step:04d}.jpg')
                    sim.step(pi(hist,sim.command),auto_reset=False,resample=False)
            report['videos'].append(dict(path=path.name,seconds=seconds,frames=frames,commands=speeds,recovery=recovery))
            (out/'recording.json').write_text(json.dumps(report,indent=2));print('COMPLETE '+str(path),flush=True)
        finally:sim.close()


if __name__=='__main__':main()
