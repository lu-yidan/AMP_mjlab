"""Export velocity actor only; history/command interface differs from legacy AMP."""
import argparse,json
from pathlib import Path
import torch
from .evaluate import load_policy
from .mdp import OBS_DIM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();pi,ck=load_policy(a.checkpoint,'cpu')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(pi,(torch.zeros(1,pi.hist,OBS_DIM),torch.zeros(1,3)),str(out),
        input_names=['observation_history','velocity_command'],output_names=['actions'],
        opset_version=18,dynamo=False,
        dynamic_axes={'observation_history':{0:'batch'},'velocity_command':{0:'batch'},'actions':{0:'batch'}})
    out.with_suffix('.json').write_text(json.dumps(dict(format=ck['format'],history=pi.hist,
        obs_dim=OBS_DIM,order='oldest to newest',command='pelvis yaw-frame vx,vy and world yaw rate; SI units',
        observation='q-default_q (29), qd*0.1 (29), pelvis angular velocity body-frame *0.25 (3), projected gravity (3), previous normalized action (29)',
        joint_names=ck['joint_names'],default_q=ck['default_q'].tolist(),dt=ck['dt'],
        action='motor order normalized offsets, use this repository G1_ACTION_SCALE and default_q; NOT legacy AMP 0.25 interface',
        recovery='Same actor; always pass the raw requested velocity even when fallen. No external mode gate.',
        reset_metadata=ck.get('reset_metadata',{}),source_checkpoint=a.checkpoint),indent=2))


if __name__=='__main__':main()
