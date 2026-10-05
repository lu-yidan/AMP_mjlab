"""The adaptation arms must differ only in their three geometric reward weights."""
import importlib.util
from dataclasses import asdict
from pathlib import Path
from mjlab.utils.os import dump_yaml

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/train_luyidan_guidance.py'
spec=importlib.util.spec_from_file_location('luyidan_guidance',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_guidance_delta_and_shared_training_contract(tmp_path):
    off,agent_off=module.build_config('off',20261005,4096,20000)
    on,agent_on=module.build_config('on',20261005,4096,20000)
    assert {k for k in on.rewards if on.rewards[k].weight!=off.rewards[k].weight}==set(module.GEOMETRY_TERMS)
    for key in module.GEOMETRY_TERMS:
        assert off.rewards[key].weight==0 and on.rewards[key].weight>0
        off.rewards[key].weight=on.rewards[key].weight
    for key in ('plate_no_progress','prone_lateral_progress','plate_completion','plate_force','a6_joint_stall','a6_foot_motion'):
        assert off.rewards[key].weight==on.rewards[key].weight!=0
    dump_yaml(tmp_path/'off.yaml',asdict(off));dump_yaml(tmp_path/'on.yaml',asdict(on))
    assert (tmp_path/'off.yaml').read_text()==(tmp_path/'on.yaml').read_text()
    agent_off.run_name=agent_on.run_name
    assert asdict(agent_off)==asdict(agent_on)
    assert agent_on.save_interval==500 and agent_on.num_steps_per_env==24
    assert on.episode_length_s==10
    assert on.events['a6_three_scene_curriculum_reset'].params['scene_weights']==(.30,.50,.20)
    assert 'push_robot' in on.events and on.observations['actor'].enable_corruption
    assert on.events['a6_dynamics'].params['enabled']
