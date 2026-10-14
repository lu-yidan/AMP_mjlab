import os
import math
import torch
import random
import pickle
from tqdm import tqdm
from legged_gym.utils.math import (
    euler_xyz_to_quat,
    quat_apply_yaw,
    quat_apply_yaw_inverse,
    quat_mul, quat_conjugate,
    quat_mul_yaw_inverse,
    quat_mul_yaw,
    quat_mul,
    quat_apply,
)
# from isaacgym.torch_utils import quat_apply, normalize
import copy
import torch


def load_imitation_dataset(folder, mapping="joint_id.txt", suffix=".npz"):
    filenames = [name for name in os.listdir(folder) if name[-len(suffix):] == suffix]
    
    datatset = {}
    for filename in tqdm(filenames):
        try:
            data = pickle.load(open(os.path.join(folder, filename), 'rb'))
            datatset[filename[:-len(suffix)]] = data
        except:
            print(f"{filename} load failed!!!")
            continue
    dataset_list = list(datatset.values())
    random.shuffle(dataset_list)
    
    lines = open(mapping).readlines()
    lines = [line[:-1].split(" ") for line in lines]
    joint_id_dict = {k: int(v) for v, k in lines}
    return dataset_list, joint_id_dict


class MotionLib:
    def __init__(self, datasets, mapping, dof_names, keyframe_names, fps=30, min_dt=0.1, device="cpu", amp_obs_type='keyframe', frame_skip=1, num_steps=2):
        self.device, self.fps = device, fps
        self.env_fps = 50
        self.num_steps = num_steps
        get_len = lambda x: list(x.values())[0].shape[0]
        datasets = [data for data in datasets if get_len(data) > max(math.ceil(min_dt * fps), 3)]

        self.motion_len = torch.tensor([get_len(data) for data in datasets], dtype=torch.long, device=device)
        self.num_motion, self.tot_len = self.motion_len.shape[0], self.motion_len.sum()
        self.motion_sampling_prob = torch.ones(self.num_motion, dtype=torch.float, device=device)

        self.motion_end_ids = torch.cumsum(self.motion_len, dim=0)
        self.motion_start_ids = torch.nn.functional.pad(self.motion_end_ids, (1, -1), "constant", 0)
        
        self.motion_base_rpy = torch.zeros(self.tot_len, 3, dtype=torch.float, device=device)
        self.motion_base_pos = torch.zeros(self.tot_len, 3, dtype=torch.float, device=device)
        self.motion_base_lin_vel = torch.zeros(self.tot_len, 3, dtype=torch.float, device=device)
        self.motion_base_ang_vel = torch.zeros(self.tot_len, 3, dtype=torch.float, device=device)
        self.motion_dof_pos = torch.zeros(self.tot_len, len(dof_names), dtype=torch.float, device=device)
        self.motion_dof_vel = torch.zeros(self.tot_len, len(dof_names), dtype=torch.float, device=device)
        self.motion_keyframe_pos = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        self.motion_keyframe_rpy = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        self.motion_keyframe_lin_vel = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        self.motion_keyframe_ang_vel = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        
        self.motion_keyframe_pos_local = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=device)
        self.motion_keyframe_quat_local = torch.zeros(self.tot_len, len(keyframe_names), 4, dtype=torch.float, device=device)

        for i, traj in enumerate(tqdm(datasets)):
            start, end = self.motion_start_ids[i], self.motion_end_ids[i]
            self.motion_base_pos[start:end] = torch.tensor(traj["base_position"], dtype=torch.float, device=device)
            self.motion_base_rpy[start:end] = torch.tensor(traj["base_pose"], dtype=torch.float, device=device)   
            self.motion_base_lin_vel[start:end-1] = (self.motion_base_pos[start+1:end] - self.motion_base_pos[start:end-1]) * self.fps
            self.motion_base_ang_vel[start:end-1] = (self.motion_base_rpy[start+1:end] - self.motion_base_rpy[start:end-1]) * self.fps
            self.motion_base_lin_vel[end-1:end] = self.motion_base_lin_vel[end-2:end-1]
            self.motion_base_ang_vel[end-1:end] = self.motion_base_ang_vel[end-2:end-1]
            
            dof_pos = torch.tensor(traj["joint_position"], dtype=torch.float, device=device)
            dof_vel = torch.tensor(traj["joint_velocity"], dtype=torch.float, device=device)
            for j, name in enumerate(dof_names):
                if name in mapping.keys():
                    self.motion_dof_pos[start:end, j] = dof_pos[:, mapping[name]]
                    self.motion_dof_vel[start:end, j] = dof_vel[:, mapping[name]]

            for k, name in enumerate(keyframe_names):
                self.motion_keyframe_pos[start:end, k] = torch.tensor(traj["link_position"][name], dtype=torch.float, device=device)
                self.motion_keyframe_rpy[start:end, k] = torch.tensor(traj["link_orientation"][name], dtype=torch.float, device=device)
                self.motion_keyframe_lin_vel[start:end, k] = torch.tensor(traj["link_velocity"][name], dtype=torch.float, device=device)
                self.motion_keyframe_ang_vel[start:end, k] = torch.tensor(traj["link_angular_velocity"][name], dtype=torch.float, device=device)
            
            self.motion_keyframe_pos[start:end, :, 0:2] -= self.motion_base_pos[start:start+1, None, 0:2]
            self.motion_base_pos[start:end, 0:2] -= self.motion_base_pos[start:start+1, 0:2].clone()

            self.motion_keyframe_pos[start:end, :, 2] -= -0.2
            self.motion_base_pos[start:end, 2] -= -0.2

            # !note: the yaw maybe inaccurate
            local_rotation = euler_xyz_to_quat(self.motion_base_rpy[start:end])[:, None]
            self.motion_keyframe_pos_local[start:end] = quat_apply_yaw_inverse(local_rotation.clone(), self.motion_keyframe_pos[start:end] - self.motion_base_pos[start:end][:, None]) 
            # import ipdb; ipdb.set_trace()
            self.motion_keyframe_quat_local[start:end] = quat_mul_yaw_inverse(local_rotation.clone(), euler_xyz_to_quat(self.motion_keyframe_rpy[start:end]))

        self.amp_obs_type = amp_obs_type
        self.frame_skip = frame_skip

    def check_timeout(self, motion_ids, motion_times):
        return torch.ceil(motion_times * self.fps) > (self.motion_len[motion_ids] - 2)
    
    def sample_motions(self, num):
        return torch.multinomial(self.motion_sampling_prob, num_samples=num, replacement=True)
            
    def sample_time(self, motion_ids, uniform=False):
        if uniform:
            phase = torch.rand(motion_ids.shape, dtype=torch.float, device=self.device)
            return phase * (self.motion_len[motion_ids] - 2) / self.fps
        return torch.zeros(motion_ids.shape, dtype=torch.float, device=self.device)
    
    def get_motion_offset(self):
        return torch.mean(self.motion_dof_pos, dim=0)
    
    def get_motion_scale(self, upper_quantile=0.97):
        upper_dof_pos = torch.quantile(self.motion_dof_pos, upper_quantile, dim=0)
        lower_dof_pos = torch.quantile(self.motion_dof_pos, 1.0 - upper_quantile, dim=0)
        return upper_dof_pos - lower_dof_pos
        
    def get_motion_state(self, motion_ids, motion_times, init_base_pos_xy, init_base_quat):
        init_base_pos = torch.cat([init_base_pos_xy, torch.zeros_like(init_base_pos_xy[..., 0:1]) ], dim=1)
        start_ids = self.motion_start_ids[motion_ids]
        
        timesteps = motion_times * self.fps
        floors, ceils = torch.floor(timesteps).long(), torch.ceil(timesteps).long()
        
        quat = torch.tensor([0, 0, -1, 1], dtype=torch.float, device=self.device).unsqueeze(0)
        quat = quat.repeat(init_base_quat.shape[0],  1)
        init_base_quat = quat_mul(quat, init_base_quat)

        blend_motion = lambda x: self.calc_blend(x, start_ids+floors, start_ids+ceils, timesteps-floors, 1-timesteps+floors)
        blend_motion_for_base_pos = lambda x: quat_apply_yaw(init_base_quat, blend_motion(x)) + init_base_pos
        blend_motion_for_body_pos = lambda x: quat_apply_yaw(init_base_quat[:, None], blend_motion(x)) + init_base_pos[:, None]
        blend_motion_for_base_quat = lambda x: quat_mul_yaw(init_base_quat, euler_xyz_to_quat(blend_motion(x)))
        blend_motion_for_body_quat = lambda x: quat_mul_yaw(init_base_quat[:, None], euler_xyz_to_quat(blend_motion(x)))
        
        motion_dict = dict(
            base_pos=blend_motion_for_base_pos(self.motion_base_pos),
            base_quat=blend_motion_for_base_quat(self.motion_base_rpy),
            base_lin_vel=blend_motion(self.motion_base_lin_vel),
            base_ang_vel=blend_motion(self.motion_base_ang_vel),
            
            dof_pos=blend_motion(self.motion_dof_pos),
            dof_vel=blend_motion(self.motion_dof_vel),
            
            keyframe_pos=blend_motion_for_body_pos(self.motion_keyframe_pos), 
            keyframe_quat=blend_motion_for_body_quat(self.motion_keyframe_rpy),
            keyframe_lin_vel=blend_motion(self.motion_keyframe_lin_vel), 
            keyframe_ang_vel=blend_motion(self.motion_keyframe_ang_vel), 
        )

        return motion_dict
    
    def post_process_motion_state(self, motion_dict):
        quat = torch.tensor([0, 0, -1, 1], dtype=torch.float, device=self.device).unsqueeze(0)
        quat = quat.repeat(motion_dict["base_quat"].shape[0],  1)
        
        for key in ['base_pos', 'base_quat', 'keyframe_pos', 'keyframe_quat']:
            print(key, motion_dict[key].shape, quat.shape)
            motion_dict[key] = quat_apply_yaw(quat, motion_dict[key])

        return motion_dict

    @staticmethod    
    def calc_blend(motion, time0, time1, w0, w1):
        motion0, motion1 = motion[time0], motion[time1]
        new_w0 = w0.reshape(w0.shape + (1,) * (motion0.dim() - w0.dim()))
        new_w1 = w1.reshape(w1.shape + (1,) * (motion1.dim() - w1.dim()))
        return new_w0 * motion0 + new_w1 * motion1
    
    def get_expert_obs(self, batch_size):
        ''' Get amp batchsize 
        '''
        if self.amp_obs_type == 'keyframe':
            motion_ids = torch.randint(0, self.tot_len - 1 - self.frame_skip, (batch_size,), device=self.device)
            motion_pos = self.motion_keyframe_pos_local[motion_ids]
            motion_pos_next = self.motion_keyframe_pos_local[motion_ids + self.frame_skip]
            motion_quat = self.motion_keyframe_quat_local[motion_ids]
            motion_quat_next = self.motion_keyframe_quat_local[motion_ids + self.frame_skip]
            amp_state = torch.cat([motion_pos, motion_quat, motion_pos_next, motion_quat_next], dim=-1).view(batch_size, -1)
            return amp_state
        elif self.amp_obs_type == 'dof_pos':
            # motion_ids = torch.randint(0, self.tot_len - (self.num_steps - 1) - self.frame_skip, (batch_size,), device=self.device)
            # motion_dof = self.motion_dof_pos[motion_ids].view(batch_size, -1)
            
            # ratio = self.fps / self.env_fps
            # for i in range(1, self.num_steps):
            #     # import ipdb; ipdb.set_trace()
            #     floor = torch.floor(motion_ids + i * ratio).long()
            #     ceil = floor + 1
            #     linear_ratio = (i * ratio) % 1
            #     motion_dof_next = motion_dof[floor] * (1 - linear_ratio) + motion_dof[ceil] * linear_ratio
            #     motion_dof = torch.cat([motion_dof, motion_dof_next], dim=-1).view(batch_size, -1)

            # amp_state = motion_dof

            motion_ids = torch.randint(0, self.num_motion, (batch_size,), device=self.device)
            start_ids = self.motion_start_ids[motion_ids]
            end_ids = self.motion_end_ids[motion_ids]
            motion_len = self.motion_len[motion_ids]

            time_in_proportion = torch.rand(batch_size).to(self.device)
            clip_tail_proportion = (self.num_steps / motion_len)
            # import ipdb; ipdb.set_trace()
            time_in_proportion = time_in_proportion.clamp(torch.zeros_like(clip_tail_proportion).to(self.device), 1 - clip_tail_proportion)

            motion_ids = start_ids + torch.floor(time_in_proportion * (end_ids - start_ids)).long()
            motion_dof = self.motion_dof_pos[motion_ids].view(batch_size, -1)
            motion_dof_vel = self.motion_dof_vel[motion_ids].view(batch_size, -1)

            ratio = self.fps / self.env_fps
            for i in range(1, self.num_steps):
                # import ipdb; ipdb.set_trace()
                floor = torch.floor(motion_ids + i * ratio).long()
                ceil = floor + 1
                linear_ratio = (i * ratio) % 1
                motion_dof_next = motion_dof[floor] * (1 - linear_ratio) + motion_dof[ceil] * linear_ratio
                motion_dof_vel_next = motion_dof_vel[floor] * (1 - linear_ratio) + motion_dof_vel[ceil] * linear_ratio 
                motion_dof = torch.cat([motion_dof, motion_dof_vel, motion_dof_next, motion_dof_vel_next], dim=-1).view(batch_size, -1)

            amp_state = motion_dof

            # motion_dof_next = self.motion_dof_pos[motion_ids + self.frame_skip]
            # amp_state = torch.cat([motion_dof, motion_dof_next], dim=-1).view(batch_size, -1)
            '''
            (base lin vel, angular vel) + (inverse yaw), (quat_mul_yaw_inverse) + rot6d, dof_pos, multiple frames
            '''
            return amp_state
        else:
            return

def compute_residual_observations(motion_dict, base_quat, body_pos, body_quat, body_lin_vel, body_ang_vel):
    res_body_pos = quat_apply_yaw_inverse(base_quat[:, None], motion_dict["keyframe_pos"] - body_pos)
    res_body_quat = quat_mul_yaw_inverse(base_quat[:, None], quat_mul(quat_conjugate(body_quat), motion_dict["keyframe_quat"]))
    res_body_lin_vel = quat_apply_yaw_inverse(base_quat[:, None], motion_dict["keyframe_lin_vel"] - body_lin_vel)
    res_body_ang_vel = quat_apply_yaw_inverse(base_quat[:, None], motion_dict["keyframe_ang_vel"] - body_ang_vel) 
    return res_body_pos, res_body_quat, res_body_lin_vel, res_body_ang_vel

import numpy as np

def sigmoid(x, value_at_1):
    scale = np.sqrt(-2 * np.log(value_at_1))
    return torch.exp(-0.5 * (x*scale)**2)


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
    lower, upper = bounds 
    assert lower < upper
    assert margin >= 0

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, sigmoid(d.double(), value_at_margin))
    
    return value