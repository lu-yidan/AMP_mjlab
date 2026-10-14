from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class PiCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.351] # x,y,z [m], updated to match Piwaist
        rot = [0.0, -1, 0, 1.0] # x,y,z,w [quat]
        target_joint_angles = { # = target angles [rad] when action = 0.0
            # left leg (6 dof)
            "l_hip_pitch_joint": -0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": -0.0,
            "l_ankle_roll_joint": 0,
            # right leg (6 dof)
            "r_hip_pitch_joint": -0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": -0.0,
            "r_ankle_roll_joint": 0,
        }

        default_joint_angles = {
            # left leg (6 dof)
            "l_hip_pitch_joint": -0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": -0.0,
            "l_ankle_roll_joint": 0,
            # right leg (6 dof)
            "r_hip_pitch_joint": -0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": -0.0,
            "r_ankle_roll_joint": 0,
        } 

    class env(LeggedRobotCfg.env):
        num_one_step_observations= 43#3+3+3*12+1
        num_actions = 12
        num_dofs = 12
        num_actor_history = 6
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 10 # episode length in seconds
        unactuated_timesteps = 30

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
            "hip_pitch": 30,
            "hip_roll": 15,
            "thigh": 15,
            "calf": 30,
            "ankle_pitch": 12,
            "ankle_roll": 5,
        }  # [N*m/rad]
        damping = {
            "hip_pitch": 0.2,
            "hip_roll": 0.2,
            "thigh": 0.2,
            "calf": 0.2,
            "ankle_pitch": 0.2,
            "ankle_roll": 0.2,
        }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionRescale * action + cur_dof_pos
        action_scale = 1#0.25#1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.7
        restitution = 0.3
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 1 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        terrain_proportions = [1, 0., 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof/urdf/pi_12dof_release_v1.urdf"
        name = "Pi"
        left_foot_name = "l_ankle_pitch"
        right_foot_name = "r_ankle_pitch"
        left_knee_name = 'l_calf'
        right_knee_name = 'r_calf'
        foot_name = "ankle_roll"
        penalize_contacts_on = ['calf', 'hip']
        terminate_after_contacts_on = []    #'torse'


        left_leg_joints = [ 'l_hip_pitch_joint', 'l_hip_roll_joint','l_thigh_joint', 'l_calf_joint', 'l_ankle_pitch_joint', 'l_ankle_roll_joint']
        right_leg_joints = [  'r_hip_pitch_joint','r_hip_roll_joint', 'r_thigh_joint','r_calf_joint', 'r_ankle_pitch_joint', 'r_ankle_roll_joint']
       
        left_hip_joints = ['l_thigh_joint']
        right_hip_joints = ['r_thigh_joint']
        left_hip_roll_joints = ['l_hip_roll_joint']
        right_hip_roll_joints = ['r_hip_roll_joint']    
        left_hip_pitch_joints = ['l_hip_pitch_joint']
        right_hip_pitch_joints = ['r_hip_pitch_joint']    

        

        left_knee_joints = ['l_calf_joint']
        right_knee_joints = ['r_calf_joint']    

        left_arm_joints = ['l_shoulder_pitch_joint', 'l_shoulder_roll_joint', 'l_shoulder_yaw_joint', 'l_elbow_joint', 'l_wrist_roll_joint']
        right_arm_joints = ['r_shoulder_pitch_joint', 'r_shoulder_roll_joint', 'r_shoulder_yaw_joint', 'r_elbow_joint', 'r_wrist_roll_joint']
        waist_joints = ["waist_yaw_joint"]
        knee_joints = ['l_calf_joint', 'r_calf_joint']
        ankle_joints = [ 'l_ankle_pitch_joint', 'l_ankle_roll_joint', 'r_ankle_pitch_joint', 'r_ankle_roll_joint']

        keyframe_name = "keyframe"
        head_name = 'keyframe_head'

        trunk_names = ["base_link"]
        base_name = 'base_link'

      
        left_lower_body_names = ['l_hip_pitch', 'l_ankle_roll', 'l_calf']
        right_lower_body_names = ['r_hip_pitch', 'r_ankle_roll', 'r_calf']

        left_ankle_names = ['l_ankle_roll']
        right_ankle_names = ['r_ankle_roll']

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_target = 0.34  # updated to match Piwaist
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        orientation_sigma = 1
        is_gaussian = True
        target_head_height = 0.37  # updated to match Piwaist head_height_target (base_height + 0.08)
        target_head_margin = 0.37
        target_base_height_phase1 = 0.25  # updated to match Piwaist
        target_base_height_phase2 = 0.25 #0.05 updated to 0.05 to get better standing style
        target_base_height_phase3 = 0.34  # updated to match Piwaist
        orientation_threshold = 0.99
        left_foot_displacement_sigma = -2#-200 updated to get better standing style
        right_foot_displacement_sigma = -2#-200 updated to get better standing style
        target_dof_pos_sigma = -0.1
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)

        reward_groups = ['task', 'regu', 'style', 'target']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [2.5, 0.1, 1, 1]

        class scales:
            task_orientation = 1
            task_head_height = 1

    class constraints( LeggedRobotCfg.rewards ):
        is_gaussian = True
        target_head_height = 0.37
        target_head_margin = 0.37
        orientation_height_threshold = 0.9
        target_base_height = 0.34  # updated to match Piwaist

        left_foot_displacement_sigma = -2#-200 updated to get better standing style
        right_foot_displacement_sigma = -2#-200 updated to get better standing style
        hip_yaw_var_sigma = -2
        target_dof_pos_sigma = -0.1
        post_task = False
        
        class scales:
            # regularization reward
            regu_dof_acc = -2.5e-7
            regu_action_rate = -0.01
            regu_smoothness = -0.01 
            regu_torques = -2.5e-6
            regu_joint_power = -2.5e-5
            regu_dof_vel = -1e-3
            regu_joint_tracking_error = -0.00025
            regu_dof_pos_limits = -100.0
            regu_dof_vel_limits = -1 

            # style reward
            # style_waist_deviation = -10
            style_hip_yaw_deviation = -10
            style_hip_roll_deviation = -10
            # style_shoulder_roll_deviation = -2.5
            style_left_foot_displacement = 2.5 #7.5 updated to get better standing style
            style_right_foot_displacement = 2.5 #7.5  updated to get better standing style
            style_knee_deviation = -0.25
            # style_shank_orientation = 10
            style_ground_parallel = 20
            style_feet_distance = -10
            style_style_ang_vel_xy = 1
            # style_soft_symmetry_action=-10  #  updated to get better standing style
            # style_soft_symmetry_body=2.5 # updated to get better standing style

            # post-task reward
            target_ang_vel_xy = 10
            target_lin_vel_xy = 10
            target_feet_height_var = 2.5
            # target_target_lower_dof_pos = 30  #  updated to get better standing style
            target_target_orientation = 10
            target_target_base_height = 10
            # target_target_knee_angle = 10 #  updated to get better standing style

    class domain_rand:
        use_random = True

        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]

        randomize_payload_mass = use_random
        payload_mass_range = [-2, 3]

        randomize_com_displacement = use_random
        com_displacement_range = [-0.03, 0.03]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = use_random
        friction_range = [0.1, 1]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        
        randomize_kp = use_random
        kp_range = [0.85, 1.15]
        
        randomize_kd = use_random
        kd_range = [0.85, 1.15]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = True 
        push_interval_s = 10
        max_push_vel_xy = 0.5

        delay = use_random
        max_delay_timesteps = 5
    
    class curriculum:
        pull_force = True
        force = 15 # 100*2=200 is the actuatl force because of a extra keyframe torso link
        dof_vel_limit = 300
        base_vel_limit = 20
        threshold_height = 0.37
        no_orientation = False

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class PiCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256]
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'Pi_ground'
        algorithm_class_name = 'PPO'
        init_at_random_ep_len = True
        max_iterations = 12000 # number of policy updates