# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
import random
from torch.distributions import Normal
import time
import numpy as np
import torch
import utils.terrain_utils.terrain_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import *
from omni.isaac.core.utils.stage import get_current_stage
from omniisaacgymenvs.robots.articulations.rover import Rover
from omniisaacgymenvs.robots.articulations.views.rover_view import RoverView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
from omniisaacgymenvs.tasks.utils.debug_utils import draw_depth
from omniisaacgymenvs.tasks.utils.rover_utils import *
from omniisaacgymenvs.utils.kinematics import Ackermann2, Ackermann
from omniisaacgymenvs.utils.terrain_utils.terrain_generation import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from scipy.spatial.transform import Rotation as R
from omni.isaac.debug_draw import _debug_draw
from omniisaacgymenvs.tasks.utils.tensor_quat_to_euler import tensor_quat_to_eul
from omniisaacgymenvs.tasks.utils.camera import Camera
from omniisaacgymenvs.tasks.utils.rock_detect import Rock_Detection
from pxr import UsdPhysics, Sdf, Gf, PhysxSchema, UsdGeom, Vt, PhysicsSchemaTools
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts.physicsUtils import *


class Memory():
    def __init__(self,num_envs, num_states, horizon, device) -> None:
        self.tracker = torch.zeros((num_envs, num_states, horizon), device=device)
        self.device = device
        self.num_envs = num_envs
        self.num_states = num_states
        self.horizon = horizon
        
    def get_state(self, timestep):
        data = self.tracker[:,:,timestep]
        
        if (data.shape[1] == 1):
            return data.squeeze(1)
        else:
            return data

    def input_state(self, state): 
        self.tracker = torch.cat((torch.reshape(state,(self.num_envs, self.num_states, 1)), self.tracker), 2)[:,:,0:self.horizon]


class RoverTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self.vis_rocks = False
        self._device = 'cuda:0'
        self.shift = torch.tensor([0 , 0, 0.0],device=self._device)
        self.Camera = Camera(self._device,self.shift,debug=False)
        self.num_exteroceptive = self.Camera.get_num_exteroceptive()
        self.Rock_detector = Rock_Detection(self._device,self.shift,debug=False)
        self.global_step = 0
        # Define action space and observation space
        self._num_proprioceptive = 4
        self._num_observations = self._num_proprioceptive + self.num_exteroceptive
        self._num_actions = 2
        self.curriculum_level = 1
        #self._device = 'cpu'
        self._sim_config = sim_config

        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        
        # self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"]["staticFriction"]
        # self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"]["dynamicFriction"]
        # self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"]["restitution"]


        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._rover_positions = torch.tensor([28.0, 30.0, 0.0])
        self.heightmap = None
        self._reset_dist = self._task_cfg["env"]["resetDist"]
       # self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self.max_episode_length = 3000
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]

        self._ball_position = torch.tensor([0, 0, 1.0])
        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.shift = torch.tensor([0 , 0, 0.0],device=self._device)
        self.stone_info = utils.terrain_utils.terrain_utils.read_stone_info("./tasks/utils/terrain/stone_info.npy")
        if self.vis_rocks:
            self._stone_ball_position = torch.zeros((len(self.stone_info), 3), device=self._device, dtype=torch.float32)
        self.shift2 = 5
        self.position_z_offset = None
        self.Camera= Camera(self._device,self.shift)
        self.stone_prim_path = "/World/stones"
        self.save_teacher_data = self._task_cfg["collect_data"]

        # Setup state trackers
        self.linear_velocity = Memory(num_envs = self._num_envs, num_states = 1, horizon = 3,device=self._device)
        self.angular_velocity = Memory(num_envs = self._num_envs, num_states = 1, horizon = 3,device=self._device)

        # Load reward weights
        self.rew_scales = self._task_cfg["rewards"]


        info = {
                "reset": 0,
                "actions": 2,
                "proprioceptive": self._num_proprioceptive,
                "exteroceptive": self._num_observations - self._num_proprioceptive}
                
        self.student = Student_Inference(self.num_envs, info)
        
        if self.save_teacher_data:
            self.curr_timestep = 0
            self.data_curr_timestep = torch.empty((self.num_envs, self.num_actions + self._num_proprioceptive + self.num_exteroceptive + 1))
            # TODO: set number of timesteps to 300 when running on machine with sufficient VRAM
            self.teacher_dataset = torch.empty((5*30, self.num_envs, self.num_actions + self._num_proprioceptive + self.num_exteroceptive + 1))
            self.dataset_nr = 0
            self.reset_info = torch.zeros((self.num_envs))
        
        # Previous actions and torques
        self.actions_nn = torch.zeros((self.num_envs, self._num_actions, 3), device=self._device)
        RLTask.__init__(self, name, env)
        return

    def _create_trimesh(self):
        vertices, triangles = load_terrain('terrain100000.ply')

        #self.v2, self.t2 = load_terrain('big_stones.ply')
        position = self.shift

        add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)  
        #add_stones_to_stage(stage=self._stage, vertices=v2, triangles=t2, position=position)      
    # Sets up the scene for the rover to navigate in
    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_rover()    # Gets the rover asset/articulations
        self.get_target()   # Gets the target sphere (only used for visualizing the target)
        self.get_terrain()
        if self.vis_rocks:
            for i in range(len(self.stone_info)):
                pos = self.stone_info[i,0:3]
                print("RADIUS:" + str(self.stone_info[i,6].item()))
                self.get_stone_ball(self.stone_info[i,6].item(), i, pos)            
        super().set_up_scene(scene)
        self._rover = RoverView(prim_paths_expr="/World/envs/.*/Rover", name="rover_view")  # Creates an objects for the rover
        positions = self._rover.get_world_poses()[0]
        #positions = positions.cpu()
        self.heightmap = torch.load("tasks/utils/terrain/heightmap_tensor.pt")
        #heightmap = heightmap.to('cpu')
        self.horizontal_scale = 0.025
        self.vertical_scale = 1
        pre_col = positions.clone()
        positions = self.avoid_pos_rock_collision(positions)
        position = self.get_pos_height(self.heightmap, positions[:,0:2],self.horizontal_scale,self.vertical_scale,self.shift[0:2])
        self.position_z_offset = torch.ones(position.shape, device=self._device) * 0.5
        positions[:,2] = torch.add(position, self.position_z_offset)
        self.initial_pos = positions
        self._rover.set_world_poses(self.initial_pos, self._rover.get_world_poses()[1])
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False)   # Creates an object for the sphere
        if self.vis_rocks:
            self._stone_balls = []
            for i in range(len(self.stone_info)):
                b = RigidPrimView(prim_paths_expr="/World/envs/ball" + str(i), name="stone_view_" + str(i), reset_xform_properties=False)
                self._stone_balls.append(b)   # Creates an object for the sphere
                scene.add(self._stone_balls[-1])
        scene.add(self._balls)  # Adds the sphere to the scene
        scene.add(self._rover)  # Adds the rover to the scene
        return

    def get_rover(self):
        # Loads the rover asset
        rover = Rover(prim_path=self.default_zero_env_path + "/Rover", name="Rover", translation=self._rover_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Rover", get_prim_at_path(rover.prim_path), self._sim_config.parse_actor_config("Rover"))

    def get_target(self):
        # Creates a target ball/sphere
        radius = 0.1
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball", 
            translation=self._ball_position, 
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)

    def get_stone_ball(self, r, i, pos):
        # Creates a target ball/sphere
        radius = r
        color = torch.tensor([1, 0, 0])
        stone_ball = DynamicSphere(
            prim_path="/World/envs" + "/ball" + str(i), 
            translation=pos,#self._stone_ball_position, 
            name="stone_ball_" + str(i),
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings("stone_ball_" + str(i), get_prim_at_path(stone_ball.prim_path), self._sim_config.parse_actor_config("stone_ball"))
        stone_ball.set_collision_enabled(False)

    def get_terrain(self):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if not self.curriculum: 
            self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
        self._create_trimesh()  

    def get_observations(self) -> dict:
        # Get root state of rover
        self.rover_positions = self._rover.get_world_poses()[0]
        self.rover_rotation = tensor_quat_to_eul(self._rover.get_world_poses()[1])

        # Calculate

        direction_vector = torch.zeros([self.num_envs, 2], device=self._device)
        direction_vector[:,0] = torch.cos(self.rover_rotation[..., 2]) # x value
        direction_vector[:,1] = torch.sin(self.rover_rotation[..., 2]) # y value
        target_vector = self.target_positions[..., 0:2] - self.rover_positions[..., 0:2]
        self.heading_diff = -torch.atan2(target_vector[:,0] * direction_vector[:,1] - target_vector[:,1]*direction_vector[:,0],target_vector[:,0]*direction_vector[:,0]+target_vector[:,1]*direction_vector[:,1])

        # Get heightmap info
        heightmap, output_pt, sources = self.Camera.get_depths(self.rover_positions,self.rover_rotation)

        # Check which rovers should reset due to rock collision
        rock_dist, rock_pt, rock_sources = self.Rock_detector.get_depths(self.rover_positions,self.rover_rotation, self._rover.get_joint_positions())
        if self.curriculum_level >= 2: 
            self.check_collision(rock_dist)
    
        # This function is used for calculating the observations/input to the rover.
        self.obs_buf[:, 0] = torch.linalg.norm(target_vector,dim=1) / 9.5
        self.obs_buf[:, 1] = (self.heading_diff) / math.pi
        self.obs_buf[:, 2] = self.linear_velocity.get_state(timestep=0) / 3
        self.obs_buf[:, 3] = self.angular_velocity.get_state(timestep=0) / 3
        self.obs_buf[:, self._num_proprioceptive:self._num_observations ] = heightmap / 2 #* 2
        
        # add curr timestep to big tensor
        if self.save_teacher_data:
            self.data_curr_timestep[:,3:] = self.obs_buf
            self.teacher_dataset[self.curr_timestep, :] = self.data_curr_timestep
            self.curr_timestep = self.curr_timestep + 1
            if self.curr_timestep >= self.teacher_dataset.shape[0]:
                self.curr_timestep = 0
                teacher_file = {
                    "info":{
                        "reset": 1,
                        "actions": 2,
                        "proprioceptive": self._num_proprioceptive,
                        "exteroceptive": self._num_observations - self._num_proprioceptive},
                    "data": self.teacher_dataset
                }
                print("Saving teacher dataset batch #" + str(self.dataset_nr))
                print("CURR DATA TS: " + str(self.teacher_dataset[self.curr_timestep, :]))
                torch.save(teacher_file, "teacher_dataset_" + str(self.dataset_nr) + ".pt")
                self.dataset_nr = self.dataset_nr + 1
                self.teacher_dataset = torch.empty(self.teacher_dataset.shape)
                # tensor is full -> save to disk and create new one

        observations = {
            self._rover.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        self.global_step += 1
        self.rover_loc = self._rover.get_world_poses()[0]
        self.rover_rot = tensor_quat_to_eul(self._rover.get_world_poses()[1])
        if self.global_step == 10:

            remove_prim(stage=self._stage, prim="/World/terrain/Mesh")
            vertices, triangles = load_terrain('map.ply')
            vertices_rocks, triangles_rocks = load_terrain('big_stones.ply')
            
            
            position = self.shift
            add_stones_to_stage(stage=self._stage, vertices=vertices_rocks, triangles=triangles_rocks, position=self.shift)
            add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
            self.curriculum_level = 2 
        # Get the environemnts ids of the rovers to reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # Reset rovers 
            self.reset_idx(reset_env_ids)
            # Reset goal targets
            self.set_targets(reset_env_ids)
        
        if self.save_teacher_data:
            self.data_curr_timestep[:,0] = self.reset_info[:]
        # Get action from model    
        _actions = actions.to(self._device)
        self.linear_velocity.input_state(_actions[:,0]) # Track linear velocity
        self.angular_velocity.input_state(_actions[:,1]) # Track angular velocity
        _actions = torch.clamp(_actions,-1.0,1.0)

        a = self.student.act(self.obs_buf)
        #print(a.shape)

        print(_actions[1])
        if self.save_teacher_data:
            self.data_curr_timestep[:,1] = _actions[:,0]
            self.data_curr_timestep[:,2] = _actions[:,1]
        #print(_actions)
        # Track states

        a = a.squeeze()
        print(a[1])
        _actions[0:4] = torch.clamp(a[0:4],-1.0,1.0)
        # Code for running in Ackermann mode
        _actions[:,0] = _actions[:,0] * 9 #1.17 # max speed
        _actions[:,1] = _actions[:,1] * 9 # 1.17#(1.17/0.58) # max speed / distance to wheel furthest away in meters
        

        
        self.actions_nn = torch.cat((torch.reshape(_actions,(self.num_envs, self._num_actions, 1)), self.actions_nn), 2)[:,:,0:3]
        self.actions_nn = self.actions_nn
        steering_angles, motor_velocities = Ackermann2(_actions[:,0], _actions[:,1])
        #steering_angles, motor_velocities = Ackermann(torch.ones_like(_actions[:,0])*0.8*3, -torch.ones_like(_actions[:,1])*3*3)
        
        #steering_angles, motor_velocities = Ackermann2(_actions[:,0]*0.0, abs(_actions[:,1]), self.device)
        # Create a n x 4 matrix for positions, where n is the number of environments/robots
        positions = torch.zeros((self._rover.count, 4), dtype=torch.float32, device=self._device)
        # Create a n x 6 matrix for velocities
        velocities = torch.zeros((self._rover.count, 6), dtype=torch.float32, device=self._device)
        
        positions[:, 0] = steering_angles[:,1] # Position of the front right(FR) motor.
        positions[:, 1] = steering_angles[:,5] # Position of the rear right(RR) motor.
        positions[:, 2] = steering_angles[:,0] # Position of the front left(FL) motor.
        positions[:, 3] = steering_angles[:,4] # Position of the rear left(RL) motor.
        velocities[:, 0] = motor_velocities[:,1] # Velocity FR
        velocities[:, 1] = motor_velocities[:,3] # Velocity CR
        velocities[:, 2] = motor_velocities[:,5] # Velocity RR
        velocities[:, 3] = motor_velocities[:,0] # Velocity FL
        velocities[:, 4] = motor_velocities[:,2] # Velocity CL
        velocities[:, 5] = motor_velocities[:,4] # Velocity RL

        # For debugging
        # positions[:, 0] = 0 # Position of the front right(FR) motor.
        # positions[:, 1] = 0 # Position of the rear right(RR) motor.
        # positions[:, 2] = 0 # Position of the front left(FL) motor.
        # positions[:, 3] = 0 # Position of the rear left(FL) motor.
        # velocities[:, 0] = 6.28/0.5 # Velocity FR
        # velocities[:, 1] = 6.28/0.5 # Velocity CR
        # velocities[:, 2] = 6.28/0.5 # Velocity RR
        # velocities[:, 3] = 6.28/0.5 # Velocity FL
        # velocities[:, 4] = 6.28/0.5 # Velocity CL
        # velocities[:, 5] = 6.28/0.5 # Velocity RL

        # Set position of the steering motors
        self._rover.set_joint_position_targets(positions,indices=None,joint_indices=self._rover.actuated_pos_indices)
        # Set velocities of the drive motors
        self._rover.set_joint_velocity_targets(velocities,indices=None,joint_indices=self._rover.actuated_vel_indices)

    def reset_idx(self, env_ids):

        num_resets = len(env_ids)
        
        if self.save_teacher_data:
            self.reset_info[:] = 0
            self.reset_info[env_ids] = 1
        
        reset_pos = torch.zeros((num_resets, 3), device=self._device)
        r = []

        # Generates a random orientation for each rover
        for i in range(num_resets):
            r.append(R.from_euler('x', random.randint(0,360), degrees=True).as_quat())
            #r.append(R.from_euler('xyz', [0, 0, 0], degrees=False).as_quat())
        reset_orientation = torch.tensor(r,device=self._device) # Convert quartenion to tensor

        indices = env_ids.to(dtype=torch.int32)
        dof_pos = torch.zeros((num_resets, 13), device=self._device)
        dof_vel = torch.zeros((num_resets, 13), device=self._device)
        
        # Apply resets    
        indices = env_ids.to(dtype=torch.int32)
        self._rover.set_joint_positions(dof_pos, indices=indices)
        self._rover.set_joint_velocities(dof_vel, indices=indices)

        self.base_pos[env_ids] = self.initial_pos[env_ids]
        #self.base_pos[env_ids, :] += self.position_z_offset[env_ids, :]

        # Indicies of rovers to reset
        indices = env_ids.to(dtype=torch.int64) 

        # Set the position/orientation of the rover after reset
        self._rover.set_world_poses(self.base_pos[indices], reset_orientation.float(), indices)

        # Book keeping 
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self._device)
        self.initial_root_pos, self.initial_root_rot = self._rover.get_world_poses()
        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses()

    def calculate_metrics(self) -> None:
        # Function for calculating the reward functions

        # Tool tensors 
        zero_reward = torch.zeros_like(self.reset_buf)
        max_reward = torch.ones_like(self.reset_buf)
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # Get states
        lin_vel = self.linear_velocity.get_state(timestep=0)        # Get current action
        lin_vel_prev = self.linear_velocity.get_state(timestep=1)   # Get last action
        ang_vel = self.angular_velocity.get_state(timestep=0)
        ang_vel_prev = self.angular_velocity.get_state(timestep=1)

        # Get rover boogie angles
        boogie_angles = self._rover.get_joint_positions()[:,0:3]

        # Get rover goal heading difference
        heading_diff = self.heading_diff

        # Distance to target
        target_dist = torch.sqrt(torch.square(self.target_positions[..., 0:2] - self.rover_positions[..., 0:2]).sum(-1))

        #draw = _debug_draw.acquire_debug_draw_interface()
        #target_list = self.target_positions[:,0:3].tolist()
        #rover_list = self.rover_positions[:,0:3].tolist()
        #N = len(target_list)
        #draw.clear_lines()
        #draw.draw_lines(target_list, rover_list, [(0.9, 0.5, 0.1, 0.9)]*N, [3]*N)

        # Heading constraint - Avoid driving backwards
        lin_vel = self.linear_velocity.get_state(0)   # Get latest lin_vel
        heading_contraint_penalty = torch.where(lin_vel < 0, -max_reward, zero_reward) * self.rew_scales["heading_contraint_reward"]
        
        if False: #self.global_step < 1000:
            heading_contraint_penalty = torch.where(lin_vel < 0, -max_reward, zero_reward) * self.rew_scales["heading_contraint_reward"]*30

        # Boogie angles - Penalty for driving over uneven terrain
        boogie_penalty = ( torch.abs(boogie_angles[:,0]) + torch.abs(boogie_angles[:,1]) + torch.abs(boogie_angles[:,2]) ) * self.rew_scales["boogie_contraint_reward"]

        # penalty for driving away from the robot
        goal_angle_penalty = torch.where(torch.abs(heading_diff) > 2, -torch.abs(heading_diff*0.3*self.rew_scales['goal_angle_reward']), zero_reward.float())

        # Motion constraint - No oscillatins
        penalty1 = torch.where((torch.abs(lin_vel * 3 - 3 * lin_vel_prev) > 0.05), torch.square(torch.abs(lin_vel * 3 - 3 * lin_vel_prev)),zero_reward.float())
        penalty2 = torch.where((torch.abs(ang_vel * 3 - 3 * ang_vel_prev) > 0.05), torch.square(torch.abs(ang_vel * 3 - 3 * ang_vel_prev)),zero_reward.float())
        motion_contraint_penalty =  torch.pow(penalty1,2) * self.rew_scales["motion_contraint_reward"]
        
        motion_contraint_penalty = motion_contraint_penalty+(torch.pow(penalty2,2)) * self.rew_scales["motion_contraint_reward"]

        # distance to target
        pos_reward = (1.0 / (1.0 + (0.33*0.33*target_dist * target_dist))) * self.rew_scales['pos_reward']
        pos_reward = torch.where(target_dist <= 0.18, 1.03*(self.max_episode_length-self.progress_buf), pos_reward.float())  # reward for getting close to target
        
        #pos_reward = torch.where(target_dist >= 9.5, -0.3*(self.max_episode_length), pos_reward.double())
        
#torch.where(target_dist >= 9.5, ones, resets)
        # Calculate combined reward
        reward = pos_reward + heading_contraint_penalty + motion_contraint_penalty + goal_angle_penalty
        #reward = torch.where(target_dist >= 9.5, -0.3*(self.max_episode_length)*ones, reward)
        if self.curriculum_level >= 2:

            # Collision penalty
            collision_tracker = torch.where(self.rock_collison == 1, max_reward*self.num_envs,zero_reward)

            reward = torch.where(self.rock_collison == 1, reward-300, reward)
        else:
            collision_tracker = zero_reward
        reward = reward / 3000
        self.rew_buf[:] = reward
        self.extras["pos_reward"] = pos_reward
        self.extras["collision_penalty"] = collision_tracker
        self.extras["uprightness_penalty"] = boogie_penalty
        self.extras["heading_contraint_penalty"] = heading_contraint_penalty
        self.extras["motion_contraint_penalty"] = motion_contraint_penalty
        self.extras["goal_angle_penalty"] = goal_angle_penalty
        self.extras["torque_penalty_driving"] = lin_vel # TODO Remove
        self.extras["torque_penalty_steering"] = ang_vel # TODO Remove

    def check_goal_collision(self, env_ids):
        ones = torch.ones_like(env_ids)
        zeros = torch.zeros_like(env_ids)
        dist_rocks = torch.cdist(self.target_positions[env_ids][:,0:2], self.stone_info[:,0:2], p=2.0)  # Calculate distance to center of all rocks
        dist_rocks[:] = dist_rocks[:] - self.stone_info[:,6]  
        nearest_rock = torch.min(dist_rocks,dim=1)[0]
        reset_buf = torch.where(nearest_rock <= 1.0, ones, zeros)
        env_ids = reset_buf * env_ids   # Multiply reset buffer with env_ids in order to get reset ids
        reset_buf_len = len(reset_buf.nonzero(as_tuple=False).squeeze(-1))  # Get number of non-zero values in the reset buffer
        return env_ids, reset_buf_len

    def generate_goals(self,env_ids,radius):
        #self.random_goals(env_ids, radius=radius) # Generate random goals
        reset_buf_len = 1
        while (reset_buf_len > 0):
            self.random_goals(env_ids, radius=radius) # Generate random goals
            env_ids, reset_buf_len = self.check_goal_collision(env_ids) # Check if goals collides with random rocks

        # valid_goals = self.avoid_pos_rock_collision(self.target_positions)
        # self.target_positions = valid_goals

    def random_goals(self, env_ids, radius):
        num_sets = len(env_ids)
        alpha = 2 * math.pi * torch.rand(num_sets, device=self._device)
        TargetRadius = radius #* math.sqrt(random.random())
        TargetCordx = 0
        TargetCordy = 0
        x = TargetRadius * torch.cos(alpha) + TargetCordx
        y = TargetRadius * torch.sin(alpha) + TargetCordy
        # TODO: Add offset of each environement 
        self.target_positions[env_ids, 0] = x + self.initial_pos[env_ids, 0]#self.position_z_offset[env_ids, 0]
        self.target_positions[env_ids, 1] = y + self.initial_pos[env_ids, 1]#self.position_z_offset[env_ids, 1]

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        # if self.global_step < 5000:
        #     radius = 2
        # elif self.global_step < 8000:
        #     radius = 3
        # elif self.global_step < 12000:
        #     radius = 4.5
        # if self.global_step < 16000:
        #     radius = 6
        # else:
        #     radius = 9
        radius = 8
        self.generate_goals(env_ids, radius=radius) # Generate goals
        envs_long = env_ids.long()
        global_pos = self.target_positions[env_ids, 0:2]#.add(self.env_origins_tensor[env_ids, 0:2])
        height= self.get_pos_height(self.heightmap, global_pos[:,0:2], self.horizontal_scale, self.vertical_scale, self.shift[0:2])
        self.target_positions[env_ids, 2] = height
        self._balls.set_world_poses(self.target_positions[envs_long], self.initial_ball_rot[envs_long].clone(), indices=env_ids)



    def get_pos_height(self, heightmap: torch.Tensor, depth_points: torch.Tensor, horizontal_scale, vertical_scale, shift):
        # Scale locations to fit heightmap
        scaledmap = (depth_points-shift)/horizontal_scale
        # Bound values inside the map
        scaledmap = torch.clamp(scaledmap, min = 0, max = heightmap.size()[0]-1)
        # Round to nearest integer
        scaledmap = torch.round(scaledmap)

        # Convert x,y coordinates to two vectors.
        x = scaledmap[:,0]
        y = scaledmap[:,1]
        x = x.type(torch.long)
        y = y.type(torch.long)

        # Lookup heights in heightmap
        heights = heightmap[x, y]
        
        # Scale to fit actual height, dependent on resolution
        heights = heights * vertical_scale

        return heights

    def is_done(self) -> None:
        # Function that checks whether or not the rover should reset
        ones = torch.ones_like(self.progress_buf)
        zeros = torch.zeros_like(self.progress_buf)
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        #resets = torch.zeros((self._num_envs, 1), device=self._device)
        resets = torch.where(self.progress_buf >= self.max_episode_length, ones, zeros)
        resets = torch.where(torch.abs(self.rover_rot[:,0]) >= 0.78*1.5, ones, resets)
        resets = torch.where(torch.abs(self.rover_rot[:,1]) >= 0.78*1.5, ones, resets)
        target_dist = torch.sqrt(torch.square(self.target_positions[..., 0:2] - self.rover_positions[..., 0:2]).sum(-1))
        resets = torch.where(target_dist >= 9.5, ones, resets)
        resets = torch.where(target_dist <= 0.18, ones,resets)

        # Check for collisions when curriculum level is 2 or higher
        if self.curriculum_level >= 2:
            resets = torch.where(self.rock_collison == 1, torch.ones_like(self.reset_buf), resets)
        self.reset_buf[:] = resets

    def avoid_pos_rock_collision(self, curr_pos):
        #curr_pos = self._rover_positions
        old_pos = torch.zeros(curr_pos.shape).cuda()
        while not torch.equal(curr_pos, old_pos):
            # what is the purpose of env_origins here? -> can it get removed?
            # shifted_pos = curr_pos[:,0:2] #- self.shift2 #.add(self.env_origins_tensor[:,0:2]) - self.shift 
            old_pos = curr_pos.clone()
            dist_rocks = torch.cdist(curr_pos[:,0:2], self.stone_info[:,0:2], p=2.0)  # Calculate distance to center of all rocks
            dist_rocks[:] = dist_rocks[:] - self.stone_info[:,6]     
            #print("Dist nearest rocks: " + str(dist_rocks))                          # Calculate distance to nearest point of all rocks
            nearest_rock = torch.min(dist_rocks,dim=1)[0]                             # Find the closest rock to each robot
            #print("Dist nearest rock: " + str(nearest_rock)) 
            curr_pos[:,0] = torch.where(nearest_rock[:] <= 1.4,torch.add(curr_pos[:,0], 0.05),curr_pos[:,0])
        return curr_pos
    



    def check_collision(self, rock_rays):
        #rock_rays = torch.where(rock_rays < -10, torch.ones_like(rock_rays)*99, rock_rays )
        nearest_rock = torch.min(rock_rays,dim=1)[0]            # Find the closest rock to each robot
        self.rock_collison = torch.where(torch.abs(nearest_rock) < 0.3, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))





class Student_Inference():
    def __init__(self, num_envs, info, device='cuda:0') -> None:
        self.cfg = self.cfg_fn()
        self.info = info
        self.model = self.load_model('best.pt')
        self.h = self.model.belief_encoder.init_hidden(num_envs).to(device)
        

    def load_model(self, model_name):
        model = Student(self.info, self.cfg)
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.cuda()

        return model

    def act(self,observations):
        with torch.no_grad():
            #print(observations.shape)
            actions, predictions, self.h = self.model(observations.unsqueeze(1),self.h)
            return actions

    def info_fn(self):
        pass

    def cfg_fn(self):
        cfg = {
            "info":{
                "reset":            0,
                "actions":          0,
                "proprioceptive":   0,
                "exteroceptive":    0,
            },
            "learning":{
                "learning_rate": 1e-4,
                "epochs": 500,
                "batch_size": 8,
            },
            "encoder":{
                "activation_function": "leakyrelu",
                "encoder_features": [80,60]},

            "belief_encoder": {
                "hidden_dim":       50,
                "n_layers":         2,
                "activation_function":  "leakyrelu",
                "gb_features": [64,64,60],
                "ga_features": [64,64,60]},

            "belief_decoder": {
                "activation_function": "leakyrelu",
                "gate_features":    [128,256,512],
                "decoder_features": [128,256,512]
            },
            "mlp":{"activation_function": "leakyrelu",
                "network_features": [256,160,128]},
                }

        return cfg



import torch.nn as nn

class Layer(nn.Module):
    def __init__(self,in_channels,out_channels, activation_function="elu"):
        super(Layer,self).__init__()
        self.activation_functions = {
            "elu" : nn.ELU(),
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" :nn.LeakyReLU(),
            "sigmoid" : nn.Sigmoid(),
            "tanh" : nn.Tanh(),
            "relu6" : nn.ReLU6()
           } 
        self.layer = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
        )
    def forward(self,x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(
            self, info, cfg):
        super(Encoder,self).__init__()
        encoder_features = cfg["encoder_features"]
        activation_function = cfg["activation_function"]
        
        self.encoder = nn.ModuleList() 
        in_channels = info["exteroceptive"]
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

    def forward(self, x):
        
        for layer in self.encoder:
            x = layer(x)
        
        return x

class Belief_Encoder(nn.Module):
    def __init__(
            self, info, cfg, input_dim=60):
        super(Belief_Encoder,self).__init__()
        self.hidden_dim = cfg["hidden_dim"]
        self.n_layers = cfg["n_layers"]
        activation_function = cfg["activation_function"]
        proprioceptive = info["proprioceptive"]
        input_dim = proprioceptive+input_dim
        

        self.gru = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.gb = nn.ModuleList()
        self.ga = nn.ModuleList()
        gb_features = cfg["gb_features"]
        ga_features = cfg["ga_features"]

        in_channels = self.hidden_dim
        for feature in gb_features:
            self.gb.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        in_channels = self.hidden_dim
        for feature in ga_features:
            self.ga.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.ga.append(nn.Sigmoid())

    def forward(self, p, l_e, h):
        #print(l_e.shape)
        # p = proprioceptive
        # e = exteroceptive
        # h = hidden state
        # x = input data, h = hidden state
        
        x = torch.cat((p,l_e),dim=2)
        out, h = self.gru(x, h)
        x_b = x_a = out

        for layer in self.gb:
            x_b = layer(x_b)
        for layer in self.ga:
            x_a = layer(x_a)

        x_a = l_e * x_a
        # TODO IMPLEMENT GATE
        belief = x_b + x_a

        return belief, h, out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cpu')
        return hidden

class Belief_Decoder(nn.Module):
    def __init__(
            self, info, cfg, n_input=50, hidden_dim=50,n_layers=2,activation_function="leakyrelu"):
        super(Belief_Decoder,self).__init__()
        exteroceptive = info["exteroceptive"]
        gate_features = cfg["gate_features"] #[128,256,512, exteroceptive]
        decoder_features = cfg["decoder_features"]#[128,256,512, exteroceptive]
        #n_input = cfg[""]
        gate_features.append(exteroceptive)
        decoder_features.append(exteroceptive)
        self.n_input = n_input
        self.gate_encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
    

        in_channels = self.n_input
        for feature in gate_features:
            self.gate_encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        self.gate_encoder.append(nn.Sigmoid())  

        in_channels = self.n_input
        for feature in decoder_features:
            self.decoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        

    def forward(self, e, h):
        gate = h#h[-1]
        decoded = h#h[-1]
       # gate = gate.repeat(e.shape[1], 1, 1).permute(1,0,2)
       # decoded = decoded.repeat(e.shape[1], 1, 1).permute(1,0,2)
        #print(h.shape)
        for layer in self.gate_encoder:
            gate = layer(gate)

        for layer in self.decoder:
            decoded = layer(decoded)
        x = e*gate
        x = x + decoded
        return x
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(1.0)

class MLP(nn.Module):
    def __init__(
            self, info, cfg, belief_dim):
        super(MLP,self).__init__()
        self.network = nn.ModuleList()  # MLP for network
        proprioceptive = info["proprioceptive"]
        action_space = info["actions"]
        activation_function = cfg["activation_function"]
        network_features = cfg["network_features"]

        in_channels = proprioceptive + belief_dim
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space))
        self.network.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def forward(self, p, belief):
        
        x = torch.cat((p,belief),dim=2)

        for layer in self.network:
            x = layer(x)
        return x, self.log_std_parameter

class Student(nn.Module):
    def __init__(
            self, info, cfg):
        super(Student,self).__init__()

        self.n_re = info["reset"]
        self.n_pr = info["proprioceptive"]
        self.n_ex = info["exteroceptive"]
        self.n_ac = info["actions"]
        
        self.encoder = Encoder(info, cfg["encoder"])
        encoder_dim = cfg["encoder"]["encoder_features"][-1]
        self.belief_encoder = Belief_Encoder(info, cfg["belief_encoder"], input_dim=encoder_dim)
        self.belief_decoder = Belief_Decoder(info, cfg["belief_decoder"])
        self.MLP = MLP(info, cfg["mlp"], belief_dim=60)

    def forward(self, x, h):

        n_ac = 0
        n_pr = self.n_pr
        n_re = self.n_re
        n_ex = self.n_ex
        print(n_ac)
        print(n_pr)
        print(n_re)
        print(n_ex)
        print("STOP")
        reset = x[:,:, 0:n_re]
        actions = x[:,:,n_re:n_re+n_ac]
        proprioceptive = x[:,:,n_re+n_ac:n_re+n_ac+n_pr]
        exteroceptive = x[:,:,-n_ex:]
        
        # n_p = self.n_p
        
        # p = x[:,:,0:n_p]        # Extract proprioceptive information  
        
        # e = x[:,:,n_p:1084]         # Extract exteroceptive information
        
        e_l = self.encoder(exteroceptive) # Pass exteroceptive information through encoder

        belief, h, out = self.belief_encoder(proprioceptive,e_l,h) # extract belief state
        
        #estimated = self.belief_decoder(exteroceptive,h)
        estimated = self.belief_decoder(exteroceptive,out)
        

        actions, log_std = self.MLP(proprioceptive,belief)

        # min_log_std= -20.0
        # max_log_std = 2.0
        # log_std = torch.clamp(log_std, 
        #                         min_log_std,
        #                         max_log_std)
        # g_log_std = log_std
        # # print(actions.shape[0])
        # # print(actions.shape[2])
        # _g_num_samples = actions.shape[0]

        # # # distribution
        # _g_distribution = Normal(actions, log_std.exp())
        # #print(_g_distribution.shape)
        # # # sample using the reparameterization trick
        # actions = _g_distribution.rsample()
        #print((actions-action).mean())

        return actions, estimated, h

