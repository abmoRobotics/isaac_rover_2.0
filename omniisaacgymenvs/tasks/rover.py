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

import numpy as np
import torch
import utils.terrain_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omniisaacgymenvs.robots.articulations.rover import Rover
from omniisaacgymenvs.robots.articulations.views.rover_view import RoverView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
from omniisaacgymenvs.tasks.utils.rover_terrain import *
from omniisaacgymenvs.utils.kinematics import Ackermann
from omniisaacgymenvs.utils.terrain_utils.terrain_generation import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from scipy.spatial.transform import Rotation as R


class RoverTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._device = 'cuda:0'
        self._sim_config = sim_config

        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._rover_positions = torch.tensor([0.0, 0.0, 1.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self._num_observations = 4
        self._num_actions = 2
        self._ball_position = torch.tensor([0, 0, 1.0])
        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 0
        self.stone_info = utils.terrain_utils.read_stone_info("/home/decamargo/Desktop/stone_info.npy")
        self.shift = 5
        # self._rover_position = torch.tensor([0, 0, 2])
        

        # self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # self.marker_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))
                # Previous actions and torques
        self.actions_nn = torch.zeros((self.num_envs, self._num_actions, 3), device=self.device)
        RLTask.__init__(self, name, env)
        return


    def _create_trimesh(self):
        # terrain_width = 50 # terrain width [m]
        # terrain_length = terrain_width # terrain length [m]
        # horizontal_scale = 0.05 #0.025
        #  # resolution per meter 
        # self.heightfield = np.zeros((int(terrain_width/horizontal_scale), int(terrain_length/horizontal_scale)), dtype=np.int16)
        # vertical_scale = 0.005 # vertical resolution [m]
        # def new_sub_terrain(): return SubTerrain1(width=terrain_width,length=terrain_length,horizontal_scale=horizontal_scale,vertical_scale=vertical_scale)
        # terrain = gaussian_terrain(new_sub_terrain(),0.5,0.0)
        # terrain = gaussian_terrain(terrain,15,5)
        # vertices, triangles = convert_heightfield_to_trimesh1(self.heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=None)
        # rock_heigtfield, self.rock_positions = add_rocks_terrain(terrain=terrain)
        # self.heightfield[0:int(terrain_width/horizontal_scale),:] = rock_heigtfield.height_field_raw
        # vertices, triangles = convert_heightfield_to_trimesh1(self.heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=None)
        vertices, triangles = load_terrain('terrainTest.ply')

        self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-35 , -35, 0.0])
        add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)  
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        
    # Sets up the scene for the rover to navigate in
    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_rover()    # Gets the rover asset/articulations
        self.get_target()   # Gets the target sphere (only used for visualizing the target)
        self.get_terrain()
        super().set_up_scene(scene)
        self._rover = RoverView(prim_paths_expr="/World/envs/.*/Rover", name="rover_view")  # Creates an objects for the rover
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False)   # Creates an object for the sphere
        scene.add(self._balls)  # Adds the sphere to the scene
        scene.add(self._rover)  # Adds the rover to the scene
        #self._rover.initialize()
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

    def get_terrain(self):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if not self.curriculum: self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
        #self.terrain_levels = torch.randint(0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        #self.terrain_types = torch.randint(0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        self._create_trimesh()  
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)


    def get_observations(self) -> dict:
        # This function is used for calculating the observations/input to the rover.
        pass

    def pre_physics_step(self, actions) -> None:
        print(self._rover.get_default_state()[0])
        #print()
        # Get the environemnts ids of the rovers to reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # Reset rovers 
            self.reset_idx(reset_env_ids)
            # Reset goal targets
            self.set_targets1(reset_env_ids)
        
        # Get action from model    
        _actions = actions.to(self._device)

        # Code for running ExoMy in Ackermann mode
        _actions[:,0] = _actions[:,0] * 3
        _actions[:,1] = _actions[:,1] * 3
        self.actions_nn = torch.cat((torch.reshape(_actions,(self.num_envs, self._num_actions, 1)), self.actions_nn), 2)[:,:,0:3]
        steering_angles, motor_velocities = Ackermann(_actions[:,0], _actions[:,1])
        
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
        positions[:, 0] = 0 # Position of the front right(FR) motor.
        positions[:, 1] = 0 # Position of the rear right(RR) motor.
        positions[:, 2] = 0 # Position of the front left(FL) motor.
        positions[:, 3] = 0 # Position of the rear left(FL) motor.
        velocities[:, 0] = -6.28/3 # Velocity FR
        velocities[:, 1] = -6.28/3 # Velocity CR
        velocities[:, 2] = -6.28/3 # Velocity RR
        velocities[:, 3] = -6.28/3 # Velocity FL
        velocities[:, 4] = -6.28/3 # Velocity CL
        velocities[:, 5] = -6.28/3 # Velocity RL

        # Set position of the steering motors
        self._rover.set_joint_position_targets(positions,indices=None,joint_indices=self._rover.actuated_pos_indices)
        # Set velocities of the drive motors
        self._rover.set_joint_velocity_targets(velocities,indices=None,joint_indices=self._rover.actuated_vel_indices)

    def reset_idx(self, env_ids):

        num_resets = len(env_ids)
        # dof_pos = self.default_dof_pos[env_ids]
        # dof_vel = velocities
        # # randomize DOF positions
        reset_pos = torch.zeros((num_resets, 3), device=self._device)
        r = []
        # Generates a random orientation for each rover
        for i in range(num_resets):
            #r.append(R.from_euler('xyz', [(random.random() * 2 * math.pi), 0, 3.14], degrees=False).as_quat())
            r.append(R.from_euler('xyz', [0, 0, 0], degrees=False).as_quat())

        reset_orientation = torch.cuda.FloatTensor(r) # Convert quartenion to tensor

        dof_pos = torch.zeros((num_resets, self._rover._num_pos_dof), device=self._device)
        dof_vel = torch.zeros((num_resets, self._rover._num_vel_dof), device=self._device)

        
      
        # # apply resets    
        # indices = env_ids.to(dtype=torch.int32)
        # self._rovers.set_joint_positions(dof_pos, indices=indices)
        # self._rovers.set_joint_velocities(dof_vel, indices=indices)

        # Indicies of rovers to reset
        indices = env_ids.to(dtype=torch.int32) 
        # Set the position/orientation of the rover after reset
        self._rover.set_world_poses(self.initial_root_pos[env_ids].clone(), reset_orientation, indices)

        # Book keeping 
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_reset(self):
        self.initial_root_pos, self.initial_root_rot = self._rover.get_world_poses()
        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses()

    def calculate_metrics(self) -> None:
        # Function for calculating the reward functions
        zero_reward = torch.zeros_like(self.reset_buf,dtype=torch.float)
        # cart_pos = self.obs_buf[:, 0]
        # cart_vel = self.obs_buf[:, 1]
        # pole_angle = self.obs_buf[:, 2]
        # pole_vel = self.obs_buf[:, 3]

        # reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
    
        # self.rew_buf[:] = reward
            # Motion constraint - No oscillatins
        penalty1 = torch.where((torch.abs(self.actions_nn[:,0,0] - self.actions_nn[:,0,1]) > 0.05), torch.square(torch.abs(self.actions_nn[:,0,0] - self.actions_nn[:,0,1])),zero_reward)
        penalty2 = torch.where((torch.abs(self.actions_nn[:,1,0] - self.actions_nn[:,1,1]) > 0.05), torch.square(torch.abs(self.actions_nn[:,1,0] - self.actions_nn[:,1,1])),zero_reward)
        motion_contraint_penalty =  torch.pow(penalty1,2) * (-0.01)#rew_scales["motion_contraint"]
        motion_contraint_penalty = motion_contraint_penalty+(torch.pow(penalty2,2)) * (-0.01)#rew_scales["motion_contraint"]


        reward_total = motion_contraint_penalty

        self.rew_buf[:] = reward_total

    def generate_goals(self,env_ids,radius=3):
        valid_goal = False
        while (not valid_goal):
            self.random_goals(env_ids, radius=radius) # Generate random goals
            valid_goal = True
            #env_ids, reset_buf_len = self.check_goal_collision(env_ids) # Check if goals collides with random rocks

    def random_goals(self, env_ids, radius):
        num_sets = len(env_ids)
        alpha = 2 * math.pi * torch.rand(num_sets, device=self.device)
        TargetRadius = radius
        TargetCordx = 0
        TargetCordy = 0
        x = TargetRadius * torch.cos(alpha) + TargetCordx
        y = TargetRadius * torch.sin(alpha) + TargetCordy
        self.target_root_positions[env_ids, 0] = x #+ self.spawn_offset[env_ids, 0]
        self.target_root_positions[env_ids, 1] = y #+ self.spawn_offset[env_ids, 1]

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        self.generate_goals(env_ids, radius=3) # Generate g     oals
        global_pos = self.target_root_positions[env_ids, 0:2].add(self.env_origins_tensor[env_ids, 0:2])
        #height_offset = height_lookup(self.tensor_map, global_pos, self.horizontal_scale, self.vertical_scale, self.shift, global_pos, torch.zeros(num_sets, 3), self.exo_depth_points_tensor)
        self.target_root_positions[env_ids, 2] = 0#height_offset
        self.marker_positions[env_ids] = self.target_root_positions[env_ids] 
        actor_indices = self.all_actor_indices[env_ids, 1].flatten()
        #self.gym.set_actor_root_state_tensor_indexed(self.sim,self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_sets)

        return actor_indices

    def set_targets1(self, env_ids):
        # Function for generating random goals
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # set target position randomly with x, y in (-1, 1) and z in (1, 2)
        self.target_positions[envs_long, 0:2] = torch.rand((num_sets, 2), device=self._device) * 2 - 1
        self.target_positions[envs_long, 2] = torch.rand(num_sets, device=self._device) + 1

        # shift the target up so it visually aligns better
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        ball_pos[:, 2] += 0.4
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

    def is_done(self) -> None:
        # Function that checks whether or not the rover should reset

        # cart_pos = self.obs_buf[:, 0]
        # pole_pos = self.obs_buf[:, 2]

        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        #resets = torch.zeros((self._num_envs, 1), device=self._device)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, 0)
        self.reset_buf[:] = resets


    def check_spawn_collision(self):
        initial_root_states = self._rover_positions
        old_initial_root_states = None
        while not initial_root_states == old_initial_root_states:
            # what is the purpose of env_origins here? -> can it get removed?
            self._rover_positions[:, 0:2] = initial_root_states[:,0:2] - self.shift #.add(self.env_origins_tensor[:,0:2]) - self.shift 
            old_initial_root_states = initial_root_states
            dist_rocks = torch.cdist(self._rover_positions[:,0:2], self.stone_info[:,0:2], p=2.0)  # Calculate distance to center of all rocks
            dist_rocks[:] = dist_rocks[:] - self.stone_info[:,6]                               # Calculate distance to nearest point of all rocks
            nearest_rock = torch.min(dist_rocks,dim=1)[0]                                   # Find the closest rock to each robot
            initial_root_states[:,0] = torch.where(nearest_rock[:] <= 0.6,initial_root_states[:,0]+0.10,initial_root_states[:,0])
