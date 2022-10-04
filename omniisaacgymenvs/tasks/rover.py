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


from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.rover import Rover

from omni.isaac.core.articulations import ArticulationView
from omniisaacgymenvs.robots.articulations.views.rover_view import RoverView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.objects import DynamicSphere
import numpy as np
import torch
import math
import random
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

        self._num_observations = 4
        self._num_actions = 2
        self._ball_position = torch.tensor([0, 0, 1.0])
        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 0
        # self._rover_position = torch.tensor([0, 0, 2])
        

        # self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # self.marker_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))

        RLTask.__init__(self, name, env)
        return

    # Sets up the scene for the rover to navigate in
    def set_up_scene(self, scene) -> None:
        self.get_rover()    # Gets the rover asset/articulations
        self.get_target()   # Gets the target sphere (only used for visualizing the target)
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

    def get_observations(self) -> dict:
        # This function is used for calculating the observations/input to the rover.
        pass

    def pre_physics_step(self, actions) -> None:

        # Get the environemnts ids of the rovers to reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # Reset rovers 
            self.reset_idx(reset_env_ids)
            # Reset goal targets
            self.set_targets1(reset_env_ids)
        
            
        #actions = actions.to(self._device)

        # Create a n x 4 matrix for positions, where n is the number of environments/robots
        positions = torch.zeros((self._rover.count, 4), dtype=torch.float32, device=self._device)
        # Create a n x 6 matrix for velocities
        velocities = torch.zeros((self._rover.count, 6), dtype=torch.float32, device=self._device)

        positions[:, 0] = 0 # Position of the front right(FR) motor.
        positions[:, 1] = 0 # Position of the rear right(RR) motor.
        positions[:, 2] = 0 # Position of the front left(FL) motor.
        positions[:, 3] = 0 # Position of the rear left(FL) motor.
        velocities[:, 0] = 6.28/3 # Velocity FR
        velocities[:, 1] = 6.28/3 # Velocity CR
        velocities[:, 2] = 6.28/3 # Velocity RR
        velocities[:, 3] = 6.28/3 # Velocity FL
        velocities[:, 4] = 6.28/3 # Velocity CL
        velocities[:, 5] = 6.28/3 # Velocity RL

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
            r.append(R.from_euler('xyz', [(random.random() * 2 * math.pi), 0, 3.14], degrees=False).as_quat())

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
        
        # cart_pos = self.obs_buf[:, 0]
        # cart_vel = self.obs_buf[:, 1]
        # pole_angle = self.obs_buf[:, 2]
        # pole_vel = self.obs_buf[:, 3]

        # reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        # self.rew_buf[:] = reward
        pass

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
