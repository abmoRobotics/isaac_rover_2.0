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

import time
import numpy as np
import torch
import utils.terrain_utils.terrain_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omniisaacgymenvs.robots.articulations.rover import Rover
from omniisaacgymenvs.robots.articulations.views.rover_view import RoverView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
from omniisaacgymenvs.tasks.utils.debug_utils import draw_depth
#from omniisaacgymenvs.tasks.utils.rover_terrain import *
from omniisaacgymenvs.tasks.utils.rover_utils import *
from omniisaacgymenvs.utils.kinematics import Ackermann
from omniisaacgymenvs.utils.terrain_utils.terrain_generation import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from scipy.spatial.transform import Rotation as R
from omni.isaac.debug_draw import _debug_draw
from omniisaacgymenvs.tasks.utils.tensor_quat_to_euler import tensor_quat_to_eul
from omniisaacgymenvs.tasks.utils.camera import Camera

class Memory():
    def __init__(self,num_envs, num_states, horizon, device) -> None:
        print(num_envs)
        print(num_states)
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
        self._device = 'cuda:0'
        self.shift = torch.tensor([0 , 0, 0.0],device=self._device)
        self.Camera= Camera(self._device,self.shift)
        self.num_exteroceptive = self.Camera.get_num_exteroceptive()

        # Define action space and observation space
        self._num_proprioceptive = 4
        self._num_observations = self._num_proprioceptive + self.num_exteroceptive
        self._num_actions = 2


        #self._device = 'cpu'
        self._sim_config = sim_config

        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._rover_positions = torch.tensor([25.0, 25.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self.max_episode_length = 500
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]

        self._ball_position = torch.tensor([0, 0, 1.0])
        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 0
        

        # Setup state trackers
        self.linear_velocity = Memory(num_envs = self._num_envs, num_states = 1, horizon=3,device=self._device)
        self.angular_velocity = Memory(num_envs = self._num_envs, num_states = 1, horizon=3,device=self._device)

        # Load reward weights
        print(self._task_cfg["rewards"])
        self.rew_scales = self._task_cfg["rewards"]
        # self._rover_position = torch.tensor([0, 0, 2])
        
        #self.stone_info = utils.terrain_utils.terrain_utils.read_stone_info("/home/decamargo/Desktop/stone_info.npy")
        # self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # self.marker_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))
                # Previous actions and torques
        self.actions_nn = torch.zeros((self.num_envs, self._num_actions, 3), device=self._device)
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

        #self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        #vertices = self.terrain.vertices
        #triangles = self.terrain.triangles
        position = self.shift
        add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)  
        #self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        #print(self.height_samples.shape)
        
    # Sets up the scene for the rover to navigate in
    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_rover()    # Gets the rover asset/articulations
        self.get_target()   # Gets the target sphere (only used for visualizing the target)
        self.get_terrain()
        super().set_up_scene(scene)
        self._rover = RoverView(prim_paths_expr="/World/envs/.*/Rover", name="rover_view")  # Creates an objects for the rover
        positions = self._rover.get_world_poses()[0]
        heightmap = torch.load("tasks/utils/terrain/heightmap_tensor.pt")
        position = rover_spawn_height(heightmap,positions[:,0:2],0.025,1,self.shift[0:2])
        position = position.unsqueeze(1)
        zero_position = torch.zeros((self._num_envs,2),device=self._device)

        self.position_z_offset = torch.cat((zero_position,position.to('cuda:0')),1)

        self.initial_pos = self._rover.get_world_poses()[0]
        #print(self.initial_pos)
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False)   # Creates an object for the sphere
        scene.add(self._balls)  # Adds the sphere to the scene
        scene.add(self._rover)  # Adds the rover to the scene
        #self._rover.initialize()
        # draw = _debug_draw.acquire_debug_draw_interface()
        # points = [[0.0,0.0,0.0]]
        # colors = [[0.0,1.0,0.0,0.5]]
        # draw.draw_points(points, colors, [22])
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
        #self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

    def get_observations(self) -> dict:
        # Get root state of rover
        self.rover_positions = self._rover.get_world_poses()[0]
        self.rover_rotation = tensor_quat_to_eul(self._rover.get_world_poses()[1])

        # Calculate
        direction_vector = torch.zeros([self.num_envs, 2], device=self._device)
        target_vector = self.target_positions[..., 0:2] - self.rover_positions[..., 0:2]
        self.heading_diff = torch.atan2(target_vector[:,0] * direction_vector[:,1] - target_vector[:,1]*direction_vector[:,0],target_vector[:,0]*direction_vector[:,0]+target_vector[:,1]*direction_vector[:,1])


        #root_positions
        heightmap, output_pt, sources = self.Camera.get_depths(self.rover_positions,self.rover_rotation)
        

        # This function is used for calculating the observations/input to the rover.
        self.obs_buf[:, 0] = torch.linalg.norm(target_vector,dim=1) / 4
        self.obs_buf[:, 1] = (self.heading_diff) / math.pi
        self.obs_buf[:, 2] = self.linear_velocity.get_state(timestep=0)
        self.obs_buf[:, 3] = self.angular_velocity.get_state(timestep=0)

        
        self.obs_buf[:, self._num_proprioceptive:self._num_observations ] = heightmap
        

        observations = {
            self._rover.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:

        #print(self._rover.get_local_poses()[0])
        #print(self._rover.get_world_poses()[0])
        #print(self.terrain_origins)
        # Get the transformation data on rovers
        #try:

        
      
        #time.sleep(2)

        # Get the environemnts ids of the rovers to reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # Reset rovers 
            self.reset_idx(reset_env_ids)
            # Reset goal targets

            self.set_targets(reset_env_ids)

        
        # Get action from model    
        _actions = actions.to(self._device)

        # Track states
        self.linear_velocity.input_state(_actions[:,0]) # Track linear velocity
        self.angular_velocity.input_state(_actions[:,1]) # Track angular velocity

        # Code for running ExoMy in Ackermann mode
        _actions[:,0] = _actions[:,0] * 1.17 # max speed
        _actions[:,1] = _actions[:,1] * (1.17/0.58) # max speed / distance to wheel furthest away in meters
        #TODO remove
        _actions = _actions
        
        self.actions_nn = torch.cat((torch.reshape(_actions,(self.num_envs, self._num_actions, 1)), self.actions_nn), 2)[:,:,0:3]
        self.actions_nn = self.actions_nn
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
        #self.base_pos[env_ids] = self.base_init_state[0:3]
        #self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
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

        reset_orientation = torch.tensor(r,device=self._device) # Convert quartenion to tensor

        dof_pos = torch.zeros((num_resets, self._rover._num_pos_dof), device=self._device)
        dof_vel = torch.zeros((num_resets, self._rover._num_vel_dof), device=self._device)

        
      
        # # apply resets    
        # indices = env_ids.to(dtype=torch.int32)
        # self._rovers.set_joint_positions(dof_pos, indices=indices)
        # self._rovers.set_joint_velocities(dof_vel, indices=indices)
        self.base_pos[env_ids] = self.initial_pos[env_ids]
        self.base_pos[env_ids, 2] += self.position_z_offset[env_ids,2]

        # Indicies of rovers to reset
        indices = env_ids.to(dtype=torch.int32) 
        # Set the position/orientation of the rover after reset
#        self._rover.set_world_poses(self.base_pos[env_ids].clone(), reset_orientation, indices)

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

        #
        heading_diff = self.heading_diff

        # Distance to target
        target_dist = torch.sqrt(torch.square(self.target_positions[..., 0:2] - self.rover_positions[..., 0:2]).sum(-1))

        target_vector = self.target_positions[..., 0:2] - self.rover_positions[..., 0:2]
        direction_vector = torch.zeros([ones.shape[0], 2], device='cuda:0')
        direction_vector[:,0] = torch.cos(self.rover_rotation[..., 2] - (math.pi/2)) # x value
        direction_vector[:,1] = torch.sin(self.rover_rotation[..., 2] - (math.pi/2)) # y value

        # Heading constraint - Avoid driving backwards
        lin_vel = self.linear_velocity.get_state(0)   # Get latest lin_vel
        heading_contraint_penalty = torch.where(lin_vel < 0, -max_reward, zero_reward) * self.rew_scales["heading_contraint_reward"]


        # penalty for driving away from the robot
        goal_angle_penalty = torch.where(torch.abs(heading_diff) > 2, -torch.abs(heading_diff*0.3*self.rew_scales['goal_angle_reward']), zero_reward.float())

        # Motion constraint - No oscillatins
        penalty1 = torch.where((torch.abs(lin_vel - lin_vel_prev) > 0.05), torch.square(torch.abs(lin_vel - lin_vel_prev)),zero_reward.float())
        penalty2 = torch.where((torch.abs(ang_vel- ang_vel_prev) > 0.05), torch.square(torch.abs(ang_vel- ang_vel_prev)),zero_reward.float())
        motion_contraint_penalty =  torch.pow(penalty1,2) * self.rew_scales["motion_contraint_reward"]
        motion_contraint_penalty = motion_contraint_penalty+(torch.pow(penalty2,2)) * self.rew_scales["motion_contraint_reward"]

        # distance to target
        pos_reward = (1.0 / (1.0 + target_dist * target_dist)) * self.rew_scales['pos_reward']
        pos_reward = torch.where(target_dist <= 0.03, 1.03*(self.max_episode_length-self.progress_buf), pos_reward.float())  # reward for getting close to target

        #TODO add collision penalty
        reward = pos_reward + heading_contraint_penalty + motion_contraint_penalty + goal_angle_penalty
        #reward = pos_reward + collision_penalty + heading_contraint_penalty + motion_contraint_penalty  + heading_diff_reward + goal_angle_penalty
        #reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        #reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = reward


        # self.rew_buf[:] = reward_total

    def generate_goals(self,env_ids,radius=3):
        self.random_goals(env_ids, radius=radius) # Generate random goals
        valid_goals = self.avoid_pos_rock_collision(self.target_positions)
        self.target_positions = valid_goals


    def random_goals(self, env_ids, radius):
        num_sets = len(env_ids)
        alpha = 2 * math.pi * torch.rand(num_sets, device=self._device)
        TargetRadius = radius
        TargetCordx = 0
        TargetCordy = 0
        x = TargetRadius * torch.cos(alpha) + TargetCordx
        y = TargetRadius * torch.sin(alpha) + TargetCordy
        self.target_positions[env_ids, 0] = x #+ self.spawn_offset[env_ids, 0]
        self.target_positions[env_ids, 1] = y #+ self.spawn_offset[env_ids, 1]


    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        self.generate_goals(env_ids, radius=3) # Generate goals
        envs_long = env_ids.long()
        print("Terrain: " + str(self.terrain.heightsamples.shape))
        print(self.target_positions)
        global_pos = self.target_positions[env_ids, 0:2]#.add(self.env_origins_tensor[env_ids, 0:2])
        height= self.get_spawn_height(self.tensor_map, global_pos, self.horizontal_scale, self.vertical_scale, self.shift)
        self.target_positions[env_ids, 2] = height
        marker_pos= self.target_positions
        self._balls.set_world_poses(marker_pos[:,0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

        #actor_indices = self.all_actor_indices[env_ids, 1].flatten()
        #self.gym.set_actor_root_state_tensor_indexed(self.sim,self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_sets)

        #return actor_indices

    
    def get_spawn_height(heightmap: torch.Tensor, depth_points: torch.Tensor, horizontal_scale, vertical_scale, shift):
        # Scale locations to fit heightmap
        print(depth_points)
        print(shift)
        print(horizontal_scale)
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
        # print(x)
        # print("y")
        # print(y)
        
        # Scale to fit actual height, dependent on resolution
        heights = heights * vertical_scale

        return heights


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
        resets = torch.where(self.progress_buf >= self.max_episode_length, 1, 0)
        self.reset_buf[:] = resets


    def avoid_pos_rock_collision(self, curr_pos):
        #curr_pos = self._rover_positions
        old_pos = None
        while not curr_pos == old_pos:
            # what is the purpose of env_origins here? -> can it get removed?
            shifted_pos = curr_pos[:,0:2] - self.shift #.add(self.env_origins_tensor[:,0:2]) - self.shift 
            old_pos = curr_pos
            dist_rocks = torch.cdist(shifted_pos[:,0:2], self.stone_info[:,0:2], p=2.0)  # Calculate distance to center of all rocks
            dist_rocks[:] = dist_rocks[:] - self.stone_info[:,6]                               # Calculate distance to nearest point of all rocks
            nearest_rock = torch.min(dist_rocks,dim=1)[0]                                   # Find the closest rock to each robot
            curr_pos[:,0] = torch.where(nearest_rock[:] <= 0.2,curr_pos[:,0]+0.10,curr_pos[:,0])
        return curr_pos
