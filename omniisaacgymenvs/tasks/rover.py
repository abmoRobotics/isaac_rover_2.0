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
from omniisaacgymenvs.tasks.utils.rover_depth import draw_depth
from omniisaacgymenvs.tasks.utils.rover_terrain import *
from omniisaacgymenvs.tasks.utils.rover_utils import *
from omniisaacgymenvs.utils.kinematics import Ackermann
from omniisaacgymenvs.utils.terrain_utils.terrain_generation import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from scipy.spatial.transform import Rotation as R
from omni.isaac.debug_draw import _debug_draw
from omniisaacgymenvs.tasks.utils.tensor_quat_to_euler import tensor_quat_to_eul
from omniisaacgymenvs.tasks.utils.camera import Camera
from pxr import UsdPhysics, Sdf, Gf, PhysxSchema, UsdGeom, Vt, PhysicsSchemaTools
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts.physicsUtils import *
class RoverTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._device = 'cuda:0'
        #self._device = 'cpu'
        self._sim_config = sim_config

        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._rover_positions = torch.tensor([25.0, 25.0, 2.0])
        self.heightmap = None
        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self._num_observations = 4
        self._num_actions = 2
        self._ball_position = torch.tensor([0, 0, 1.0])
        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.shift = torch.tensor([0 , 0, 0.0],device=self._device)
        self.stone_info = utils.terrain_utils.terrain_utils.read_stone_info("./stone_info.npy")
        self.shift2 = 5
        self.position_z_offset = None
        # self._rover_position = torch.tensor([0, 0, 2])
        self.Camera= Camera(self._device,self.shift)
        self.stone_prim_path = "/World/stones"
        
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
        #vertices, triangles = load_terrain('terrainTest.ply')
        vertices, triangles = load_terrain('terrainTest.ply')
        v2, t2 = load_terrain('stones_only.fbx')

        #self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        #vertices = self.terrain.vertices
        #triangles = self.terrain.triangles
        position = self.shift
        add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)  
        add_stones_to_stage(stage=self._stage, vertices=v2, triangles=t2, position=position)  
        #self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        #print(self.height_samples.shape)
        
    # Sets up the scene for the rover to navigate in
    def set_up_scene(self, scene) -> None:
        self.position_z_offset
        self._stage = get_current_stage()
        self.get_rover()    # Gets the rover asset/articulations
        self.get_target()   # Gets the target sphere (only used for visualizing the target)
        self.get_terrain()
        super().set_up_scene(scene)
        self._rover = RoverView(prim_paths_expr="/World/envs/.*/Rover", name="rover_view")  # Creates an objects for the rover
        positions = self._rover.get_world_poses()[0]
        #positions = positions.cpu()
        self.heightmap = torch.load("tasks/utils/heightmap_tensor.pt")
        #heightmap = heightmap.to('cpu')
        self.horizontal_scale = 0.025
        self.vertical_scale = 1
        pre_col = positions.clone()
        positions = self.avoid_pos_rock_collision(positions)
        position = self.get_pos_height(self.heightmap, positions[:,0:2],self.horizontal_scale,self.vertical_scale,self.shift[0:2])
        self.position_z_offset = torch.ones(position.shape, device=self._device)
        positions[:,2] = torch.add(position, self.position_z_offset)
        #zero_position = torch.zeros((self._num_envs,2),device=self._device)
        self.initial_pos = positions#self._rover.get_world_poses()[0]
        self._rover.set_world_poses(self.initial_pos, self._rover.get_world_poses()[1])
        #print(self.initial_pos)
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False)   # Creates an object for the sphere
        scene.add(self._balls)  # Adds the sphere to the scene
        scene.add(self._rover)  # Adds the rover to the scene
        #self._rover.initialize()
        draw = _debug_draw.acquire_debug_draw_interface()
        points = [[0.0,0.0,0.0]]
        colors = [[0.0,1.0,0.0,0.5]]
        draw.draw_points(points, colors, [22])
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
        # This function is used for calculating the observations/input to the rover.
        pass


    def report_hit(self, hit):
        print("HIT: " + str(hit))


    def pre_physics_step(self, actions) -> None:
        self.check_collision(self._rover.get_world_poses()[0])
        #print(self._rover.get_local_poses()[0])
        #print(self._rover.get_world_poses()[0])
        #print(self.terrain_origins)
        # Get the transformation data on rovers
        #try:
        #path_tuple = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(self.stone_prim_path))         
        #numHits = get_physx_scene_query_interface().overlap_mesh(path_tuple[0], path_tuple[1], self.report_hit, False)  

        self.rover_loc = self._rover.get_world_poses()[0]
        self.rover_rot = tensor_quat_to_eul(self._rover.get_world_poses()[1])

        a = torch.zeros((8,3),device=self._device)
        b = torch.ones((8,3),device=self._device)
        #c_tester = self.Camera.get_depths(self.rover_loc,self.rover_rot)
        
      
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

        # Code for running ExoMy in Ackermann mode
        _actions[:,0] = _actions[:,0] * 3
        _actions[:,1] = _actions[:,1] * 3
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
        #self.base_pos[env_ids, :] += self.position_z_offset[env_ids, :]

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
        # penalty1 = torch.where((torch.abs(self.actions_nn[:,0,0] - self.actions_nn[:,0,1]) > 0.05), torch.square(torch.abs(self.actions_nn[:,0,0] - self.actions_nn[:,0,1])),zero_reward)
        # penalty2 = torch.where((torch.abs(self.actions_nn[:,1,0] - self.actions_nn[:,1,1]) > 0.05), torch.square(torch.abs(self.actions_nn[:,1,0] - self.actions_nn[:,1,1])),zero_reward)
        # motion_contraint_penalty =  torch.pow(penalty1,2) * (-0.01)#rew_scales["motion_contraint"]
        # motion_contraint_penalty = motion_contraint_penalty+(torch.pow(penalty2,2)) * (-0.01)#rew_scales["motion_contraint"]


        # reward_total = motion_contraint_penalty

        # self.rew_buf[:] = reward_total


    def generate_goals(self,env_ids,radius=5):
        self.random_goals(env_ids, radius=radius) # Generate random goals
        #valid_goals = self.avoid_pos_rock_collision(self.target_positions)
        #self.target_positions = valid_goals


    def random_goals(self, env_ids, radius):
        num_sets = len(env_ids)
        alpha = 2 * math.pi * torch.rand(num_sets, device=self._device)
        TargetRadius = radius * math.sqrt(random.random())
        TargetCordx = 0
        TargetCordy = 0
        x = TargetRadius * torch.cos(alpha) + TargetCordx
        y = TargetRadius * torch.sin(alpha) + TargetCordy
        # TODO: Add offset of each environement 
        self.target_positions[env_ids, 0] = x + self.initial_root_pos[env_ids, 0]#self.position_z_offset[env_ids, 0]
        self.target_positions[env_ids, 1] = y + self.initial_root_pos[env_ids, 1]#self.position_z_offset[env_ids, 1]


    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        self.generate_goals(env_ids, radius=3) # Generate goals
        envs_long = env_ids.long()
        global_pos = self.target_positions[env_ids, 0:2]#.add(self.env_origins_tensor[env_ids, 0:2])
        height= self.get_pos_height(self.heightmap, global_pos[:,0:2], self.horizontal_scale, self.vertical_scale, self.shift[0:2])
        self.target_positions[env_ids, 2] = height
        self._balls.set_world_poses(self.target_positions[envs_long], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

        #actor_indices = self.all_actor_indices[env_ids, 1].flatten()
        #self.gym.set_actor_root_state_tensor_indexed(self.sim,self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_sets)

        #return actor_indices

    
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
        # print(x)
        # print("y")
        # print(y)
        
        # Scale to fit actual height, dependent on resolution
        heights = heights * vertical_scale

        return heights


    def is_done(self) -> None:
        # Function that checks whether or not the rover should reset

        # cart_pos = self.obs_buf[:, 0]
        # pole_pos = self.obs_buf[:, 2]

        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        #resets = torch.zeros((self._num_envs, 1), device=self._device)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, 0)
        self.reset_buf[:] = resets


    def avoid_pos_rock_collision(self, curr_pos):
        #curr_pos = self._rover_positions
        old_pos = torch.zeros(curr_pos.shape).cuda()
        while not torch.equal(curr_pos, old_pos):
            # what is the purpose of env_origins here? -> can it get removed?
            shifted_pos = curr_pos[:,0:2] #- self.shift2 #.add(self.env_origins_tensor[:,0:2]) - self.shift 
            old_pos = curr_pos.clone()
            dist_rocks = torch.cdist(shifted_pos[:,0:2], self.stone_info[:,0:2], p=2.0)  # Calculate distance to center of all rocks
            dist_rocks[:] = dist_rocks[:] - self.stone_info[:,6]                               # Calculate distance to nearest point of all rocks
            nearest_rock = torch.min(dist_rocks,dim=1)[0]                      # Find the closest rock to each robot
            curr_pos[:,0] = torch.where(nearest_rock[:] <= 0.5,torch.add(curr_pos[:,0], 0.05),curr_pos[:,0])
        print("Pos after: " + str(curr_pos))
        return curr_pos

    
    def check_collision(self, curr_pos):
            dist_rocks = torch.cdist(curr_pos[:,0:2], self.stone_info[:,0:2], p=2.0)  # Calculate distance to center of all rocks
            dist_rocks[:] = dist_rocks[:] - self.stone_info[:,6]                          # Calculate distance to nearest point of all rocks
            nearest_rock = torch.min(dist_rocks,dim=1)[0]             # Find the closest rock to each robot
            for n in range(len(nearest_rock)):
                if nearest_rock[n] <= 0.5:
                    self.reset_buf[n] = 1
