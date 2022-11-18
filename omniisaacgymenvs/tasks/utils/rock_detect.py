import numpy as np
import matplotlib.pyplot as plt
import torch
import open3d as o3d
from omniisaacgymenvs.tasks.utils.debug_utils import draw_depth
from omniisaacgymenvs.tasks.utils.ray_casting import ray_distance
import time

class Rock_Detection():
    """Ray–Triangle intersection based on Möller–Trumbore intersection algorithm
    

    Attributes
    ----------
    device : str
        Device for execution
    shift : torch.tensor
        translation of the map, relative to the global reference frame of the simulation
    debug : bool, optional
        Enables debugging functionalities (default is False)

    Methods
    -------
    get_depths(positions, rotations)
        Returns a depthmap for a robot based on position and rotation
    """

    
    def __init__(self, device, shift,debug=False):
        """
        Parameters
        ----------
        device : str
            Device for execution
        shift : torch.tensor
            translation of the map, relative to the global reference frame of the simulation
        debug : bool, optional
            Enables debugging functionalities (default is False)
        """
        self.debug = debug
        self.device = device    
        self.partition = True    
        self.num_partitions = 1 # Slices the input data to reduce VRAM usage. 
        self.horizontal = 0.05  # Resolution of the triangle map
        self.rock_indices, self.triangles, self.vertices = self._load_triangles_with_indices()   # Load into [1200,1200, 100, 3] array
        self.shift = shift          # Shift of the map 
        self.dtype = torch.float16  # data type for performing the calculations

    def get_num_exteroceptive(self):
        return self.num_exteroceptive

    def get_depths(self, positions, rotations, joint_states):
        """
        Returns a depthmap for a robot based on position and rotation

        Parameters
        ----------
        positions : torch.tensor
            Device for execution
        rotations : torch.tensor
            translation of the map, relative to the global reference frame of the simulation
        """
        sources, directions = self._get_wheel_rays(positions, rotations, joint_states)
        depth_points = sources[:,:,0:2]
        triangles,triangles_with_indices = self._height_lookup(self.rock_indices,depth_points,self.horizontal,self.shift)
        self.initialmemory = torch.cuda.memory_reserved(0)
        triangles = self.vertices[self.triangles[triangles_with_indices.long()].long()]
        # We partition the data to reduce VRAM usage
        if self.partition:
            partitions = self.num_partitions
            output_distances = None
            for i in range(partitions):
                step_size = int(sources.shape[1]/partitions)
         
          
                s = sources[:,i*step_size:i*step_size+step_size]
     
                d = directions[:,i*step_size:i*step_size+step_size]
                t = triangles[:,i*step_size:i*step_size+step_size]
       
                dim0, dim1, dim2 = s.shape[0], s.shape[1], s.shape[2]
                t_dim0, t_dim1, t_dim2, t_dim3, t_dim4 = t.shape[0], t.shape[1], t.shape[2], t.shape[3], t.shape[4]
      
                s = s.repeat(1,1,t_dim2)

                s = s.reshape(dim0*dim1*t_dim2,dim2)
     
                #s = s.repeat(t_dim2,1)
              
                d = d.repeat(1,1,t_dim2)
                d = d.reshape(dim0*dim1*t_dim2,dim2)
                #d = d.repeat(t_dim2,1)
    
                t2 = t.reshape(t_dim0*t_dim1,t_dim2,t_dim3,t_dim4)
     
                t = t.reshape(t_dim0*t_dim1*t_dim2,t_dim3,t_dim4)
                self.t = t
         
                #torch.save(t,'tritest.pt')
                #torch.save(s, 'sourcetest.pt')
                self.initialmemory2 = torch.cuda.memory_reserved(0)
                
                distances,pt = ray_distance(s,d,t)
                self.initialmemory3 = torch.cuda.memory_reserved(0)
                distances = distances.reshape(dim0,dim1,t_dim2)
     
                pt = pt.reshape(dim0,dim1,t_dim2,3)
                # Find the maximum value of the 100 triangles to extract the "heightmap" value.
                
                indices =torch.max(distances,2).indices
                distances = torch.max(distances,2).values
                indices = indices.unsqueeze(dim=2).unsqueeze(dim=2).repeat(1,1,1,3)
                pt = pt.gather(dim=2,index=indices)
                pt = pt.squeeze(dim=2)
               # pt = pt[:,:,indices]
                #distances = distances.reshape(dim0,dim1)

                if output_distances is None:
                    output_distances = distances.clone().detach()#torch.tensor(distances,device=self.device,)
                    output_pt = pt.clone().detach()#torch.tensor(pt,device=self.device,)
                else: 
                    output_distances = torch.cat((output_distances,distances),1)
                    output_pt = torch.cat((output_pt,pt),1)

        else:
            dim0, dim1, dim2 = sources.shape[0], sources.shape[1], sources.shape[2]
            s = sources.reshape(dim0*dim1,dim2)
            d = directions.reshape(dim0*dim1,dim2)
            t = triangles.reshape(t_dim0*t_dim1*t_dim2,t_dim3,t_dim4)
            output_distances = ray_distance(s,d,t)

        
        d1,d2,d3 = sources.shape[0],sources.shape[1] ,sources.shape[2] 
        #print((self.initialmemory3 - torch.cuda.memory_reserved(0))/1_000_000_000)
        if self.debug:
            try:
                draw_depth(sources.reshape(d1*d2,d3),output_pt.reshape(d1*d2,d3))

            except:
                print("Isaac Sim not running")
        return output_distances, output_pt, sources

    def _load_triangles_with_indices(self):
        """Loads triangles with indicies to vertices, each triangle is a 3D vector"""
        rock_indices = torch.load("tasks/utils/terrain/knn_terrain/map_indices.pt")
        rock_indices  = rock_indices.swapaxes(0,1)
        rock_indices  = rock_indices.swapaxes(1,2)
        triangles = torch.load("tasks/utils/terrain/knn_terrain/triangles.pt")
        vertices = torch.load("tasks/utils/terrain/knn_terrain/vertices.pt")
        return rock_indices, triangles, vertices



    def _get_wheel_rays(self, positions, rotations, joint_states):
        """
        Returns sources and directions for rays surrounding each wheel of the rovers based on position, rotation and joint states.

        Parameters
        ----------
        positions : torch.tensor [num_robot, 3]
            Rover positions in global frame
        rotations : torch.tensor [num_robots, 3]
            Rover rotations in euler angles.
        joint_states : torch.tensor [num_robots, 13]
            Joint positions of the Rover articulations
        """

        """
        FL_Boogie_Revolute  0
        FR_Boogie_Revolute  1
        R_Boogie_Revolute   2
        CL_Drive_Continous  3
        FL_Steer_Revolute   4
        CR_Drive_Continous  5
        FR_Steer_Revolute   6
        RL_Steer_Revolute   7
        RR_Steer_Revolute   8
        FL_Drive_Continous  9
        FR_Drive_Continous  10
        RL_Drive_Continous  11
        RR_Drive_Continous  12
        """

        # X:0, Y:1, Z:2
    # Setup
        # Location of the rays around the wheel. The last vector descibes the direction of projection
        wheel_rays = wheel_FL = torch.tensor(  [[[0.215/2],[0.130/2],[0.1]],      #Ray 1            
                                                [[0.215/2],[-0.130/2],[0.1]],     #Ray 2
                                                [[-0.215/2],[0.130/2],[0.1]],     #Ray 3
                                                [[-0.215/2],[-0.130/2],[0.1]],    #Ray 4
                                                [[0], [0], [-1]]],              #Direction
                                                device=self.device)

        # The first transformation, from wheel to boogie joint
        wheel_positions0 = torch.tensor([[[0.286],[0.385],[-0.197]],   #Fl
                                        [[0.286],[-0.385],[-0.197]],   #FR
                                        [[-0.146],[0.447],[-0.197]],   #CL
                                        [[-0.146],[-0.447],[-0.197]],  #CR
                                        [[-0.440],[0.385],[-0.197]],   #RL
                                        [[-0.440],[-0.385],[-0.197]]], #RR
                                        device=self.device)
        
        # The last rotation, from boogie joint to base-frame of rover
        wheel_positions1 = torch.tensor([[[0.153],[0],[0.03]],      #Fl
                                        [[0.153],[0],[0.03]],       #FR
                                        [[0.153],[0],[0.03]],      #CL
                                        [[0.153],[-0],[0.03]],     #CR
                                        [[0],[0],[0.03]],     #RL
                                        [[0],[0],[0.03]]],   #RR
                                        device=self.device)

        # Get number of points, robots, wheels and rays from input
        num_robots = positions.size()[0]
        rays_per_wheel = wheel_rays.shape[0]-1 # -1 because direction is not included.
        wheels = wheel_positions0.shape[0]
        num_points = wheels * (wheel_rays.shape[0]) 

    # Formatting
        # Expand wheel positions, and set translation for direction vector equal to zero.
        wheel_positions0 = wheel_positions0.repeat_interleave(rays_per_wheel+1,dim=0)
        wheel_positions0[4::5] = torch.zeros(wheel_positions0[4::5].shape, device=self.device)
        wheel_positions0 = wheel_positions0.expand(num_robots,-1,-1,-1)

        # Expand wheel positions, and set translation for direction vector equal to zero.
        wheel_positions1 = wheel_positions1.repeat_interleave(rays_per_wheel+1,dim=0)
        wheel_positions1[4::5] = torch.zeros(wheel_positions1[4::5].shape, device=self.device)
        wheel_positions1 = wheel_positions1.expand(num_robots,-1,-1,-1)

        # Each robot has some amount of wheels, resulting in X rays per robot
        rays = wheel_rays.repeat(wheels, 1, 1)
        
        # Expand rays to one row for each robot, for X, Y and Z.
        x = torch.transpose(rays[:,0,:], 0, 1).repeat(num_robots, 1)
        y = torch.transpose(rays[:,1,:], 0, 1).repeat(num_robots, 1)
        z = torch.transpose(rays[:,2,:], 0, 1).repeat(num_robots, 1)

        # Add a dimenion to joint_states - Makes concatenation easier
        joint_states = joint_states.expand(1, -1, -1)
        
    # Steering transform
        # Get steering angles for wheels.                          FL                   FR         CL                                     CR                                                RL                   RR
        steer_angles = torch.transpose(torch.cat((joint_states[:,:,4], joint_states[:,:,6], torch.zeros_like(joint_states[:,:,6]), torch.zeros_like(joint_states[:,:,6]), -joint_states[:,:,7], joint_states[:,:,8]), 0), 0, 1)
        steer_angles = steer_angles.repeat_interleave((rays_per_wheel+1),1)

        # Compute sin and cos to rover steering angles
        sinst = torch.sin(-steer_angles)
        cosst = torch.cos(-steer_angles)

        # Transform points
        x1 = wheel_positions0[:,:,0,0] + x*cosst + y*sinst
        y1 = wheel_positions0[:,:,1,0] + y*cosst - x*sinst
        z1 = wheel_positions0[:,:,2,0] + z

    # Suspension transform
        # Get joint angles for the respective rotation axises
        #                       FL                                                FR                                     CL                                     CR                                     RL                                     RR
        susY = torch.transpose( torch.cat((-joint_states[:,:,0],                  joint_states[:,:,1],                   -joint_states[:,:,0],                  joint_states[:,:,1],                   torch.zeros_like(joint_states[:,:,0]), torch.zeros_like(joint_states[:,:,0]) ), 0), 0, 1)
        susX = torch.transpose( torch.cat((torch.zeros_like(joint_states[:,:,0]), torch.zeros_like(joint_states[:,:,0]), torch.zeros_like(joint_states[:,:,0]), torch.zeros_like(joint_states[:,:,0]), -joint_states[:,:,2],                  -joint_states[:,:,2]                  ), 0), 0, 1)
        susX = susX.repeat_interleave((rays_per_wheel+1),1)
        susY = susY.repeat_interleave((rays_per_wheel+1),1)

        # Compute sin and cos to rover suspension angles
        sinsux = torch.sin(susX)
        cossux = torch.cos(susX) 
        sinsuy = torch.sin(susY) 
        cossuy = torch.cos(susY) 

        # Transform points
        x2 = wheel_positions1[:,:,0,0] + x1 * cossuy - sinsuy * (z1*cossux - y1*sinsux)
        y2 = wheel_positions1[:,:,1,0] + y1 * cossux + z1 * sinsux
        z2 = wheel_positions1[:,:,2,0] + x1 * sinsuy + cossuy * (z1*cossux - y1*sinsux)

    # Rover transform
        # Compute sin and cos to rover rotation angles - Sizes: [n, 1]
        sinxr = torch.transpose(torch.sin(-rotations[:,0].expand(1, num_robots)), 0, 1)
        cosxr = torch.transpose(torch.cos(-rotations[:,0].expand(1, num_robots)), 0, 1)
        sinyr = torch.transpose(torch.sin(-rotations[:,1].expand(1, num_robots)), 0, 1)
        cosyr = torch.transpose(torch.cos(-rotations[:,1].expand(1, num_robots)), 0, 1)
        sinzr = torch.transpose(torch.sin(-rotations[:,2].expand(1, num_robots)), 0, 1)
        coszr = torch.transpose(torch.cos(-rotations[:,2].expand(1, num_robots)), 0, 1)

        # Zeros, for not transforming direction vector
        zeros = torch.zeros(num_robots,1, device = self.device)
        
        # Get transforms, and set to zero for direction vector
        rover_xl = torch.transpose(positions[:, 0].expand(rays_per_wheel,num_robots), 0, 1)
        rover_xl = torch.cat((rover_xl,zeros),1)
        rover_xl = rover_xl.repeat(1, wheels)

        rover_yl = torch.transpose(positions[:, 1].expand(rays_per_wheel,num_robots), 0, 1)
        rover_yl = torch.cat((rover_yl,zeros),1)
        rover_yl = rover_yl.repeat(1, wheels)

        rover_zl = torch.transpose(positions[:, 2].expand(rays_per_wheel,num_robots), 0, 1)
        rover_zl = torch.cat((rover_zl,zeros),1)
        rover_zl = rover_zl.repeat(1, wheels)
            
        # Transform points in the pointcloud
        x_p = rover_xl + sinzr * (y2*cosxr + z2*sinxr) + coszr * (x2*cosyr - sinyr*(z2*cosxr - y2*sinxr))
        y_p = rover_yl + coszr * (y2*cosxr + z2*sinxr) - sinzr * (x2*cosyr - sinyr*(z2*cosxr - y2*sinxr))
        z_p = rover_zl + x2*sinyr + cosyr*(z2*cosxr - y2*sinxr)

    # Formatting
        #Stack points in a [x, y, 3] matrix, and return
        sources = torch.stack((x_p[:,:], y_p[:,:], z_p[:,:]), 2)

        # Get ray direction vectors, and repeat for each wheel
        ray_dir = sources[:,rays_per_wheel::rays_per_wheel+1,:].repeat_interleave(rays_per_wheel,1)
    
        # Delete direction vector from sources
        sources = sources.reshape(-1,5,3)[:,:4].reshape(num_robots, wheels * (rays_per_wheel), 3)

        return sources.type(torch.float16), ray_dir.type(torch.float16)

    def _height_lookup(self, triangle_matrix: torch.Tensor, depth_points: torch.Tensor, horizontal_scale, shift):
        """Look up the nearest triangles relative to an x, y coordinate"""
        # Heightmap 1200x1200x100
        # depth_points: Points in 3D [n_envs,n_points, 3]
        # horizontal_scale = 0.05
        # shift: [n, 2]
        shift = shift[0:2]
        # Scale locations to fit heightmap
        scaledmap = (depth_points-shift)/horizontal_scale
        # Bound values inside the map
        scaledmap = torch.clamp(scaledmap, min = 0, max = triangle_matrix.size()[0]-1)
        # Round to nearest integer
        scaledmap = torch.round(scaledmap)

        # Convert x,y coordinates to two vectors.
        x = scaledmap[:,:,0]
        y = scaledmap[:,:,1]
        x = x.reshape([(depth_points.size()[0]* depth_points.size()[1]), 1])
        y = y.reshape([(depth_points.size()[0]* depth_points.size()[1]), 1])
        x = x.long()
        y = y.long()
        
        # Get nearets array of triangles for searching
        triangles = 0
        triangles_with_indices = triangle_matrix[x, y]
        triangles_with_indices = triangles_with_indices.reshape([depth_points.shape[0],depth_points.shape[1],triangle_matrix.shape[2]])

        # Return the found heights
        return triangles, triangles_with_indices

if __name__ == "__main__":
    a = torch.tensor([0.0,0.0],device='cuda:0')
    cam = Camera('cuda:0', a)
    pos2 = torch.tensor([
            [6.5921, 5.5467, 0.6557],
            [5.4156, 4.4939, 0.7105],
            [5.4735, 5.5226, 0.6976],
            [4.5162, 4.5003, 0.6982],
            [4.5202, 5.5012, 0.7037],
            [3.5420, 4.5217, 0.7294]], device='cuda:0',dtype=torch.float16)
    rotations2 = torch.tensor([
            [-0.0949,  0.1376, -0.0917],
            [ 0.1036, -0.0723, -0.0339],
            [-0.1516, -0.0358, -0.0008],
            [ 0.0136,  0.0715, -0.0126],
            [-0.0114,  0.0963, -0.0064],
            [-0.0706, -0.0316,  0.0137]], device='cuda:0',dtype=torch.float16)

    output_distances, output_pt, sources = cam.get_depths(pos2.repeat(100,1),rotations2.repeat(100,1))
    print(output_distances.shape)
