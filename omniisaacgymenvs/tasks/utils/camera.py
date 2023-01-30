import numpy as np
import matplotlib.pyplot as plt
import torch
import open3d as o3d
from omniisaacgymenvs.tasks.utils.heightmap_distribution import heightmap_distribution
from omniisaacgymenvs.tasks.utils.debug_utils import draw_depth
from omniisaacgymenvs.tasks.utils.ray_casting import ray_distance
from omniisaacgymenvs.utils.heightmap_distribution import Heightmap
import time

class Camera():
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
        self.heightmap = Heightmap(self.device)
        self.num_partitions = 4 # Slices the input data to reduce VRAM usage. 
        self.horizontal = 0.1  # Resolution of the triangle map
        #self.map_values = self._load_triangles()    # Load triangles into an [1200,1200, 100, 3, 3] array
        self.map_indices, self.triangles, self.vertices = self._load_triangles_with_indices()   # Load into [1200,1200, 100, 3] array
        self.heightmap_distribution = self.heightmap.get_distribution() # Get distribution of points in the local reference frame of the rover
        self.num_exteroceptive = self.heightmap_distribution.shape[0] 

        self.shift = shift          # Shift of the map 
        self.dtype = torch.float16  # data type for performing the calculations

    def get_num_exteroceptive(self):
        return self.num_exteroceptive

    def get_depths(self, positions, rotations):
        """
        Returns a depthmap for a robot based on position and rotation

        Parameters
        ----------
        positions : torch.tensor
            Device for execution
        rotations : torch.tensor
            translation of the map, relative to the global reference frame of the simulation
        """
        sources, directions = self._depth_transform(positions,rotations,self.heightmap_distribution)
        depth_points = sources[:,:,0:2]
        triangles,triangles_with_indices = self._height_lookup(self.map_indices,depth_points,self.horizontal,self.shift)
        #triangles = self.vertices[self.triangles[triangles_with_indices.long()].long()]

        # We partition the data to reduce VRAM usage
        if self.partition:
            partitions = self.num_partitions 
            output_distances = None
            for i in range(partitions if sources.shape[1]%partitions == 0 else partitions + 1):
                if i == partitions + 1:
                    step_size = int(sources.shape[1]%partitions)
                step_size = int(sources.shape[1]/partitions)
                triangles = self.vertices[self.triangles[triangles_with_indices[:,i*step_size:i*step_size+step_size].long()].long()]

                s = sources[:,i*step_size:i*step_size+step_size]
     
                d = directions[:,i*step_size:i*step_size+step_size]
                #t = triangles[:,i*step_size:i*step_size+step_size]
                t = triangles
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
         
                distances,pt = ray_distance(s,d,t)
                distances = distances.reshape(dim0,dim1,t_dim2)
     
                pt = pt.reshape(dim0,dim1,t_dim2,3)
                # Find the maximum value of the 100 triangles to extract the "heightmap" value.
                
                indices =torch.min(distances,2).indices
                distances = torch.min(distances,2).values
                indices = indices.unsqueeze(dim=2).unsqueeze(dim=2).repeat(1,1,1,3)
                pt = pt.gather(dim=2,index=indices)
                pt = pt.squeeze(dim=2)
                
                if output_distances is None:
                    output_distances = distances.clone().detach()
                    output_pt = pt.clone().detach()
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
        # self.debug = True
        if self.debug:
            try:
                draw_depth(sources.reshape(d1*d2,d3),output_pt.reshape(d1*d2,d3))
            except:
                print("Isaac Sim not running")
        return output_distances, output_pt, sources

    def _load_triangles(self):
        """Loads triangles with explicit values, each triangle is a 3 x 3 matrix"""
        map_values = torch.load("tasks/utils/terrain/knn_terrain/map_values.pt")
        map_values=map_values.swapaxes(0,1)
        map_values=map_values.swapaxes(1,2)
        return map_values

    def _load_triangles_with_indices(self):
        """Loads triangles with indicies to vertices, each triangle is a 3D vector"""
        map_indices = torch.load("tasks/utils/terrain/knn_terrain/map_indices.pt")
        map_indices  = map_indices.swapaxes(0,1)
        map_indices  = map_indices.swapaxes(1,2)
        triangles = torch.load("tasks/utils/terrain/knn_terrain/triangles.pt")
        vertices = torch.load("tasks/utils/terrain/knn_terrain/vertices.pt")
        return map_indices, triangles, vertices



    def _depth_transform(self, rover_l, rover_r, rover_depth_points):
        """Transforms the local heightmap of the rover, into the global frame"""
        # X:0, Y:1, Z:2

        # Get number of points and number of robots from input
        num_points = rover_depth_points.size()[0] + 1 # +1 to add plane-normal
        num_robots = rover_r.size()[0]

        # Expand depth point vectors to be martix of size[1, x](from vector of size[x])
        x = rover_depth_points[:,0].expand(1, num_points-1)
        y = rover_depth_points[:,1].expand(1, num_points-1)
        z = rover_depth_points[:,2].expand(1, num_points-1)

        # Add [0, 0, -1] as the last point
        x = torch.cat((x, torch.tensor([0], device=self.device).expand(1, 1)), 1)
        y = torch.cat((y, torch.tensor([0], device=self.device).expand(1, 1)), 1)
        z = torch.cat((z, torch.tensor([-1], device=self.device).expand(1, 1)), 1)

        # Compute sin and cos to all angles - Sizes: [n, 1]
        sinxr = torch.transpose(torch.sin(-rover_r[:,0].expand(1, num_robots)), 0, 1)
        cosxr = torch.transpose(torch.cos(-rover_r[:,0].expand(1, num_robots)), 0, 1)
        sinyr = torch.transpose(torch.sin(-rover_r[:,1].expand(1, num_robots)), 0, 1)
        cosyr = torch.transpose(torch.cos(-rover_r[:,1].expand(1, num_robots)), 0, 1)
        sinzr = torch.transpose(torch.sin(-rover_r[:,2].expand(1, num_robots)), 0, 1)
        coszr = torch.transpose(torch.cos(-rover_r[:,2].expand(1, num_robots)), 0, 1)

        # Expand location vector to be of size[x, y], from size[x]: [n, p]
        rover_xl = torch.transpose(rover_l[:, 0].expand(num_points,num_robots), 0, 1)
        rover_yl = torch.transpose(rover_l[:, 1].expand(num_points,num_robots), 0, 1)
        rover_zl = torch.transpose(rover_l[:, 2].expand(num_points,num_robots), 0, 1)

        # Transform points in the pointcloud
        x_p = rover_xl + sinzr * (y*cosxr + z*sinxr) + coszr * (x*cosyr - sinyr*(z*cosxr - y*sinxr))
        y_p = rover_yl + coszr * (y*cosxr + z*sinxr) - sinzr * (x*cosyr - sinyr*(z*cosxr - y*sinxr))
        z_p = rover_zl + x*sinyr + cosyr*(z*cosxr - y*sinxr)
        
        # Extract the plane-normal as the last point
        x = (x_p[:,num_points-1]-rover_l[:, 0]).unsqueeze(1)
        y = (y_p[:,num_points-1]-rover_l[:, 1]).unsqueeze(1)
        z = (z_p[:,num_points-1]-rover_l[:, 2]).unsqueeze(1)
        rover_dir = torch.cat((x, y, z),1).unsqueeze(1)
     
        rover_dir = rover_dir.repeat(1,num_points-1,1)
        
        #Stack points in a [x, y, 3] matrix, and return
        sources = torch.stack((x_p[:,0:num_points-1], y_p[:,0:num_points-1], z_p[:,0:num_points-1]), 2)

        return sources.type(self.dtype), rover_dir.type(self.dtype)



    def _plot_mesh(self, triangles_with_indices):
        """Function for plotting triangles"""
        tri = self.triangles[triangles_with_indices.long()]
        vert = self.vertices
        mesh = o3d.geometry.TriangleMesh()
        np_triangles = tri.cpu().detach().numpy()
        np_vertices = vert.cpu().detach().numpy()
        mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
        o3d.visualization.draw_geometries([mesh])
    
    def plot_first(self):
        """Function that plots the first triangles in an array of triangles"""
        triangles_with_indices = self.map_indices[200, 100]
        triangles_with_indices = triangles_with_indices.flatten()
        self._plot_mesh(triangles_with_indices)

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
        #triangles = triangle_matrix[x, y]
        #triangles = triangles.reshape([depth_points.shape[0],depth_points.shape[1],triangle_matrix.shape[2],triangle_matrix.shape[3],triangle_matrix.shape[4]])
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

    output_distances, output_pt, sources = cam.get_depths(pos2.repeat(300,1),rotations2.repeat(300,1))
    print(output_distances.shape)
