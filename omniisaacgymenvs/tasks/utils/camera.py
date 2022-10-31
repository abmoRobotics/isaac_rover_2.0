import numpy as np
import matplotlib.pyplot as plt
import torch
import open3d as o3d
from omniisaacgymenvs.tasks.utils.heightmap_distribution import heightmap_distribution
from omniisaacgymenvs.tasks.utils.rover_depth import draw_depth
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

        self.device = device    
        self.partition = True    
        self.num_partitions = 1 # Slices the input data to reduce VRAM usage. 
        self.horizontal = 0.05  # Resolution of the triangle map
        #self.map_values = self._load_triangles()    # Load triangles into an [1200,1200, 100, 3, 3] array
        self.map_indices, self.triangles, self.vertices = self._load_triangles_with_indices()   # Load into [1200,1200, 100, 3] array
        self.heightmap_distribution = self._heightmap_distribution()    # Get distribution of points in the local reference frame of the rover
        self.shift = shift          # Shift of the map 
        self.dtype = torch.float16  # data type for performing the calculations

#TODO Remove this function
    def get_depths_test(self, sources, directions):
        # Test function
        triangles = self._height_lookup(self.map_values,sources[:,:,0:2],self.horizontal,self.shift)
        triangles = torch.tensor([[[[[0.0,0.0,0.0],[0.0,2.0,0.0],[2.0,0.0,0.0]],[[0.0,0.0,0.2],[0.0,2.0,0.2],[2.0,0.0,0.2]],[[0.0,0.0,0.3],[0.0,2.0,0.3],[2.0,0.0,0.3]]]]],device='cuda:0')
        dim0, dim1, dim2 = sources.shape[0], sources.shape[1], sources.shape[2]
        t_dim0, t_dim1, t_dim2, t_dim3, t_dim4 = triangles.shape[0], triangles.shape[1], triangles.shape[2], triangles.shape[3], triangles.shape[4] 
        s = sources.reshape(dim0*dim1,dim2)
        d = directions.reshape(dim0*dim1,dim2)
        t = triangles.reshape(t_dim0*t_dim1*t_dim2,t_dim3,t_dim4)

        distances,pt = self._ray_distance(s,d,t)
        distances = distances.reshape(dim0,dim1,t_dim2)
        pt = pt.reshape(dim0,dim1,t_dim2,3)
        # Find the maximum value of the 100 triangles to extract the "heightmap" value.
        indices =torch.max(distances,2).indices
        distances = torch.max(distances,2).values
        indices = indices.unsqueeze(dim=2).unsqueeze(dim=2).repeat(1,1,1,3)
        pt = pt.gather(dim=2,index=indices)
        pt = pt.squeeze(dim=2)
        return distances,pt


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
                
                distances,pt = self._ray_distance(s,d,t)
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
            output_distances = self._ray_distance(s,d,t)

        
        d1,d2,d3 = sources.shape[0],sources.shape[1] ,sources.shape[2] 
        #print((self.initialmemory3 - torch.cuda.memory_reserved(0))/1_000_000_000)
        try:
            pass
            #draw_depth(sources.reshape(d1*d2,d3),output_pt.reshape(d1*d2,d3))
        except:
            print("Isaac Sim not running")
        return output_distances, output_pt, sources

    def _load_triangles(self):
        """Loads triangles with explicit values, each triangle is a 3 x 3 matrix"""
        map_values = torch.load("tasks/utils/map_values.pt")
        map_values=map_values.swapaxes(0,1)
        map_values=map_values.swapaxes(1,2)
        return map_values

    def _load_triangles_with_indices(self):
        """Loads triangles with indicies to vertices, each triangle is a 3D vector"""
        map_indices = torch.load("tasks/utils/map_indices.pt")
        map_indices  = map_indices.swapaxes(0,1)
        map_indices  = map_indices.swapaxes(1,2)
        triangles = torch.load("tasks/utils/triangles.pt")
        vertices = torch.load("tasks/utils/vertices.pt")
        return map_indices, triangles, vertices

    def _ray_distance(self, sources: torch.Tensor, directions: torch.Tensor, triangles: torch.Tensor):
        """
        Checks if there is an intersection between a triangle and calculates the distance based on Möller–Trumbore intersection algorithm

        Parameters
        ----------
        sources : torch.tensor
            3D points of the ray source – dimension: [n_rays, 3]
        directions : torch.tensor
            Direction vector for the ray – dimension: [n_rays, 3]
        triangles : torch.tensor
            Triangles for ray intersection – dimension: [n_rays, 3, 3]
        """
        # This is for multiple rays intersection with ONE triangle each
        # Sources: [n_rays, 3]
        # Direction: [n_rays, 3]
        # Direction: [n_rays, 3, 3]

        # Sources is 
        # Normalize and opposite of direction
     
        #d = -(directions/directions.abs().sum())#(directions.swapaxes(0,1)[:] * 1/torch.sum(sources,dim=1)).swapaxes(0,1)
     
        # Normalize and opposite of direction
        d = -torch.nn.functional.normalize(directions) # 4000 x 3 
        
 
        a = triangles[:,2]      # 4000 x 3 
        b = triangles[:,1] - a  # 4000 x 3 
        c = triangles[:,0] - a  # 4000 x 3 

        g = sources - a     # 4000 x 3 

        self.a = a
        self.g = g
        self.sourcesLong = sources
        a=a.swapaxes(0,1)   # 3 x 4000   
        b=b.swapaxes(0,1)   # 3 x 4000
        c=c.swapaxes(0,1)   # 3 x 4000
        d=d.swapaxes(0,1)   # 3 x 4000
        g=g.swapaxes(0,1)   # 3 x 4000

        zeros = torch.zeros(g.shape[1],device=self.device,dtype=self.dtype)
        ones = torch.ones(g.shape[1],device=self.device,dtype=self.dtype)
        error_tensor = ones * -99.0

       # kk = b.float().cross(c.float()) 
       
       ## det2=torch.mm(torch.transpose(kk,0,1),d.float())
        detdet = b.cross(c) 
        
        # 4000 
        #det = (b[0][:] * c[1][:]* d[2][:] + c[0][:]*d[1][:]*b[2][:]+d[0][:]*b[1][:]*c[2][:]) - (b[2][:] * c[1][:]* d[0][:] + c[2][:]*d[1][:]*b[0][:]+d[2][:]*b[1][:]*c[0][:])

        det = (detdet[0][:]*d[0][:]+detdet[1][:]*d[1][:]+detdet[2][:]*d[2][:])
        del detdet
        
        # Calculate Scalar Triple Product for [g,c,d] 
        nn = g.cross(c)
        nn = (nn[0][:]*d[0][:]+nn[1][:]*d[1][:]+nn[2][:]*d[2][:])/det
        n = torch.where(det == zeros,error_tensor,nn)
        del nn

        # Calculate Scalar Triple Product for [b,g,d]
        mm = b.cross(g)
        mm = (mm[0][:]*d[0][:]+mm[1][:]*d[1][:]+mm[2][:]*d[2][:])/det
        m = torch.where(det == ones,error_tensor,mm) 
        del mm

        # Calculate Scalar Triple Product for [b,c,g]
        kk = b.cross(c)
        kk = (kk[0][:]*g[0][:]+kk[1][:]*g[1][:]+kk[2][:]*g[2][:])/det
        #k = torch.where(det == 0.0,-99.0,(b[0][:] * c[1][:]* g[2][:] + c[0][:]*g[1][:]*b[2][:]+g[0][:]*b[1][:]*c[2][:]) - (b[2][:] * c[1][:]* g[0][:] + c[2][:]*g[1][:]*b[0][:]+g[2][:]*b[1][:]*c[0][:]) / det)  
        k = torch.where(det == ones,error_tensor,kk) 
        del kk

        # filter out based on the condition ((n >= 0.0) & (m >= 0.0) & (n + m <= 1.0))
        k_after_check = torch.where(((n >= zeros) & (m >= zeros) & (n + m <= ones)),k,error_tensor)

        # Calucate the intersection point
        pt = sources -(d*k_after_check).swapaxes(0,1)
        
        #torch.cuda.empty_cache()


        #print((self.initialmemory2 - torch.cuda.memory_reserved(0))/1_000_000_000)
        #time.sleep(20)
        return k_after_check, pt

    def _depth_transform(self, rover_l, rover_r, rover_depth_points):
        """Transforms the local heightmap of the rover, into the global frame
        

        """
        # X:0, Y:1, Z:2
        # rover_r = rover_r.cpu()
        # rover_l = rover_l.cpu()
        # rover_depth_points = rover_depth_points.cpu()
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
        
        
        rover_dir = rover_dir.repeat(1,num_points-1,1)#.swapaxes(0,1)
        #Stack points in a [x, y, 3] matrix, and return
        sources = torch.stack((x_p[:,0:num_points-1], y_p[:,0:num_points-1], z_p[:,0:num_points-1]), 2)
        # print(sources.shape)
        # print(rover_dir.shape)
        #print(rover_dir)
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

    def _heightmap_distribution(self, x_limit=1.12, y_limit=1.8, square=False, y_start=0.5, delta=0.05, front_heavy=0, plot=False):
        """Returns distribution of heightmap points in the local frame of the robot"""
        point_distribution = []

        # If delta variable not set, exit.
        if delta == 0:
            # print("Need delta value!")
            exit()

        xd = 0
        yd = 0

        y = y_start
        while y < y_limit:
            
            x = 0

            delta += front_heavy

            flag = True
            if square==False:
                limit = self._limit_at_x(y)
                if x_limit < self._limit_at_x(y):
                    limit = x_limit
            else:
                limit = x_limit


            while x < limit:
                
                if x < -limit:
                    x += delta
                    xd += 1
                    flag = False

                if flag:
                    x -= delta
                    xd -= 1
                else:
                    #point_distribution.append([x, -y, -0.26878])
                    point_distribution.append([y, -x, -0.26878])
                    x += delta
                    xd += 1

            y += delta
            yd +=1

        point_distribution = np.round(point_distribution, 4)

        xd = (int)(xd/yd)*2-1

        if plot == True:
            fig, ax = plt.subplots()
            ax.scatter(point_distribution[:,0], point_distribution[:,1])
            ax.set_aspect('equal')
            plt.show()

        return torch.tensor(point_distribution, device=self.device)

    def _limit_at_x(self, x):
        return x*(4.3315)-0.129945

    def _OuterLine(self, x):
        y = -0.2308*x-0.03
        return y

    def _InnerLine(self, x):
        y = 0.7641*x-0.405
        return y

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

#TODO remove if not used    
    def _heightmap_overlay(self,dim, point_distrubution):
        zeros = torch.zeros_like(point_distrubution[:,0])
        ones = torch.ones_like(point_distrubution[:,0])
        belowOuter = point_distrubution[:,1] <= self._OuterLine(torch.abs(point_distrubution[:,0]))
        belowInner = point_distrubution[:,1] <= self._InnerLine(torch.abs(point_distrubution[:,0]))
        overlay = torch.where(belowInner, ones, zeros)

        return overlay

if __name__ == "__main__":
    a = torch.tensor([0.0,0.0],device='cuda:0')
    cam = Camera('cuda:0', a)

    # source = torch.tensor([[[-0.1,-0.1,1.0]]],device='cuda:0',dtype=torch.float16)
    # direction = torch.tensor([[[-0.5,-0.5,1]]],device='cuda:0',dtype=torch.float16)

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
    # rotationsz = torch.tensor([
    #         [0.0,  0.0, 0.0],
    #         [0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0]], device='cuda:0')
    # pos = torch.tensor([
    #         [3.5420, 4.5217, 1.7294],
    #         [3.5420, 4.5217, 0.7294]], device='cuda:0')
    # rotations = torch.tensor([
    #         [-0.0,  0.0, -0.0],
    #         [-0.0,  0.0, -0.0]], device='cuda:0')
    # pos4 = torch.tensor([
    #         [3.5420, 4.5217, 1.7294],
    #         [3.5420, 4.5217, 0.7294],
    #         [2.5420, 4.5217, 0.7294],
    #         [3.5420, 4.5217, 0.7294]], device='cuda:0')
    # rotations4 = torch.tensor([
    #         [0.0,  0.0, 0.0],
    #         [ 0.0,  0.0, 0.0],
    #         [ 0.0,  0.0, 0.0],
    #         [ 0.0,  0.0, 0.0]], device='cuda:0')
    # pos3 = torch.tensor([
    #         [3.5420, 4.5217, 0.7294]], device='cuda:0')
    # rotations3 = torch.tensor([
    #         [-0.0,  0.0, -0.0]], device='cuda:0')
    # pos5 = torch.tensor([
    #         [3.5421-0.2, 4.5217, 0.7294],
    # ], device='cuda:0')
    # rotations5 = torch.tensor([
    #        [0.0,  0.0, 0.0],], device='cuda:0')        
    #print(pos2.repeat(10,1).shape)

    output_distances, output_pt, sources = cam.get_depths(pos2.repeat(100,1),rotations2.repeat(100,1))
    # pos5 = torch.tensor([
    #         [3.5421, 4.5217, 0.7294]], device='cuda:0')

    # output_distances1, output_pt1, sources1, m1,n1,k1,k_after_check1, t1,a1,b1,c1,d1,g1,triangleslong1,sourcesLong1 = cam.get_depths(pos5,rotations5)
    # print(m1.shape)

    # print(a.shape)
    # print(a1.shape)
    # print(triangleslong.shape)
    # print("triangles long: ", torch.sum(triangleslong[100:200]-triangleslong1[0:100]))
    # print("Sources: ", torch.sum(sourcesLong[100:200]-sourcesLong1[0:100]))
    # print("D: ", torch.sum(d[:,100:200]-d1[:,0:100]))
    # print("G: ", torch.sum(g[100:200]-g1[0:100]))
    # print("M: ", torch.sum(m[100:200]-m1[0:100]))
    # print("N: ", torch.sum(n[100:200]-n1[0:100]))
    # print("K: ", torch.sum(k[100:200]-k1[0:100]))

    # #print(sourcesLong.shape)
    # #print(sourcesLong[0:100])


    #print(output_distances)
    # print(output_pt[0])
    # print(sources[0])
    # print(output_distances1[0])
    # # print(output_pt1[0])
    # # print(sources1[0])
    # # # # # #cam.plot_first()


