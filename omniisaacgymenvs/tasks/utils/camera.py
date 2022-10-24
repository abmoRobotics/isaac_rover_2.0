import numpy as np
import matplotlib.pyplot as plt
import torch

from omniisaacgymenvs.tasks.utils.heightmap_distribution import heightmap_distribution
from omniisaacgymenvs.tasks.utils.rover_depth import draw_depth

class Camera():
    def __init__(self, device, shift):
        self.device = device
        self.partition = True
        self.num_partitions = 1
        self.horizontal = 0.05
        self.map_values = self._load_triangles()
        #TODO Refeactor this
        self.map_values=self.map_values.swapaxes(0,1)
        self.map_values=self.map_values.swapaxes(1,2)
        self.heightmap_distribution = self._heightmap_distribution()
        self.shift = shift

    def get_depths_test(self, sources, directions):
        triangles = self._height_lookup(self.map_values,sources[:,:,0:2],self.horizontal,self.shift)
        triangles = torch.tensor([[[[[0.0,0.0,0.0],[0.0,2.0,0.0],[2.0,0.0,0.0]]]]],device='cuda:0')

        dim0, dim1, dim2 = sources.shape[0], sources.shape[1], sources.shape[2]
        t_dim0, t_dim1, t_dim2, t_dim3, t_dim4 = triangles.shape[0], triangles.shape[1], triangles.shape[2], triangles.shape[3], triangles.shape[4] 
        s = sources.reshape(dim0*dim1,dim2)
        d = directions.reshape(dim0*dim1,dim2)
        t = triangles.reshape(t_dim0*t_dim1*t_dim2,t_dim3,t_dim4)
        distances,pt = self._ray_distance(s,d,t)
        distances = distances.reshape(dim0,dim1,t_dim2)
        distances = torch.max(distances,2).values
        return distances,pt

    def get_depths(self, positions, rotations):
        # 
        sources, directions = self._depth_transform(positions,rotations,self.heightmap_distribution)
        print(sources.shape)
        depth_points = sources[:,:,0:2]
        # print('Output of height lookup')
        # print(depth_points.shape)
        triangles = self._height_lookup(self.map_values,depth_points,self.horizontal,self.shift)
        # print(triangles.shape)
        #print(sources.shape)
        # We partition the data to reduce VRAM usage
        if self.partition:
            partitions = self.num_partitions
            output_distances = None
            for i in range(partitions):
                step_size = int(sources.shape[1]/partitions)
                # # print(sources.shape)
                # # print(directions.shape)
                s = sources[:,i*step_size:i*step_size+step_size]
                d = directions[:,i*step_size:i*step_size+step_size]
                t = triangles[:,i*step_size:i*step_size+step_size]
                dim0, dim1, dim2 = s.shape[0], s.shape[1], s.shape[2]
                t_dim0, t_dim1, t_dim2, t_dim3, t_dim4 = t.shape[0], t.shape[1], t.shape[2], t.shape[3], t.shape[4]
                # # print(s.shape)
                # # print(d.shape)
                # # print(t.shape)
                s = s.reshape(dim0*dim1,dim2)
                s = s.repeat(t_dim2,1)
                # print(s.shape)
                d = d.reshape(dim0*dim1,dim2)
                d = d.repeat(t_dim2,1)
                # print(d.shape)
                t2 = t.reshape(t_dim0*t_dim1,t_dim2,t_dim3,t_dim4)
                # print('t')
                # print(t2.shape)
                t = t.reshape(t_dim0*t_dim1*t_dim2,t_dim3,t_dim4)
                # print(t.shape)
                distances,pt = self._ray_distance(s,d,t)
                
                distances = distances.reshape(dim0,dim1,t_dim2)
                pt = pt.reshape(dim0,dim1,t_dim2,3)
                # Find the maximum value of the 100 triangles to extract the "heightmap" value.
                # #print(torch.max(distances,2).values.shape)
                indices =torch.max(distances,2).indices
                distances = torch.max(distances,2).values
                indices = indices.unsqueeze(dim=2).unsqueeze(dim=2).repeat(1,1,1,3)
                pt = pt.gather(dim=2,index=indices)
                pt = pt.squeeze(dim=2)
               # pt = pt[:,:,indices]
                #distances = distances.reshape(dim0,dim1)
                #print(distances.shape)
                if output_distances is None:
                    output_distances = torch.tensor(distances,device=self.device,)
                    output_pt = torch.tensor(pt,device=self.device,)
                else: 
                    output_distances = torch.cat((output_distances,distances),1)
                    output_pt = torch.cat((output_pt,pt),1)

        else:
            dim0, dim1, dim2 = sources.shape[0], sources.shape[1], sources.shape[2]
            s = sources.reshape(dim0*dim1,dim2)
            d = directions.reshape(dim0*dim1,dim2)
            t = triangles.reshape(t_dim0*t_dim1*t_dim2,t_dim3,t_dim4)
            output_distances = self._ray_distance(s,d,t)
        #print(output_distances.shape)
        # print(output_distances)
        # print(indices)
        # print(output_pt[:,indices].shape)
        # print(output_pt.shape)
        
         
        draw_depth(sources.reshape(40,3),output_pt.reshape(40,3))
        return output_distances

    def _load_triangles(self):
        map_values = torch.load("tasks/utils/map_values.pt")
        return map_values

    def _ray_distance(self, sources: torch.Tensor, directions: torch.Tensor, triangles: torch.Tensor):
        # This is for multiple rays intersection with ONE triangle each
        # Sources: [n_rays, 3]
        # Direction: [n_rays, 3]
        # Direction: [n_rays, 3, 3]

        # Sources is 
        # Normalize and opposite of direction
        #print(directions)
        #d = -(directions/directions.abs().sum())#(directions.swapaxes(0,1)[:] * 1/torch.sum(sources,dim=1)).swapaxes(0,1)
        d = -torch.nn.functional.normalize(directions)
        #print(d)
        # print(sources.shape)
        # a = torch.ones([k_len,3], device='cuda:0') 
        # b = torch.ones([k_len,3], device='cuda:0')*2 - a
        # c = torch.ones([k_len,3], device='cuda:0')*3 - a
        # print(triangles.shape)
        # a = triangles[:,0]
        # b = triangles[:,1] - a
        # c = triangles[:,2] - a
        a = triangles[:,2]
        b = triangles[:,1] - a
        c = triangles[:,0] - a
        #print(c)
        #print(triangles[0])
        #print(sources[0])

        g = sources - a 

        # print(b.shape)
        # print(c.shape)
        # print(d.shape)
        # print(g.shape)
        # print("A")
        # print(a)
        # print(b)
        # print(c)
        # print(d)
        # print(g)
        # print("stop")
        #torch.cuda.synchronize()
        a=a.swapaxes(0,1)
        b=b.swapaxes(0,1)
        c=c.swapaxes(0,1)
        d=d.swapaxes(0,1)
        g=g.swapaxes(0,1)
        
        #for i in range(10):
        # print(d)
        # print(c)
        kk = b.float().cross(c.float())
        # print(kk.shape)
        #kk = kk* b
        det2=torch.mm(torch.transpose(kk,0,1),d.float())
        # print(det2)
        det = (b[0][:] * c[1][:]* d[2][:] + c[0][:]*d[1][:]*b[2][:]+d[0][:]*b[1][:]*c[2][:]) - (b[2][:] * c[1][:]* d[0][:] + c[2][:]*d[1][:]*b[0][:]+d[2][:]*b[1][:]*c[0][:])
        # print(det)
        #det = torch.linalg.det(detdet)
        # print(det)
        #del a

        nn = g.half().cross(c.half())
        nn = (nn[0][:]*d[0][:]+nn[1][:]*d[1][:]+nn[2][:]*d[2][:])/det
        #n = torch.where(det == 0,-99.0,(g[0][:] * c[1][:]* d[2][:] + c[0][:]*d[1][:]*g[2][:]+d[0][:]*g[1][:]*c[2][:]) - (g[2][:] * c[1][:]* d[0][:] + c[2][:]*d[1][:]*g[0][:]+d[2][:]*g[1][:]*c[0][:]) / det)
        n = torch.where(det == 0,-99.0,nn)
        mm = b.half().cross(g.half())
        mm = (mm[0][:]*d[0][:]+mm[1][:]*d[1][:]+mm[2][:]*d[2][:])/det
        #m = torch.where(det == 0,-99.0,(b[0][:] * g[1][:]* d[2][:] + g[0][:]*d[1][:]*b[2][:]+d[0][:]*b[1][:]*g[2][:]) - (b[2][:] * g[1][:]* d[0][:] + g[2][:]*d[1][:]*b[0][:]+d[2][:]*b[1][:]*g[0][:]) / det) 
        m = torch.where(det == 0,-99.0,mm) 
        
        # g,c,d
        #n = g[0][:]*b[1][:]*d[2][:]
        #del d
        k = torch.where(det == 0,-99.0,(b[0][:] * c[1][:]* g[2][:] + c[0][:]*g[1][:]*b[2][:]+g[0][:]*b[1][:]*c[2][:]) - (b[2][:] * c[1][:]* g[0][:] + c[2][:]*g[1][:]*b[0][:]+g[2][:]*b[1][:]*c[0][:]) / det)  
        # print(det)
        # print(n)
        # print(m)
        # print(k)
        #TODO implement append functionality
        #k_after_check = torch.where(((n >= 0.0) & (m >= 0.0) & (n + m <= 1.0) & (k >= 0.0)),k,-99.0) 
        # print("n")
        # print(n)
        # print(m)
        k_after_check = torch.where(((n >= 0.0) & (m >= 0.0) & (n + m <= 1.0)),k,-99.0)
        # print(sources.shape)
        # # print(directions)
        # print(((-d*k_after_check)[:,0]).shape)
        # print(((-d*k_after_check)).shape)
        pt = sources -(d*k_after_check).swapaxes(0,1)
       # pt = torch.where(k_after_check)
        # print(k.shape)
        return k_after_check, pt

    def _depth_transform(self, rover_l, rover_r, rover_depth_points):
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
        # print("roverdir")
        

        rover_dir = rover_dir.repeat(1,num_points-1,1)#.swapaxes(0,1)
        #Stack points in a [x, y, 3] matrix, and return
        sources = torch.stack((x_p[:,0:num_points-1], y_p[:,0:num_points-1], z_p[:,0:num_points-1]), 2)
        # print(sources.shape)
        # print(rover_dir.shape)
        return sources, rover_dir

    def _heightmap_distribution(self, x_limit=0.2, y_limit=0.2, square=False, y_start=0.0, delta=0.1, front_heavy=0, plot=False):

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
                    point_distribution.append([x, -y, 0])
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
        triangles = triangle_matrix[x, y]
        triangles = triangles.reshape([depth_points.shape[0],depth_points.shape[1],triangle_matrix.shape[2],triangle_matrix.shape[3],triangle_matrix.shape[4]])

        # Return the found heights
        return triangles
    
    def _heightmap_overlay(self,dim, point_distrubution):
        zeros = torch.zeros_like(point_distrubution[:,0])
        ones = torch.ones_like(point_distrubution[:,0])
        belowOuter = point_distrubution[:,1] <= self._OuterLine(torch.abs(point_distrubution[:,0]))
        belowInner = point_distrubution[:,1] <= self._InnerLine(torch.abs(point_distrubution[:,0]))
        overlay = torch.where(belowInner, ones, zeros)

        return overlay

# a = torch.tensor([0.0,0.0],device='cuda:0')
# cam = Camera('cuda:0', a)

# source = torch.tensor([[[-0.1,-0.1,1.0]]],device='cuda:0',dtype=torch.float64)
# direction = torch.tensor([[[-0.5,-0.5,1]]],device='cuda:0',dtype=torch.float64)
# #print(source.shape)
# #print(direction.shape)

# b,pt = cam.get_depths_test(source,direction)
# #print(b)
# print(b,pt)