import dis
from importlib_metadata import distribution
import numpy as np
#import heigtmap_distribution
import matplotlib.pyplot as plt
import torch
import operator
import numpy as np
import math

class Heightmap():
    def __init__(self, device):
        self.device = device
        
        # Define the borders of the area using lines. Define where points should be with respect to line.
        self.coarse_border = [[[1.220,0.118],[4.4455,3.150],'over'],[[-1.220,0.118],[-4.4455,3.150],'over'],[[1.220,0.118],[-1.220,0.118],'over']] 
        self.coarse_radius = 5.0

        self.fine_border = [[[1.220,0.118],[4.4455,3.150],'over'],[[-1.220,0.118],[-4.4455,3.150],'over'],[[1.220,0.118],[-1.220,0.118],'over']] 
        self.fine_radius = 1.2

        self.beneath_border = [[[0.32,0],[0.320,1],'left'],[[-0.320,0],[-0.320,1],'right'],[[-0.320,-0.5],[0.320,-0.5],'over'],[[-0.320,0.6],[0.320,0.6],'under']] 

        self.delta_coarse = 0.2
        self.delta_fine = 0.05
        
        self.see_beneath = True
        self.HD_enabled = True

        self.z_offset = -0.26878

        self.heightmap_distribution() # Create the heightmap distribution

        #self.calculate_grids() # Create the heightmap in a grid

    def heightmap_distribution(self, plot=False):
        
        point_distribution = []

        coarse_idx = []
        fine_idx = []
        beneath_idx = []

        # The coarse map
        y = -10
        while y < 10:
        
            x = -10
            
            while x < 10:
                x += self.delta_coarse
                if self._inside_borders([x, y], self.coarse_border) and self._inside_circle([x, y], [0,0], self.coarse_radius): # REMEMBER TO CHANGE BELOW
                    point_distribution.append([x, y, self.z_offset])

            y += self.delta_coarse

        for idx, point in enumerate(point_distribution):
            if self._inside_borders(point[0:2], self.coarse_border) and self._inside_circle(point[0:2], [0,0], self.coarse_radius): # REMEMBER TO CHANGE ABOVE
                coarse_idx.append(idx)

        # The fine map
        if self.HD_enabled:
            y = -10
            while y < 10:
            
                x = -10
                
                while x < 10:
                    x += self.delta_fine
                    if self._inside_borders([x, y], self.fine_border) and self._inside_circle([x, y], [0,0], 1.2):
                        if [x, y, self.z_offset] not in point_distribution:
                            point_distribution.append([x, y, self.z_offset])

                y += self.delta_fine

            for idx, point in enumerate(point_distribution):
                if self._inside_borders(point[0:2], self.fine_border) and self._inside_circle(point[0:2], [0,0], self.fine_radius): # REMEMBER TO CHANGE ABOVE
                    fine_idx.append(idx)

        # Points underneath belly pan
        if self.see_beneath:
            y = -10
            while y < 10:
            
                x = -10
                
                while x < 10:
                    x += self.delta_fine
                    if self._inside_borders([x, y], self.beneath_border) and self._inside_circle([x, y], [0,0], self.fine_radius): # REMEMBER TO CHANGE BELOW
                        if [x, y, self.z_offset] not in point_distribution:
                            point_distribution.append([x, y, self.z_offset])

                y += self.delta_fine        

            for idx, point in enumerate(point_distribution):
                if self._inside_borders(point[0:2], self.beneath_border) and self._inside_circle(point[0:2], [0,0], self.fine_radius): # REMEMBER TO CHANGE ABOVE
                    beneath_idx.append(idx)


        point_distribution = np.round(point_distribution, 4)

        p_distribution = torch.tensor(point_distribution, device=self.device)

        #Swap X and Y axes - They are different in simulation
        self.distribution = torch.index_select(p_distribution, 1, torch.tensor([1,0,2], device=self.device))

        self.coarse_idx = torch.tensor(coarse_idx, device=self.device)
        self.fine_idx = torch.tensor(fine_idx, device=self.device)
        self.beneath_idx = torch.tensor(beneath_idx, device=self.device)

        if plot == True:
            fig, ax = plt.subplots()
            ax.scatter(point_distribution[:,0], point_distribution[:,1])
            ax.set_aspect('equal')
            plt.show()

    def _get_depth_from_idx(self, idx, rays):
        return rays[:,idx]

    def _get_depth_from_grid_idx(self, idx, rays, shape):
        return rays[idx]

    def get_sparse_grid(self, rays):
        return self._get_depth_from_idx(idx, rays)

    def get_sparse_vector(self, rays):
        return self._get_depth_from_idx(self.coarse_idx, rays)

    def get_dense_grid(self, rays):
        return self._get_depth_from_idx(idx, rays)

    def get_dense_vector(self, rays):
        return self._get_depth_from_idx(self.fine_idx, rays)

    def get_beneath_grid(self, rays):
        return self._get_depth_from_idx(idx, rays)

    def get_beneath_vector(self, rays):
        return self._get_depth_from_idx(self.beneath_idx, rays)

    def get_num_sparse_vector(self):
        return self.coarse_idx.shape[0]

    def get_num_dense_vector(self):
        return self.fine_idx.shape[0]
        
    def get_num_beneath_vector(self):
        return self.beneath_idx.shape[0]

    def get_distribution(self):
        return self.distribution

    def _inside_borders(self, point, borderLines):

        x, y = point

        passCondition = True

        for line in borderLines:
            a = np.subtract(line[0],line[1])
            if a[0] == 0:
                a = float("inf")
            else:
                a = a[1]/a[0]
            
            b = line[0][1]-a*line[0][0] # b = y - a*x


            if a == 0:
                if y > b and line[2] == 'below':
                    passCondition = False
                if y < b and line[2] == 'over':
                    passCondition = False
                continue
            
            if a == float("inf"):
                if x < line[0][0] and line[2] == 'right':
                    passCondition = False
                if x > line[0][0] and line[2] == 'left':
                    passCondition = False
                continue


            if y < a*x+b and line[2] == 'over':
                passCondition = False
            if y > a*x+b and line[2] == 'below':
                passCondition = False
            if x < (y-b)/a and line[2] == 'right':
                passCondition = False
            if x < (y-b)/a and line[2] == 'left':
                passCondition = False    

        return passCondition

    def _inside_circle(self, point, centre, radius):

        point = np.subtract(point,centre)

        dist = math.sqrt(point[0]**2 + point[1]**2)

        if dist < radius:
            return True
        else:
            return False


if __name__ == '__main__':

    heightmap = Heightmap('cuda:0')

    heightmap.get_beneath_vector(torch.ones_like(heightmap.get_distribution()))

    exit()