import dis
from importlib_metadata import distribution
import numpy as np
#import heigtmap_distribution
import matplotlib.pyplot as plt
import torch
import operator
import numpy as np
import math

def heightmap_distribution(plot=False, device='cuda:0'):

    # Define the borders of the area using lines. Define where points should be with respect to line.
    border = [[[1.220,0.118],[4.4455,3.150],'over'],[[-1.220,0.118],[-4.4455,3.150],'over'],[[1.220,0.118],[-1.220,0.118],'over']] 
    HDborder = [[[1.220,0.118],[4.4455,3.150],'over'],[[-1.220,0.118],[-4.4455,3.150],'over'],[[1.220,0.118],[-1.220,0.118],'over']] 
    BeneathBorder = [[[0.32,0],[0.320,1],'left'],[[-0.320,0],[-0.320,1],'right'],[[-0.320,-0.5],[0.320,-0.5],'over'],[[-0.320,0.6],[0.320,0.6],'under']] 

    delta_coarse = 0.2
    delta_fine = 0.05

    point_distribution = []

    z_offset = -0.26878

    see_beneath = False

    # The coarse map
    y = -10
    while y < 10:
    
        x = -10
        
        while x < 10:
            x += delta_coarse
            if inside_borders([x, y], border) and inside_circle([x, y], [0,0], 5.0):
                point_distribution.append([y, x, z_offset])

        y += delta_coarse

    # The fine map
    y = -10
    while y < 10:
    
        x = -10
        
        while x < 10:
            x += delta_fine
            if inside_borders([x, y], border) and inside_circle([x, y], [0,0], 1.2):
                if [x,y] not in point_distribution:
                    point_distribution.append([y, x, z_offset])

        y += delta_fine

    if see_beneath:

        y = -10
        while y < 10:
        
            x = -10
            
            while x < 10:
                x += delta_fine
                if inside_borders([x, y], BeneathBorder) and inside_circle([x, y], [0,0], 1.2):
                    if [x,y] not in point_distribution:
                        point_distribution.append([y, x, z_offset])

            y += delta_fine        



    point_distribution = np.round(point_distribution, 4)


    print(len(point_distribution))

    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(point_distribution[:,0], point_distribution[:,1])
        ax.set_aspect('equal')
        plt.show()

    distribution = torch.tensor(point_distribution, device=device)

    return distribution

def inside_borders(point, borderLines):

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

def inside_circle(point, centre, radius):

    point = np.subtract(point,centre)

    dist = math.sqrt(point[0]**2 + point[1]**2)

    if dist < radius:
        return True
    else:
        return False

if __name__ == "__main__":
    heightmap_distribution(plot=True)