import dis
from importlib_metadata import distribution
import numpy as np
#import heigtmap_distribution
import matplotlib.pyplot as plt
import torch

def heightmap_distribution(x_limit, y_limit, square=False, y_start=0.296, delta=0, front_heavy=0, plot=True):

    point_distribution = []

    # If delta variable not set, exit.
    if delta == 0:
        print("Need delta value!")
        exit()

    xd = 0
    yd = 0

    y = y_start
    while y < y_limit:
        
        x = 0

        delta += front_heavy

        flag = True
        if square==False:
            limit = limit_at_x(y)
            if x_limit < limit_at_x(y):
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
                point_distribution.append([x, -y])
                x += delta
                xd += 1

        y += delta
        yd +=1

    point_distribution = np.round(point_distribution, 4)

    xd = (int)(xd/yd)*2-1

    dim = [xd, yd]

    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(point_distribution[:,0], point_distribution[:,1])
        ax.set_aspect('equal')
        plt.show()

    return dim, point_distribution

def limit_at_x(x):
    return x*(4.3315)-0.129945

def OuterLine(x):
    y = -0.2308*x-0.03
    return y

def InnerLine(x):
    y = 0.7641*x-0.405
    return y

def heightmap_overlay(dim, point_distrubution):
    zeros = torch.zeros_like(point_distrubution[:,0])
    ones = torch.ones_like(point_distrubution[:,0])
    belowOuter = point_distrubution[:,1] <= OuterLine(torch.abs(point_distrubution[:,0]))
    belowInner = point_distrubution[:,1] <= InnerLine(torch.abs(point_distrubution[:,0]))
    overlay = torch.where(belowInner, ones, zeros)

    return overlay
