from ctypes import sizeof
import torch
import numpy as np
from omni.isaac.debug_draw import _debug_draw
import random



def depth_transform(rover_r, rover_l, rover_depth_points):
    # X:0, Y:1, Z:2

    # Get number of points and number of robots from input
    num_points = rover_depth_points.size()[0] + 1 # +1 to add plane-normal
    num_robots = rover_r.size()[0]

    # Expand depth point vectors to be martix of size[1, x](from vector of size[x])
    x = rover_depth_points[:,0].expand(1, num_points-1)
    y = rover_depth_points[:,1].expand(1, num_points-1)
    z = rover_depth_points[:,2].expand(1, num_points-1)

    # Add [0, 0, -1] as the last point
    x = torch.cat((x, torch.tensor([0], device='cuda:0').expand(1, 1)), 1)
    y = torch.cat((y, torch.tensor([0], device='cuda:0').expand(1, 1)), 1)
    z = torch.cat((z, torch.tensor([-1], device='cuda:0').expand(1, 1)), 1)

    # Compute sin and cos to all angles - Sizes: [n, 1]
    sinxr = torch.transpose(torch.sin(rover_r[:,0].expand(1, num_robots)), 0, 1)
    cosxr = torch.transpose(torch.cos(rover_r[:,0].expand(1, num_robots)), 0, 1)
    sinyr = torch.transpose(torch.sin(rover_r[:,1].expand(1, num_robots)), 0, 1)
    cosyr = torch.transpose(torch.cos(rover_r[:,1].expand(1, num_robots)), 0, 1)
    sinzr = torch.transpose(torch.sin(rover_r[:,2].expand(1, num_robots)), 0, 1)
    coszr = torch.transpose(torch.cos(rover_r[:,2].expand(1, num_robots)), 0, 1)

    # Expand location vector to be of size[x, y], from size[x]: [n, p]
    rover_xl = torch.transpose(rover_l[:, 0].expand(num_points,num_robots), 0, 1)
    rover_yl = torch.transpose(rover_l[:, 1].expand(num_points,num_robots), 0, 1)
    rover_zl = torch.transpose(rover_l[:, 2].expand(num_points,num_robots), 0, 1)

    # Transform points in the pointcloud
    x_p = rover_xl + cosyr * (x*coszr + y*sinzr) - z*sinyr
    y_p = rover_yl + cosxr * (y*coszr - x*sinzr) + sinxr * (z*cosyr + sinyr*(x*coszr + y*sinzr))
    z_p = rover_zl + cosxr * (z*cosyr + sinyr * (x*coszr + y*sinzr)) - sinxr * (y*coszr - x*sinzr)
    
    # Extract the plane-normal as the last point
    rover_dir = [x_p[:,num_points-1]-rover_l[:, 0], y_p[:,num_points-1]-rover_l[:, 1], z_p[:,num_points-1]-rover_l[:, 2]]

    #Stack points in a [x, y, 3] matrix, and return
    return torch.stack((x_p[:,0:num_points-1], y_p[:,0:num_points-1], z_p[:,0:num_points-1]), 2), rover_dir

def draw_depth(depth_points: torch.tensor, points ):
    draw = _debug_draw.acquire_debug_draw_interface()
    
    rover_distribution = depth_points.tolist()

    N = len(rover_distribution)

    print(N)

    rover_distributionZ = []
    
    for i in range(N):
        rover_distributionZ[i] = rover_distribution[i, 2]+0.1
    
    print(len(rover_distributionZ))

    rover_distribution2 = [rover_distribution[0], rover_distribution[1], rover_distributionZ]

    colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(N)]
    sizes = [random.randint(1, 25) for _ in range(N)]

    draw.draw_lines(rover_distribution, rover_distribution2, colors, sizes)

    print(len(point_list_1[0]))
    print(len(rover_distribution[0]))