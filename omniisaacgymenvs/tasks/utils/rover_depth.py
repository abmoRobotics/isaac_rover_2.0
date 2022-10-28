from ctypes import sizeof
import torch
import numpy as np
from omni.isaac.debug_draw import _debug_draw
import random



def depth_transform(rover_l, rover_r, rover_depth_points):
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
    rover_dir = [x_p[:,num_points-1]-rover_l[:, 0], y_p[:,num_points-1]-rover_l[:, 1], z_p[:,num_points-1]-rover_l[:, 2]]

    #Stack points in a [x, y, 3] matrix, and return
    return torch.stack((x_p[:,0:num_points-1], y_p[:,0:num_points-1], z_p[:,0:num_points-1]), 2), rover_dir

def draw_depth(heightmap_points: torch.tensor, depth_points: torch.tensor,):
    draw = _debug_draw.acquire_debug_draw_interface()
    # print(heightmap_points.shape)
    # print(depth_points.shape)
    rover_distribution = heightmap_points.tolist()
    depth_points = depth_points.tolist()
    N = len(rover_distribution)

    rover_distributionZ = []
    rover_distribution2 = []
    
    for i in range(N):
        rover_distributionZ.append(rover_distribution[i][2]+0.1)

    for i in range(N):
        rover_distribution2.append([rover_distribution[i][0], rover_distribution[i][1], rover_distributionZ[i]])

    colors = [3 for _ in range(N)]
    sizes = [[3] for _ in range(N)]
    draw.clear_lines()
    #print(rover_distribution)
    #print(depth_points)
    draw.draw_lines(rover_distribution, depth_points, [(0.9, 0.5, 0.1, 0.9)]*N, [3]*N)
    # if depth_points:
    #     draw.draw_lines(rover_distribution, rover_distribution2, [(0.9, 0.5, 0.1, 0.9)]*N, [3]*N)
    # else:
    #     draw.draw_lines(rover_distribution, depth_points, [(0.9, 0.5, 0.1, 0.9)]*N, [3]*N)


def draw_point(points, color):
    draw = _debug_draw.acquire_debug_draw_interface()
    N = 1

    point_list = points.tolist()
    draw.clear_points()
    draw.draw_points([point_list], color, [55])

def draw_coord():
    draw = _debug_draw.acquire_debug_draw_interface()
    N = 4
    point_list = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
    colors = [[0.1, 0.1, 0.1, 0.9],[1.0, 0.0, 0.0, 0.9],[0.0, 1.0, 0.0, 0.9],[0.0, 0.0, 1.0, 0.9]]
    draw.clear_points()
    draw.draw_points(point_list, colors, [55]*N)