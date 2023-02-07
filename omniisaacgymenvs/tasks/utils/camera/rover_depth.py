from ctypes import sizeof
import torch
import numpy as np
from omni.isaac.debug_draw import _debug_draw
import random


"""

    depth_transform(rover_l, rover_r, rover_depth_points):
    This function transfroms the points on a rover's depth-sensor pointcloud
    from the rover's local coordinate frame to the global coordinate frame. 
    
    Parameters:
        rover_l (torch.tensor): This is a tensor containing the current location of each rover.
            It is of shape [num_robots, 3], where each row contains the x, y, z coordinates of the rover.
        rover_r (torch.tensor): This is a tensor containing the current orientation of each rover.
            It is of shape [num_robots, 3], where each row contains the roll, pitch, yaw rotation of the rover.
        rover_depth_points (torch.tensor): This is a tensor containing the depth-sensor points of each rover.
            It is of shape [num_points, 3], where each row contains the x, y, z coordinates of the point.
            The points are in the rover's local coordinate frame.
    
    Returns:
        pointcloud (torch.tensor): This is a tensor containing the pointcloud of all the rovers in the global coordinate frame.
            It is of shape [num_robots, num_points, 3], where each row contains the x, y, z coordinates of the point.
        rover_dir (list): This is a list containing the orthogonal directions of each rover in the global coordinate frame.
            It is of shape [num_robots, 3], where each row contains the x, y, z components of the direction.
"""
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

