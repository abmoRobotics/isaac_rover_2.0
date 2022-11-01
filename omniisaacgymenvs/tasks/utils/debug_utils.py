import torch
import numpy as np
from omni.isaac.debug_draw import _debug_draw
import random



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