#!/usr/bin/env python
from binascii import crc32
import numpy as np
#from .locomotion_modes import LocomotionMode
import math

import torch
import time





# This is the new kinematics function
# The inputs are m/s and rad/s. The dimensions are for the mars rover 2.0 project - units in meters.
# The code is made with basis in appendix D in the repport "Mapless Mobile Navigation on Mars using Reinforcement Learning"
@torch.jit.script
def Ackermann(lin_vel, ang_vel, device='cuda:0'):
    # type: (Tensor, Tensor, str) -> Tuple[Tensor, Tensor]
    # All measurements in Meters!
    num_robots = lin_vel.shape[0]
    wheel_diameter = 0.2 
    # Locations of the wheels, with respect to center(between middle wheels) (X is right, Y is forward)
    wheel_FL = torch.unsqueeze(torch.transpose(torch.tensor(  [[-0.385],[0.438]],  device=device).repeat(1,num_robots), 0, 1),0)
    wheel_FR = torch.unsqueeze(torch.transpose(torch.tensor(  [[0.385],[0.438]],   device=device).repeat(1,num_robots), 0, 1),0)
    wheel_ML = torch.unsqueeze(torch.transpose(torch.tensor(  [[-0.447],[0.0]],    device=device).repeat(1,num_robots), 0, 1),0)
    wheel_MR = torch.unsqueeze(torch.transpose(torch.tensor(  [[0.447],[0.0]],     device=device).repeat(1,num_robots), 0, 1),0)
    wheel_RL = torch.unsqueeze(torch.transpose(torch.tensor(  [[-0.385],[-0.411]], device=device).repeat(1,num_robots), 0, 1),0)
    wheel_RR = torch.unsqueeze(torch.transpose(torch.tensor(  [[0.385],[-0.411]],  device=device).repeat(1,num_robots), 0, 1),0)
    
    # Wheel locations, collected in a single variable
    wheel_locations = torch.cat((wheel_FL, wheel_FR, wheel_ML, wheel_MR, wheel_RL, wheel_RR), 0) 
    
    # The distance at which the rover should switch to turn on the spot mode.
    bound = 0.45 
    
    # Turning point
    P = torch.unsqueeze(lin_vel/ang_vel, 0)
    P = torch.copysign(P, -ang_vel)
    zeros = torch.zeros_like(P)
    P = torch.transpose(torch.cat((P,zeros), 0), 0, 1) # Add a zero component in the y-direction.
    P[:,0] = torch.squeeze(torch.where(torch.abs(P[:,0]) > bound, P[:,0], zeros)) # If turning point is between wheels, turn on the spot.
    lin_vel = torch.where(P[:,0] != 0, lin_vel, zeros) # If turning on the spot, set lin_vel = 0.

    # Calculate distance to turning point
    P = P.repeat((6,1,1))
    dist = torch.transpose((P - wheel_locations).pow(2).sum(2).sqrt(), 0, 1)

    # Motors on the left should turn opposite direction
    motor_side = torch.transpose(torch.tensor([[-1.0],[1.0],[-1.0],[1.0],[-1.0],[1.0]], device=device).repeat((1, num_robots)), 0, 1)
    
    # When not turning on the spot, wheel velocity is actually determined by the linear direction
    wheel_linear = torch.transpose(torch.copysign(ang_vel, lin_vel).repeat((6,1)), 0, 1)
    # When turning on the spot, wheel velocity is determined by motor side.
    wheel_turning = torch.transpose(ang_vel.repeat((6,1)), 0, 1) * motor_side
    ang_velocities = torch.where(torch.transpose(lin_vel.repeat((6,1)), 0, 1) != 0, wheel_linear, wheel_turning)
    
    # The velocity is determined by the disance from the wheel to the turning point, and the angular velocity the wheel should travel with
    motor_velocities = dist * ang_velocities

    # If the turning point is more than 1000 meters away, just go straight.
    motor_velocities = torch.where(dist > 1000, torch.transpose(lin_vel.repeat((6,1)), 0, 1), motor_velocities)

    # Convert linear velocity above ground to rad/s
    motor_velocities = (motor_velocities/wheel_diameter)
    
    steering_angles = torch.transpose(torch.where(torch.abs(wheel_locations[:,:,0]) > torch.abs(P[:,:,0]), torch.atan2(wheel_locations[:,:,1], wheel_locations[:,:,0] - P[:,:,0]), torch.atan2(wheel_locations[:,:,1], wheel_locations[:,:,0] - P[:,:,0])), 0, 1)
    steering_angles = torch.where(steering_angles < -3.14/2, steering_angles + math.pi, steering_angles)
    steering_angles = torch.where(steering_angles > 3.14/2, steering_angles - math.pi, steering_angles)

    return steering_angles, motor_velocities

if __name__ == "__main__":
    lin = torch.ones((1)) * 0.0
    ang = torch.ones((1)) * -2.0
    print(Ackermann(lin,ang, 'cpu'))