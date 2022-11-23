#!/usr/bin/env python
from binascii import crc32
import numpy as np
#from .locomotion_modes import LocomotionMode
import math

import torch
import time




@torch.jit.script
def Ackermann(lin_vel, ang_vel):
    # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
    
    wheel_x = 12.0
    wheel_y = 20.0
    # Distance from center og the rover to the top (centimeters):
    y_top = 19.5 # check if it's correct
    y_top_tensor = torch.tensor(y_top,device='cuda:0').repeat(lin_vel.size(dim=0))
    # Distance from center of the rover to the side (centimeters):
    x_side = 15.0 # check if it's correct

    # Calculate radius for each robot
    radius = torch.where(ang_vel != 0, torch.div(torch.abs(lin_vel),torch.abs(ang_vel))*100,torch.zeros(lin_vel.size(dim=0),device='cuda:0'))

    # Initiate zero tensors
    motor_velocities = torch.zeros(lin_vel.size(dim=0),6,device='cuda:0')
    steering_angles = torch.zeros(lin_vel.size(dim=0),6,device='cuda:0')

    #         """
    #         Steering angles conditions 
    #         """
    steering_condition1 = ((radius <= x_side) & (ang_vel != 0))
    steering_condition2 = ((torch.logical_not(radius <= x_side)) & (((ang_vel > 0) & ((torch.sign(lin_vel) > 0))) | ((ang_vel < 0) & ((torch.sign(lin_vel)) < 0))))
    steering_condition3 = ((torch.logical_not(radius <= x_side)) & (((ang_vel < 0) & ((torch.sign(lin_vel) > 0))) | ((ang_vel > 0) & ((torch.sign(lin_vel)) < 0))))
    #         """
    #         Steering angles calculation 
    #         """  
    #  
    # If the turning point is within the chassis of the robot, turn on the spot:
    turn_on_the_spot = torch.tensor(torch.atan2(y_top,x_side),device='cuda:0').repeat(lin_vel.size(dim=0))
    steering_angles[:,0] = torch.where(steering_condition1, turn_on_the_spot, steering_angles[:,0])
    steering_angles[:,1] = torch.where(steering_condition1, -turn_on_the_spot, steering_angles[:,1])
    steering_angles[:,4] = torch.where(steering_condition1, -turn_on_the_spot, steering_angles[:,4])
    steering_angles[:,5] = torch.where(steering_condition1, turn_on_the_spot, steering_angles[:,5])
    
    # Steering angles if turning anticlockwise moving forward or clockwise moving backwards
    steering_angles[:,0] = torch.where(steering_condition2, -torch.atan2(y_top_tensor,(radius-x_side)), steering_angles[:,0])
    steering_angles[:,1] = torch.where(steering_condition2, -torch.atan2(y_top_tensor,(radius+x_side)), steering_angles[:,1])
    steering_angles[:,4] = torch.where(steering_condition2, torch.atan2(y_top_tensor,(radius-x_side)), steering_angles[:,4])
    steering_angles[:,5] = torch.where(steering_condition2, torch.atan2(y_top_tensor,(radius+x_side)), steering_angles[:,5])

    # Steering angles if turning clockwise moving forward or anticlockwise moving backwards
    steering_angles[:,0] = torch.where(steering_condition3,torch.atan2(y_top_tensor,(radius+x_side)), steering_angles[:,0])
    steering_angles[:,1] = torch.where(steering_condition3,torch.atan2(y_top_tensor,(radius-x_side)), steering_angles[:,1])
    steering_angles[:,4] = torch.where(steering_condition3,-torch.atan2(y_top_tensor,(radius+x_side)), steering_angles[:,4])
    steering_angles[:,5] = torch.where(steering_condition3,-torch.atan2(y_top_tensor,(radius-x_side)), steering_angles[:,5])



    #    
    #  Motor speeds conditions
    #         
    velocity_condition1 = (radius <= x_side) & (ang_vel > 0)
    velocity_condition2 = (radius <= x_side) & (ang_vel < 0) #  elif radius[idx] <= x_side and ang_vel[idx] < 0: 
    velocity_condition3 = torch.logical_not((radius <= x_side)) & (ang_vel > 0)# ang_vel[idx] > 0:
    velocity_condition4 = torch.logical_not((radius <= x_side)) & (ang_vel < 0)# ang_vel[idx] < 0:
    #         """
    #         Motor speeds calculation 
    #         """   
    # Speed turning in place (counter clockwise), velocity of corner wheels = angular velocity 
    frontLeft = torch.sqrt((y_top*y_top)+(x_side*x_side))*abs(ang_vel)
    centerLeft = x_side*abs(ang_vel)
    relation = centerLeft/frontLeft # relation between corner wheel and center wheel velocity (center wheels slower)
    motor_velocities[:,0] = torch.where(velocity_condition1, -torch.abs(ang_vel), motor_velocities[:,0])
    motor_velocities[:,1] = torch.where(velocity_condition1, torch.abs(ang_vel), motor_velocities[:,1])
    motor_velocities[:,2] = torch.where(velocity_condition1, -torch.abs(ang_vel)*relation, motor_velocities[:,2])
    motor_velocities[:,3] = torch.where(velocity_condition1, torch.abs(ang_vel)*relation, motor_velocities[:,3])
    motor_velocities[:,4] = torch.where(velocity_condition1, -torch.abs(ang_vel), motor_velocities[:,4])
    motor_velocities[:,5] = torch.where(velocity_condition1, torch.abs(ang_vel), motor_velocities[:,5])

    # Speed turning in place (clockwise), velocity of corner wheels = angular velocity 
    frontLeft = torch.sqrt((y_top*y_top)+(x_side*x_side))*abs(ang_vel)
    centerLeft = x_side*abs(ang_vel)
    relation = centerLeft/frontLeft # relation between corner wheel and center wheel velocity (center wheels slower)

    motor_velocities[:,0] = torch.where(velocity_condition2, torch.abs(ang_vel),   motor_velocities[:,0])
    motor_velocities[:,1] = torch.where(velocity_condition2, -torch.abs(ang_vel), motor_velocities[:,1])
    motor_velocities[:,2] = torch.where(velocity_condition2, torch.abs(ang_vel)*relation, motor_velocities[:,2])
    motor_velocities[:,3] = torch.where(velocity_condition2, -torch.abs(ang_vel)*relation, motor_velocities[:,3])
    motor_velocities[:,4] = torch.where(velocity_condition2, torch.abs(ang_vel), motor_velocities[:,4])
    motor_velocities[:,5] = torch.where(velocity_condition2, -torch.abs(ang_vel), motor_velocities[:,5])
    


    # Speed turning anticlockwise moving forward/backward, velocity of frontRight wheel = linear velocity 
    frontLeft = (torch.sqrt((y_top_tensor*y_top_tensor)+((radius-x_side)*(radius-x_side)))*torch.abs(ang_vel))*torch.sign(lin_vel)
    frontRight = (torch.sqrt((y_top_tensor*y_top_tensor)+((radius+x_side)*(radius+x_side)))*torch.abs(ang_vel))*torch.sign(lin_vel)
    frontRelation = frontLeft/frontRight # relation of speed between the front wheels (frontLeft is slower)
    centerLeft = ((radius-x_side)*torch.abs(ang_vel))*torch.sign(lin_vel)
    centerRight = ((radius+x_side)*torch.abs(ang_vel))*torch.sign(lin_vel)
    centerRelation = centerLeft/centerRight # relation of speed between the center wheels (centerLeft is slower)
    frontCenterRelation = centerRight/frontRight # relation between center and front wheels (center is slower)
    
    motor_velocities[:,0] = torch.where(velocity_condition3, lin_vel*frontRelation, motor_velocities[:,0])
    motor_velocities[:,1] = torch.where(velocity_condition3, lin_vel, motor_velocities[:,1])
    motor_velocities[:,2] = torch.where(velocity_condition3, lin_vel*frontCenterRelation*centerRelation, motor_velocities[:,2])
    motor_velocities[:,3] = torch.where(velocity_condition3, lin_vel*frontCenterRelation, motor_velocities[:,3])
    motor_velocities[:,4] = torch.where(velocity_condition3, lin_vel*frontRelation, motor_velocities[:,4])
    motor_velocities[:,5] = torch.where(velocity_condition3, lin_vel, motor_velocities[:,5])

    # Speed turning clockwise moving forward/backward, velocity of frontLeft wheel = linear velocity
    frontLeft = (torch.sqrt((y_top_tensor*y_top_tensor)+((radius+x_side)*(radius+x_side)))*torch.abs(ang_vel))*torch.sign(lin_vel)
    frontRight = (torch.sqrt((y_top_tensor*y_top_tensor)+((radius-x_side)*(radius-x_side)))*torch.abs(ang_vel))*torch.sign(lin_vel)
    frontRelation = frontRight/frontLeft # relation of speed between the front wheels (frontRight is slower)
    centerLeft = ((radius+x_side)*torch.abs(ang_vel))*torch.sign(lin_vel)
    centerRight = ((radius-x_side)*torch.abs(ang_vel))*torch.sign(lin_vel)
    centerRelation = centerRight/centerLeft # relation of speed between the center wheels (centerRight is slower)
    frontCenterRelation = centerLeft/frontLeft # relation between center and front wheels (center is slower)
    
    motor_velocities[:,0] = torch.where(velocity_condition4, lin_vel, motor_velocities[:,0])
    motor_velocities[:,1] = torch.where(velocity_condition4, lin_vel*frontRelation, motor_velocities[:,1])
    motor_velocities[:,2] = torch.where(velocity_condition4, lin_vel*frontCenterRelation, motor_velocities[:,2])
    motor_velocities[:,3] = torch.where(velocity_condition4, lin_vel*frontCenterRelation*centerRelation, motor_velocities[:,3])
    motor_velocities[:,4] = torch.where(velocity_condition4, lin_vel, motor_velocities[:,4])
    motor_velocities[:,5] = torch.where(velocity_condition4, lin_vel, motor_velocities[:,5])
    
    return steering_angles, motor_velocities

# This is the new kinematics function
# The inputs are m/s and rad/s. The dimensions are for the mars rover 2.0 project - units in meters.
# The code is made with basis in appendix D in the repport "Mapless Mobile Navigation on Mars using Reinforcement Learning"
@torch.jit.script
def Ackermann2(lin_vel, ang_vel, device='cuda:0'):
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
    motor_velocities = motor_velocities/wheel_diameter
    
    steering_angles = torch.transpose(torch.where(wheel_locations[:,:,0] > P[:,:,0], torch.atan2(wheel_locations[:,:,1], wheel_locations[:,:,0] - P[:,:,0]), torch.atan2(-wheel_locations[:,:,1], -wheel_locations[:,:,0] - P[:,:,0])), 0, 1)
    steering_angles = torch.where(steering_angles < -3.14/2, steering_angles + math.pi, steering_angles)
    steering_angles = torch.where(steering_angles > 3.14/2, steering_angles - math.pi, steering_angles)
    return steering_angles, motor_velocities

if __name__ == "__main__":
    lin = torch.ones((1)) * 1.0
    ang = torch.ones((1)) * -2.0
    print(Ackermann2(lin,ang, 'cpu'))