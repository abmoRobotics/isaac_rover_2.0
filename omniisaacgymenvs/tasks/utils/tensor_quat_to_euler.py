from ast import Pass
from cmath import sin
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

def tensor_quat_to_eul(quats):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # Quaternions format: W, X, Y, Z
    # Quat index:         0, 1, 2, 3
    # Euler angles:       ZYX

    euler_angles = torch.zeros([len(quats), 3], device='cuda:0')
    ones = torch.ones([len(quats)], device='cuda:0')
    zeros = torch.zeros([len(quats)], device='cuda:0')

    #Roll
    sinr_cosp = 2 * (quats[:,0] * quats[:,1] + quats[:,2] * quats[:,3])
    cosr_cosp = ones - (2 * (quats[:,1] * quats[:,1] + quats[:,2] * quats[:,2]))
    euler_angles[:,0] = torch.atan2(sinr_cosp, cosr_cosp)

    #Pitch
    sinp = 2 * (quats[:,0]*quats[:,2] - quats[:,3] * quats[:,1])
    condition = (torch.sign(sinp - ones) >= zeros)
    euler_angles[:,1] = torch.where(condition, torch.copysign((ones*torch.pi)/2, sinp), torch.asin(sinp)) 

    #Yaw    
    siny_cosp = 2 * (quats[:,0] * quats[:,3] + quats[:,1] * quats[:,2])
    cosy_cosp = ones - (2 * (quats[:,2] * quats[:,2] + quats[:,3] * quats[:,3]))
    euler_angles[:,2] = torch.atan2(siny_cosp, cosy_cosp)
    
    return euler_angles
