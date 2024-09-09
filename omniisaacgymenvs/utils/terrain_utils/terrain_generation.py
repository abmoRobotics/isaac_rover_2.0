from cmath import pi, sqrt
import numpy as np
import math
#from isaacgym.terrain_utils import *
from scipy.stats import qmc
import pymeshlab
import random
import torch

def cfa(cfa: float, rock_diameter: float):
    # https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/96JE03319?download=true
    # https://www.researchgate.net/publication/340235161_Geographic_information_system_based_detection_and_quantification_of_boulders_using_HiRISE_imagery_a_case_study_in_Jezero_Crater
    q = lambda k: 1.79 + (0.152 / k) # Rate at which the total area covered by rocks decreases with increasing diameter q(k)
    Fk = lambda D, k: k * math.exp(-q(k)*D) # The cumulative fractional area covered by rocks larger than D

    return Fk(rock_diameter,cfa)

def add_rocks_terrain(terrain, rock_height = (0.1,0.2)):

    k = 0.03    # total fractional area covered by rocks
    #sample_size = int(0.5 / terrain.horizontal_scale)
    #probs = np.arange(terrain.horizontal_scale, sample_size, terrain.horizontal_scale)
    rock = torch.empty(1,4,device="cuda:0") # [x, y, z, radius]
    rocks = torch.empty(0,4,device="cuda:0")
    res = terrain.horizontal_scale
    rock_distribution_step = max(0.10,res)
    scale = int(rock_distribution_step / res)
    num_rock_sizes = int(0.5/rock_distribution_step)
    sampler_halton = qmc.Halton(d=2, scramble=False)
    for i in range(1,num_rock_sizes):
        terrain_height_scaler = 1
        rock_radius = (i * rock_distribution_step) / 2        
        lower_bound = terrain.length * terrain.width * cfa(k,rock_diameter=i*rock_distribution_step)
        upper_bound = terrain.length * terrain.width * cfa(k,rock_diameter=(i+1)*rock_distribution_step)
        num_rocks = int((lower_bound-upper_bound)/(rock_radius*rock_radius*3.1415))

        random_positions = (sampler_halton.random(n=num_rocks))

        random_positions[:,0] *= terrain.num_rows
        random_positions[:,1] *= terrain.num_cols
        random_positions = random_positions.astype(int)
        # print()
        kernel = scaleMax(gaussian_kernel(i*scale+1,sigma=1,normalized=False))

        if kernel[int((i*scale+1)/2)-1,int((i*scale+1)/2)] < 1:
            kernel[int((i*scale+1)/2),int((i*scale+1)/2)] = kernel[int((i*scale+1)/2)-1,int((i*scale+1)/2)]
            #terrain_height_scaler = 0.5

        kernel = np.multiply(np.ones((1+i*scale,1+i*scale)) * random.uniform(rock_height[0], rock_height[1]), scaleMax(kernel) * (1/(1+np.exp(-i*scale*0.3)))*2) * terrain_height_scaler#+ np.ones((1+i,1+i)) * i* scale *0.2
        for p in random_positions:
            try:
                terrain.height_field_raw[p[0]: p[0]+1+i*scale, p[1]: p[1]+1+i*scale] += (kernel*1/terrain.vertical_scale) * (random.random()*0.4 + 0.6) # 0.6*0.4 is random height scaling
                try:
                    rock[0][0] = p[0] * res + rock_radius # x
                    rock[0][1] = p[1] * res + rock_radius # y
                    rock[0][2] = 0                  # z
                    rock[0][3] = rock_radius        # radius
                    rocks = torch.cat((rocks, rock))  
                except Exception:
                    print("ERROR POSITION OF ROCKS NOT SAVED")
                    raise Exception
            except Exception:
                pass

    return terrain, rocks

def NormalizeData(data) -> np.ndarray:
    #Normalizes data
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def scaleMax(data) -> np.ndarray:
    # Scales the maximum value to one
    return (data / np.max(data))

def scaleMin(data) -> np.ndarray:
    # Scales the minimum value to zero
    return (data - np.min(data))

def gaussian_distribution(n_samples: int, sigma=0.3, normalized=True) -> np.ndarray:

    # Discrete sampling of range [-1:1] with n samples
    # With step size 1 / ( (n_samples - 1) / 2 )
    
    step_size = 2 / (n_samples - 1)
    # Add epsilon to avoid rounding errors influenzing the size of the kernal
    epsilon = 1e-7
    sampled_values = np.arange(-1, 1+epsilon, step_size)
    # Calculate gaussian distribution
    gaussian_distribution = [(1/(sigma*math.sqrt(2*math.pi))) * math.exp(-0.5*(x/sigma)*(x/sigma))   for x in sampled_values]
    
    # Normalize data between 0 and 1
    if normalized == True:
        gaussian_distribution = NormalizeData(gaussian_distribution)

    return gaussian_distribution

def gaussian_kernel(n_samples: int, sigma=0.3, normalized=True) -> np.ndarray:

    # Take the outer product of a gaussian distribution
    gaussian_kernel = np.outer(gaussian_distribution(n_samples,sigma,normalized=normalized),gaussian_distribution(n_samples,sigma,normalized=normalized))

    return gaussian_kernel

def gaussian_terrain(terrain,kernel_radius=15,max_height=5):#terrain,gaussian_radius: float,height,n):
    """
    Parameters:
        gaussian_radius (float): Radius of gaussian kernel in meter
        height (float): Maximimum (and minimum) height of terrain
        n (int): Number of kernels per 100m^2
    """
    random.seed(10)
    kernel_radius = kernel_radius # radius in meters [m]
    max_height = max_height # Max height in meters [m]

    # Size of kernel size kernel_diameter*kernel_diameter is equal to 2 * kernel_radius [m] / resolution + 1
    kernel_diameter = ((2 * kernel_radius) / terrain.horizontal_scale ) + 1 
    kernel_radius_unitless = ((kernel_diameter - 1) / 2)

    # Get normalized gaussian kernel 
    kernel = gaussian_kernel(kernel_diameter,sigma=0.4)


    n_kernels = int((terrain.length/ (kernel_radius * 2)) * (terrain.width/ (kernel_radius * 2))) + 8

    # Generate random placement of kernels
    sampler_halton = qmc.Halton(d=2, scramble=False)
    HaltonSample = sampler_halton.random(n=n_kernels)
    HaltonSample[:,0] *= terrain.num_rows
    HaltonSample[:,1] *= terrain.num_cols
    HaltonSample = HaltonSample.astype(int)
    gaussian_heightfield = np.zeros((terrain.num_rows, terrain.num_cols), dtype=np.int16)
    len(HaltonSample)
   # HaltonSample = [[90,90]]
    for i in range(len(HaltonSample)):
        from_x = int( max(0, HaltonSample[i,0]-kernel_radius_unitless) )
        to_x = int( min(terrain.num_rows, HaltonSample[i,0]+kernel_radius_unitless) )
        from_y = int( max(0, HaltonSample[i,1]-kernel_radius_unitless) )
        to_y = int( min(terrain.num_cols, HaltonSample[i,1]+kernel_radius_unitless) )

        from_x_kernel = int( abs(min(0, HaltonSample[i,0]-kernel_radius_unitless)) )
        to_x_kernel = int(  abs(min(kernel_diameter-1,terrain.num_rows-(HaltonSample[i,0]-kernel_radius_unitless))) )
        from_y_kernel = int( abs(min(0, HaltonSample[i,1]-kernel_radius_unitless)) )
        to_y_kernel = int(  abs(min(kernel_diameter-1,terrain.num_cols-(HaltonSample[i,1]-kernel_radius_unitless))) )

        # Fixed height
        #terrain.height_field_raw[from_y: to_y, from_x: to_x] += (kernel[from_y_kernel:to_y_kernel,from_x_kernel:to_x_kernel] * max_height* 1/terrain.vertical_scale )

        # Random height
        terrain.height_field_raw[from_y: to_y, from_x: to_x] += (kernel[from_y_kernel:to_y_kernel,from_x_kernel:to_x_kernel] * random.uniform(-max_height,max_height)* 1/terrain.vertical_scale )
        #terrain.height_field_raw[from_y: to_y, from_x: to_x] += (kernel[from_y_kernel:to_y_kernel,from_x_kernel:to_x_kernel]* 0/terrain.vertical_scale )
       # print(HaltonSample[i])

    return terrain

def convert_heightfield_to_trimesh1(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:
        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)
    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]
    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles

def polygon_reduction(vertices, triangles, target_vertices=50000):
        #Target number of vertex
        TARGET=target_vertices
        # Create mesh
        m = pymeshlab.Mesh(vertices, triangles)
        # Create a meshset
        ms = pymeshlab.MeshSet()
        # Add mesh to meshset
        ms.add_mesh(m, "terrain")

        #Estimate number of faces to have 100+10000 vertex using Euler
        numFaces = 100 + 2*TARGET
        #Simplify the mesh - only first simplification is aggresive
        while (ms.current_mesh().vertex_number() > TARGET):
            ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces, preservenormal=True)
            print("Decimated to", numFaces, "faces mesh has", ms.current_mesh().vertex_number(), "vertex")
            #Refine estimation to converge to TARGET vertex number
            numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)

        # Save final mesh
        m = ms.current_mesh()
        
        # Get vertices as float32 (Supported by isaac gym)
        vertices = m.vertex_matrix().astype('float32')
        # Get faces as unit32 (Supported by isaac gym)
        faces =  m.face_matrix().astype('uint32')
        return vertices, faces

class SubTerrain1:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.num_rows = int(self.width/self.horizontal_scale)
        self.num_cols = int(self.length/horizontal_scale)
        self.height_field_raw = np.zeros((self.num_rows, self.num_cols), dtype=np.float64)