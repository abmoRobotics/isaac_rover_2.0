import pymeshlab
import open3d as o3d 
import torch
import numpy as np
ms = pymeshlab.MeshSet()


def load_terrain(file_name):
    default_path = 'tasks/utils/'
    ms.load_new_mesh(default_path + file_name)
    m = ms.current_mesh()
    # Get vertices as float32 (Supported by isaac gym)
    vertices = m.vertex_matrix().astype('float32')
    # Get faces as unit32 (Supported by isaac gym)
    faces =  m.face_matrix().astype('uint32')
    return vertices, faces



def calculate_knn_grid(file_name):
    default_path = 'tasks/utils/'
    # Resolution of the knn map
    res_x = 1200 # 
    res_y = 1200 #
    res = 0.05 # 5 cm resolution

    # Load mesh
    file_name = "terrainTest.ply"
    mesh = o3d.io.read_triangle_mesh(default_path + file_name) # Read the point cloud 

    # Get vertices and triangles from mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Calculate center of each triangle 
    x = (vertices[triangles[:,0]][:,0] + vertices[triangles[:,1]][:,0] + vertices[triangles[:,2]][:,0]) / 3
    y = (vertices[triangles[:,0]][:,1] + vertices[triangles[:,1]][:,1] + vertices[triangles[:,2]][:,1]) / 3
    z = (vertices[triangles[:,0]][:,2] + vertices[triangles[:,1]][:,2] + vertices[triangles[:,2]][:,2]) / 3
    center = np.array([x,y])
    center = torch.tensor(center,device='cuda:0',dtype=torch.float16)

    # Create matrix which contains value corresponding to the spatial x, y position
    xx = torch.arange(0,res_x*res,res,device='cuda:0',dtype=torch.float16)
    yy = torch.arange(0,res_y*res,res,device='cuda:0',dtype=torch.float16)
    map = torch.ones((2,res_x,res_y),device='cuda:0',dtype=torch.float16)
    map[0] = map[0]*xx
    map[1] = torch.transpose(map[1]*yy, 0, 1)
    map = map.swapaxes(0,1)
    map = map.swapaxes(1,2)
    
    print(map[0:10,0].unsqueeze(-1).shape)
    exit()
    # Create empty array for storing triangles 
    x = torch.tensor([[1, 1],[0,0]],device='cuda:0',dtype=torch.float16)
    # Create empty array for storing knn
    knn_indices = torch.ones(100,1200,1200,device='cuda:0',dtype=torch.int32)

    print(center.shape)
    #x = x.repeat(center.shape[1],1).swapaxes(0,1)
    x = x.unsqueeze(-1)
    center = center.repeat(100,1,1)

    # For loop for finding the 100 nearest triangles(relative to x,y) to each spatial x,y position in the map matrix.
    for x in range(12):
        print(x)
        for y in range(1200):
            # Get 100 x,y positions from the map
            #data = torch.ones(100,2,1,device='cuda:0')
            data = map[x*100:x*100+100,y].unsqueeze(-1)

            # Calculate the difference between the center of the triangles and the data.
            dist = torch.norm(center - data,dim=1)
            #print(dist)
            # Get the smallest 100 distances
            knn = dist.topk(100, largest=False)
            #print(knn)
            # Save the indices
            knn_indices[:,x*100:x*100+100,y] = knn.indices

    vertices = torch.tensor(vertices,device='cuda:0',dtype=torch.float16)
    triangles = torch.tensor(triangles,device='cuda:0',dtype=torch.int32)

    torch.save(knn_indices, 'map_indices.pt')
    torch.save(vertices, 'vertices.pt')
    torch.save(triangles, 'triangles.pt')

    # Store with values in triangles
    knn_values =  vertices[triangles[knn_indices.long()].long()].to(torch.float16)
    torch.save(knn_values, 'map_values.pt')
