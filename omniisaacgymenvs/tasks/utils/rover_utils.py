import torch
import open3d as o3d
import numpy as np

def ray_distance(sources: torch.Tensor, directions: torch.Tensor, triangles: torch.Tensor):
    # This is for multiple rays intersection with ONE triangle each
    # Sources: [n_rays, 3]
    # Direction: [n_rays, 3]
    # Direction: [n_rays, 3, 3]

    # Sources is 
    # Normalize and opposite of direction
    d = -(directions.swapaxes(0,1)[:] * 1/torch.sum(sources,dim=1)).swapaxes(0,1)

    # a = torch.ones([k_len,3], device='cuda:0') 
    # b = torch.ones([k_len,3], device='cuda:0')*2 - a
    # c = torch.ones([k_len,3], device='cuda:0')*3 - a
    a = triangles[:,0]
    b = triangles[:,1]
    c = triangles[:,2]


    g = sources - a 

    torch.cuda.synchronize()
    a=a.swapaxes(0,1)
    b=b.swapaxes(0,1)
    c=c.swapaxes(0,1)
    d=d.swapaxes(0,1)
    g=g.swapaxes(0,1)
    for i in range(10):
        det = (a[0][:] * b[1][:]* c[2][:] + b[0][:]*c[1][:]*a[2][:]+c[0][:]*a[1][:]*b[2][:]) - (a[2][:] * b[1][:]* c[0][:] + b[2][:]*c[1][:]*a[0][:]+c[2][:]*a[1][:]*b[0][:])
        #det = torch.linalg.det(detdet)
        
        #del a
        n = torch.where(det == 0,0,(g[0][:] * c[1][:]* d[2][:] + c[0][:]*d[1][:]*g[2][:]+d[0][:]*g[1][:]*c[2][:]) - (g[2][:] * c[1][:]* d[0][:] + c[2][:]*d[1][:]*g[0][:]+d[2][:]*g[1][:]*c[0][:]) / det)
        m = torch.where(det == 0,0,(b[0][:] * g[1][:]* d[2][:] + g[0][:]*d[1][:]*b[2][:]+d[0][:]*b[1][:]*g[2][:]) - (b[2][:] * g[1][:]* d[0][:] + g[2][:]*d[1][:]*b[0][:]+d[2][:]*b[1][:]*g[0][:]) / det) 
        #del d
        k = torch.where(det == 0,0,(b[0][:] * c[1][:]* g[2][:] + c[0][:]*g[1][:]*b[2][:]+g[0][:]*b[1][:]*c[2][:]) - (b[2][:] * c[1][:]* g[0][:] + c[2][:]*g[1][:]*b[0][:]+g[2][:]*b[1][:]*c[0][:]) / det)  
        
        #TODO implement append functionality
        k_after_check = torch.where(((n >= 0.0) & (m >= 0.0) & (n + m <= 1.0) & (k >= 0.0)),1,0) 
    
    return k_after_check

def get_knn_triangles():
    device = 'cuda:0'
    # Resolution of the knn map
    res_x = 1200 # 
    res_y = 1200 #
    res = 0.05 # 5 cm resolution

    # Load mesh
    input_file = "tasks/utils/terrainTest.ply"
    mesh = o3d.io.read_triangle_mesh(input_file) # Read the point cloud 

    # Get vertices and triangles from mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    #print(vertices.shape)
    #print(triangles.shape)
    #print(vertices[triangles[:,0]][:,0].shape)
    # Calculate center of each triangle 
    x = (vertices[triangles[:,0]][:,0] + vertices[triangles[:,1]][:,0] + vertices[triangles[:,2]][:,0]) / 3
    y = (vertices[triangles[:,0]][:,1] + vertices[triangles[:,1]][:,1] + vertices[triangles[:,2]][:,1]) / 3
    z = (vertices[triangles[:,0]][:,2] + vertices[triangles[:,1]][:,2] + vertices[triangles[:,2]][:,2]) / 3
    center = np.array([x,y])
    center = torch.tensor(center,device=device,dtype=torch.float16)
    print(center.shape)
  
    # Create matrix which contains value corresponding to the spatial x, y position
    xx = torch.arange(0,res_x*res,res,device=device,dtype=torch.float16)
    yy = torch.arange(0,res_y*res,res,device=device,dtype=torch.float16)
    map = torch.ones((2,res_x,res_y),device=device,dtype=torch.float16)
    map[0] = map[0]*xx
    map[1] = torch.transpose(map[1]*yy, 0, 1)
    map = map.swapaxes(0,1)
    map = map.swapaxes(1,2)
    print(map[0,0:10,:])
    
    index =torch.tensor([1,0],device=device,dtype=torch.int64)
    map = torch.index_select(map,2,index) # 1200 x 1200 x 2
    #print(m[0:10,0])
    #exit()
    #print(map[0:100,0].unsqueeze(-1).shape)
    # Create empty array for storing triangles 
    #x = torch.tensor([[1, 1],[0,0]],device=device,dtype=torch.float16)
    # Create empty array for storing knn
    knn_indices = torch.ones(100,1200,1200,device=device,dtype=torch.int32)

    #print(center.shape)
    #x = x.repeat(center.shape[1],1).swapaxes(0,1)
    #x = x.unsqueeze(-1)
    center = center.repeat(100,1,1) # 100 x 2 x 398677
    print(center.shape)
    # For loop for finding the 100 nearest triangles(relative to x,y) to each spatial x,y position in the map matrix.
    for x in range(12):
        print(x)
        for y in range(1200):
            # Get 100 x,y positions from the map
            data = map[x*100:x*100+100,y].unsqueeze(-1) # 100 x 2 x 1
            #print(center[:,:,10])
            #print(data[:,:,0])
            #print(torch.norm(center[:,:,10] - data[:,:,0],dim=1))
            temp = torch.norm(center[:,:,10] - data[:,:,0],dim=1)
            # Calculate the difference between the center of the triangles and the data.
            dist = torch.norm(center - data,dim=1)
            #print(dist[:,10] - temp)
            #print(dist.shape)
            #exit()
            #print(dist[50,0])
            
            #print(dist)
            # Get the smallest 100 distances
            knn = dist.topk(100, largest=False)
            # print(knn.values[0,:])
            # print(knn.indices[0,:])
            
            #print(knn.values[0])
            
            #print(knn)
            # Save the indices
            knn_indices[:,x*100:x*100+100,y] = knn.indices.swapaxes(0,1) # 100 x 1200 x 1200
            # print(knn_indices[:,0,0])
   
            vertices = torch.tensor(vertices,device=device,dtype=torch.float16)
            triangles = torch.tensor(triangles,device=device,dtype=torch.int32)
            #print(vertices[triangles[knn_indices[:,x*100,y].long()].long()])
            #exit()
 

    vertices = torch.tensor(vertices,device=device,dtype=torch.float16)
    triangles = torch.tensor(triangles,device=device,dtype=torch.int32)

    torch.save(knn_indices, 'map_indices.pt')
    torch.save(vertices, 'vertices.pt')
    torch.save(triangles, 'triangles.pt')

    # Store with values in triangles
    knn_values =  vertices[triangles[knn_indices.long()].long()].to(torch.float16)
    torch.save(knn_values, 'map_values.pt')

def height_lookup(triangle_matrix: torch.Tensor, depth_points: torch.Tensor, horizontal_scale, shift):
    # Heightmap 1200x1200x100
    # depth_points: Points in 3D [n_envs,n_points, 3]
    # horizontal_scale = 0.05
    # shift: [n, 2]
    
    # Scale locations to fit heightmap
    scaledmap = (depth_points-shift)/horizontal_scale
    # Bound values inside the map
    scaledmap = torch.clamp(scaledmap, min = 0, max = triangle_matrix.size()[0]-1)
    # Round to nearest integer
    scaledmap = torch.round(scaledmap)

    # Convert x,y coordinates to two vectors.
    x = scaledmap[:,:,0]
    y = scaledmap[:,:,1]
    x = x.reshape([(depth_points.size()[0]* depth_points.size()[1]), 1])
    y = y.reshape([(depth_points.size()[0]* depth_points.size()[1]), 1])
    x = x.type(torch.long)
    y = y.type(torch.long)
    
    
    # Get nearets array of triangles for searching
    triangles = triangle_matrix[x, y]
    triangles = triangles.reshape([depth_points.shape[0],depth_points.shape[1],triangle_matrix.shape[2],triangle_matrix.shape[3],triangle_matrix.shape[4]])

    # Return the found heights
    return triangles

def rover_spawn_height(heightmap: torch.Tensor, depth_points: torch.Tensor, horizontal_scale, vertical_scale, shift):
    # Scale locations to fit heightmap
    print(depth_points)
    print(shift)
    print(horizontal_scale)
    scaledmap = (depth_points-shift)/horizontal_scale
    # Bound values inside the map
    scaledmap = torch.clamp(scaledmap, min = 0, max = heightmap.size()[0]-1)
    # Round to nearest integer
    scaledmap = torch.round(scaledmap)

    # Convert x,y coordinates to two vectors.
    x = scaledmap[:,0]
    y = scaledmap[:,1]
    x = x.type(torch.long)
    y = y.type(torch.long)

    # Lookup heights in heightmap
    heights = heightmap[x, y]
    # print(x)
    # print("y")
    # print(y)
    
    # Scale to fit actual height, dependent on resolution
    heights = heights * vertical_scale

    return heights

def test_height_lookup():
    shift = torch.tensor([0.0,0.0,0.0], device='cuda:0')
    map_values = torch.load("map_values.pt")
    map_values=map_values.swapaxes(0,1)
    map_values=map_values.swapaxes(1,2)
    depth_points = torch.tensor([[[1,1,0],[1,1,1]],[[1,2,0],[2,3,0]]],device='cuda:0')
    horizontal = 0.05
    print(map_values.shape)
    print(height_lookup(map_values,depth_points,horizontal,shift).shape)
    print(depth_points.shape)

#get_knn_triangles()
# heightmap = torch.load("heightmap_tensor.pt")
# # for i in range(1200):
# #     print(heightmap[i][180])
# #     print(heightmap.shape)
# #print(heightmap[1200][0])
# print(heightmap[1200][240])