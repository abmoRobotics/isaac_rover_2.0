import torch

def ray_distance(sources: torch.Tensor, directions: torch.Tensor, triangles: torch.Tensor, device='cuda:0',dtype=torch.float16):
    """
    Checks if there is an intersection between a triangle and calculates the distance based on Möller–Trumbore intersection algorithm

    Parameters
    ----------
    sources : torch.tensor
        3D points of the ray source – dimension: [n_rays, 3]
    directions : torch.tensor
        Direction vector for the ray – dimension: [n_rays, 3]
    triangles : torch.tensor
        Triangles for ray intersection – dimension: [n_rays, 3, 3]
    """
    # This is for multiple rays intersection with ONE triangle each
    # Sources: [n_rays, 3]
    # Direction: [n_rays, 3]
    # Direction: [n_rays, 3, 3]

    # Sources is 
    # Normalize and opposite of direction

    # Utility variables
    epsilon = 0.1
    zeros = torch.zeros(sources.shape[0],device=device,dtype=dtype) - epsilon
    ones = torch.ones(sources.shape[0],device=device,dtype=dtype) + epsilon
    error_tensor = ones * -99.0

    # Normalize and opposite of direction
    d = -torch.nn.functional.normalize(directions) # 4000 x 3 
    
    # get the three vertices of each triangle and subtract the last vertice
    a = triangles[:,2]      # 4000 x 3 
    b = triangles[:,1] - a  # 4000 x 3 
    c = triangles[:,0] - a  # 4000 x 3 
    g = sources - a     # 4000 x 3 
    
    # Calculate the determinant of [b,c,d] using the Scalar Triple Product
    det = b.cross(c) 
    det = (det[:,0]*d[:,0]+det[:,1]*d[:,1]+det[:,2]*d[:,2])
    
    # Calculate Scalar Triple Product for [g,c,d] 
    n = g.cross(c)
    n = (n[:,0]*d[:,0]+n[:,1]*d[:,1]+n[:,2]*d[:,2])/det
    n = torch.where(det == zeros,error_tensor,n)

    # Calculate Scalar Triple Product for [b,g,d]
    m = b.cross(g)
    m = (m[:,0]*d[:,0]+m[:,1]*d[:,1]+m[:,2]*d[:,2])/det
    m = torch.where(det == ones,error_tensor,m) 

    # Calculate Scalar Triple Product for [b,c,g]
    k = b.cross(c)
    k = (k[:,0]*g[:,0]+k[:,1]*g[:,1]+k[:,2]*g[:,2])/det
    k = torch.where(det == ones,error_tensor,k) 

    # filter out based on the condition ((n >= 0.0) & (m >= 0.0) & (n + m <= 1.0))
    k_after_check = torch.where(((n >= zeros) & (m >= zeros) & (n + m <= ones)),k,error_tensor)


    # Calucate the intersection point
    pt = sources -(d.swapaxes(0,1)*k_after_check).swapaxes(0,1)
    
    #torch.cuda.empty_cache()
    return k_after_check, pt


