import torch


def ray_distance(sources, directions, triangles):
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
        # k = torch.where(det2 == 0,0,torch.linalg.det(kk) / det2)  
        #time.sleep(20)

    #k_after_check = torch.where(((n >= 0.0) & (m >= 0.0) & (n + m <= 1.0) & (k >= 0.0)),1,0) 