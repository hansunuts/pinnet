import numpy as np
from typing import Tuple
import torch
import torch.nn.functional as F
from pyntcloud import PyntCloud 
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def normalise_pc(
    pc:torch.Tensor, 
    padding: float = 0.1,
    R=None
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
    ''' Normalise point coordinates to [0+padding/2, 1-padding/2], 
        use the axis with largest (max - min) as the base

        Args:
            pc (n, 3)
            padding: gap between the farest point and the cube edge
            R: (3, 3)
        Returns:
            pc_norm (n, 3)
            transform (4, 4) Translation after scaling
    '''
    pc = pc.type(torch.float32)
    center = pc.mean(0)
    pc_offset = pc - center
    
    max = torch.abs(pc_offset).max()
    
    s = (0.5-padding) / max
    t = -s*center
    #t = [0, 0, 0]

    mat_scale = torch.Tensor(
        [[  s,  0,  0,  0],
         [  0,  s,  0,  0],
         [  0,  0,  s,  0],
         [  0,  0,  0,  1]]).to(pc.device)
    
    if R is not None:
        mat_rotation = torch.Tensor(
            [[R[0][0],  R[0][1],    R[0][2],    0],
            [R[1][0],  R[1][1],    R[1][2],    0],
            [R[2][0],  R[2][1],    R[2][2],    0],
            [0,        0,          0,          1]]).to(pc.device)
    
    mat_translation = torch.Tensor(
        [[  1,      0,      0,      t[0]],
         [  0,      1,      0,      t[1]],
         [  0,      0,      1,      t[2]],
         [  0,      0,      0,      1]]).to(pc.device)
    
    if R is None:
        transform = mat_translation @ mat_scale
    else:
        transform = mat_translation @ (mat_rotation @ mat_scale)
    
    pc_h = F.pad(pc, (0, 1, 0, 0), 'constant', value=1).type(torch.float32)
    pc_norm = (transform[None, :, :] @ pc_h[:, :, None]).squeeze(-1)[:, :3]
        
    return pc_norm, transform

def normalise_pc_np(pc:np.ndarray, padding: float = 0.0, R=None) -> np.ndarray:
    ''' Normalise point coordinates to [0+padding/2, 1-padding/2], 
        use the axis with largest (max - min) as the base

        Args:
            pc (n, 3)
            padding: gap between the farest point and the cube edge
        Returns:
            pc_norm (n, 3)
            transform (4, 4) Translation after scaling
    '''
    pc = torch.from_numpy(pc)
    pc_norm, transform = normalise_pc(pc, padding, R)

    pc = pc_norm.numpy()
    transform = transform.numpy()
    
    return pc, transform

def random_select_np(pcd_data:np.ndarray, n_pt:int):
    '''
    Get down sampled pcd. Randomly select points.
    Args:
        pcd_data: [n, 3]
        n_pt: point num after downsample
    Return:
        pcd_data_s [n_pt, 3]
    '''
    
    rand_idcs = np.random.choice(
                            pcd_data.shape[0], 
                            n_pt, 
                            replace=True)
    pcd_data_s = pcd_data[rand_idcs]

    if rand_idcs.shape[0] < n_pt: # in down_sample happens
        fix_idx = np.asarray(range(pcd_data_s.shape[0]))
        while pcd_data_s.shape[0] + fix_idx.shape[0] < n_pt:
            fix_idx = np.concatenate((fix_idx, np.asarray(range(pcd_data_s.shape[0]))), 
                                        axis=0)
        random_idx = np.random.choice(pcd_data_s.shape[0], 
                                        n_pt - fix_idx.shape[0], 
                                        replace=False)
        rand_idcs = np.concatenate((fix_idx, random_idx), axis=0)
    
        pcd_data_s = pcd_data_s[rand_idcs]
    
    return pcd_data_s

def random_select(
    pc:torch.Tensor,
    n_pt:int
    ) -> torch.Tensor:
    pc_np = pc.cpu().detach().numpy()
    pc_s = torch.from_numpy(random_select_np(pc_np, n_pt)).to(pc.device)
    return pc_s

def point_to_pc_dist(
    p:torch.Tensor,
    pc:torch.Tensor,
    ) -> float:
    '''point to point cloud distance.
    Args:
        p:  (3) point
        pc: (n, 3) point cloud
        
    Return:
        dist:   distance
    '''
    dists = torch.norm(pc - p.unsqueeze(0), dim=1)
    return dists.min()

def get_edge_points(
    pc:torch.Tensor,
    thre:float=0.1,
    k_n:int=50
    ) -> torch.Tensor:
    '''
    Args:
        pc: (n, 3)
        thre: larger value is stricter
        k_n: neighbour numbers to consider
    Returns:
        edge: (n_edge, 3)
    '''
    device = pc.device
    pc = pc.cpu().detach().numpy()
    pcd1 = PyntCloud(points=pd.DataFrame(data=pc, columns=["x", "y", "z"]))

    pcd_np = np.zeros((len(pcd1.points),6))

    # find neighbors
    kdtree_id = pcd1.add_structure("kdtree")
    k_neighbors = pcd1.get_neighbors(k=k_n, kdtree=kdtree_id) 

    # calculate eigenvalues
    ev = pcd1.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

    x = pcd1.points['x'].values 
    y = pcd1.points['y'].values 
    z = pcd1.points['z'].values 

    e1 = pcd1.points['e3('+str(k_n+1)+')'].values
    e2 = pcd1.points['e2('+str(k_n+1)+')'].values
    e3 = pcd1.points['e1('+str(k_n+1)+')'].values

    sum_eg = np.add(np.add(e1,e2),e3)
    sigma = np.divide(e1,sum_eg)
    sigma_value = sigma
    #pdb.set_trace()
    #img = ax.scatter(x, y, z, c=sigma, cmap='jet')

    # visualize the edges
    sigma = sigma>thre

    # Save the edges and point cloud
    thresh_min = sigma_value < thre
    sigma_value[thresh_min] = 0
    thresh_max = sigma_value > thre
    sigma_value[thresh_max] = 255

    pcd_np[:,0] = x
    pcd_np[:,1] = y
    pcd_np[:,2] = z
    pcd_np[:,3] = sigma_value

    edge_np = np.delete(pcd_np, np.where(pcd_np[:,3] == 0), axis=0) 
    edge = torch.from_numpy(edge_np[:, :3]).type(torch.float32).to(device)
    
    return edge

def get_high_curvature_points(point_cloud, radius, curvature_threshold):
    """
    Compute curvatures of points in a 3D point cloud.
    
    Args:
    - point_cloud (torch.Tensor): Input point cloud with shape (n, 3).
    - radius (float): Radius for selecting local neighborhood.
    - curvature_threshold (float): Threshold for classifying points based on curvature.
    
    Returns:
    - high_curvature_points (torch.Tensor): Points with high curvature or on edges.
    """
    device = point_cloud.device
    
    # Convert to numpy array for NearestNeighbors
    point_cloud_np = point_cloud.cpu().numpy()
    
    # Build nearest neighbors search tree
    nbrs = NearestNeighbors(n_neighbors=9, radius=radius, algorithm='auto').fit(point_cloud_np)
    
    high_curvature_indices = []
    
    for i in range(len(point_cloud)):
        query_point = point_cloud_np[i]
        
        # Query local neighborhood points
        indices = nbrs.radius_neighbors([query_point], return_distance=False)[0]
        if len(indices) < 9:
            continue
        
        neighborhood = point_cloud[indices]
        
        # Fit a quadratic surface to the local neighborhood
        centroid = neighborhood.mean(dim=0)
        centered_points = neighborhood - centroid
        cov_matrix = centered_points.T @ centered_points
        eigenvalues, _ = torch.symeig(cov_matrix, eigenvectors=True)
        
        # Compute curvature as the ratio of smallest to largest eigenvalue
        curvature = eigenvalues[0] / eigenvalues[2]
        
        if curvature > curvature_threshold:
            high_curvature_indices.append(i)
    
    high_curvature_points = point_cloud[high_curvature_indices].to(device)
    return high_curvature_points