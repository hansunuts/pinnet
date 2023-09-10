import random
import torch
import numpy as np
from common.pointcloud import random_select
from common.utils import np_to_torch

def pc_density_change(
    pc:torch.Tensor,
    n_max:int,
    n_min:int,
    ) -> torch.Tensor:
    '''
    Args:
        pc: (n, 3)
        n_max: max points in the point cloud
        n_min: min points in the point cloud
    Returns:
        pc: (n', 3)
    '''
    n_pts = random.randint(n_min, n_max)
    pc = random_select(pc, n_pts)
    return pc

def pc_noise(
    pc:torch.Tensor,
    sigma:float=0.01,
    clip:float=0.05
    ):
    '''
    Args:
        pc: (n, 3)
        sigma: sigma value for the normal distribution.
        clip: max offset
    '''
    device = pc.device
    N, C = pc.shape
    assert(clip > 0)
    noise = torch.clip(sigma * torch.randn(N, C), -1*clip, clip).to(device)
    pc_noise = pc + noise
    return pc_noise

def add_gaussian_noise(point_cloud, sigma):
    """
    Add Gaussian noise to a point cloud.

    Args:
        point_cloud (torch.Tensor): Input point cloud with shape (n, 3).
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Point cloud with added Gaussian noise, same shape as input.
    """
    # Generate Gaussian noise with the same shape as the input point cloud
    noise = torch.randn(point_cloud.shape) * sigma

    # Add the noise to the original point cloud
    noisy_point_cloud = point_cloud + noise

    return noisy_point_cloud

def pc_remove_volumes(
    pc:torch.Tensor,
    volume_centers:torch.Tensor,
    volume_size:float,
    volume_type:str='cube'
    ) -> torch.Tensor:
    ''' Remove points in some volumes in the point cloud
    Args:
        pc: (n, 3)
        volume_centers: (m, 3)
        volume_size: 
        volume_type: 'cube', 'shpere', 'polyhedron'. only cube is available.
    Returns:
        pc: (n', 3)
    '''
    volumes_max = volume_centers + volume_size / 2 # (m, 3)
    volumes_min = volume_centers - volume_size / 2 # (m, 3)
    volumes_mask = ((((pc[:, None, :] < volumes_max[None, :, :]).sum(-1) == 3) * ((pc[:, None, :] > volumes_min[None, :, :]).sum(-1) == 3)).sum(-1) == 0)
    pc_removed = pc[volumes_mask]
    return pc_removed
    
def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def random_rotation(pcd:torch.Tensor):
    '''
    Apply a random rigid transform to a point cloud 

    Args:
        pcd [n, 3]

    Returns:
        pcd [n, 3]
        R [3, 3]
    '''
    pcd = pcd.cpu().detach().numpy()

    # rotation
    rotate_angle = 360.0 / 180.0 * (np.pi)
    angles_3d = np.random.rand(3) * rotate_angle
    R = angles2rotation_matrix(angles_3d)
    pcd = np.dot(pcd, R)

    pc = torch.from_numpy(pcd).type(torch.float)
    R = torch.from_numpy(R).type(torch.float)
    
    return pc, R
    