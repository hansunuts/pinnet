from typing import Tuple
import numpy as np
import torch

def nms(
    pts: torch.Tensor,
    sals: torch.Tensor,
    nms_radius:float,
    n_max_kp:int,
    ) -> Tuple[
        torch.Tensor,   # kps
        torch.Tensor,   # kp_sals
    ]:
    '''
    2D or 3D nms
    Args:
        pts: (n, 3) xyz or (n, 2) xy
        sals: (n)
        nms_radius:
        n_max_kp:
    
    Returns:
        kps     (n', 3) or (n', 2) n' < n_max_kp
        kp_sals (n')
    '''
    valid_kp_counter = 0
    valid_kps = torch.zeros_like(pts, dtype=torch.float).to(pts.device)
    valid_sals = torch.zeros_like(sals, dtype=torch.float).to(pts.device)

    while pts.shape[0] > 0 and valid_kp_counter < n_max_kp:
        min_idx = torch.argmax(sals, dim=0)
        valid_kps[valid_kp_counter, :] = pts[min_idx, :]
        valid_sals[valid_kp_counter] = sals[min_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = torch.linalg.norm(
            (valid_kps[valid_kp_counter:valid_kp_counter + 1, :] - pts), 
            dim=1, 
            keepdim=False)  # n
        mask = distance_array > nms_radius  # n

        pts = pts[mask, ...]
        sals = sals[mask]

        # increase counter
        valid_kp_counter += 1

    if n_max_kp >= valid_kp_counter:
        return (valid_kps[0:valid_kp_counter, :], 
                valid_sals[0:valid_kp_counter])
    else:
        return (valid_kps[0:n_max_kp, :], 
                valid_sals[0:n_max_kp])

def nms_a(
    pts:torch.Tensor,
    sals:torch.Tensor,
    attached:torch.Tensor,
    nms_radius:float,
    n_max_kp:int,
    ) -> torch.Tensor:
    '''
    2D or 3D nms
    Args:
        pts: (n, 3) xyz or (n, 2) xy
        sals: (n)
        attached: (n, dim)
        nms_radius:
        n_max_kp:
    
    Returns:
        kps     (n', 3) or (n', 2) n' < n_max_kp
        kp_sals (n')
        kp_attached(n', dim)
    '''
    valid_kp_counter = 0
    valid_kps = torch.zeros_like(pts, dtype=torch.float).to(pts.device)
    valid_sals = torch.zeros_like(sals, dtype=torch.float).to(pts.device)
    valid_attached = torch.zeros_like(attached, dtype=torch.float).to(pts.device)

    while pts.shape[0] > 0 and valid_kp_counter < n_max_kp:
        min_idx = torch.argmax(sals, dim=0)
        valid_kps[valid_kp_counter, :] = pts[min_idx, :]
        valid_sals[valid_kp_counter] = sals[min_idx]
        valid_attached[valid_kp_counter] = attached[min_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = torch.linalg.norm(
            (valid_kps[valid_kp_counter:valid_kp_counter + 1, :] - pts), 
            dim=1, 
            keepdim=False)  # n
        mask = distance_array > nms_radius  # n
        
        pts = pts[mask, ...]
        sals = sals[mask]
        attached = attached[mask, ...]

        # increase counter
        valid_kp_counter += 1

    if n_max_kp >= valid_kp_counter:
        return (valid_kps[0:valid_kp_counter, :], 
                valid_sals[0:valid_kp_counter],
                valid_attached[0:valid_kp_counter, :])
    else:
        return (valid_kps[0:n_max_kp, :], 
                valid_sals[0:n_max_kp],
                valid_attached[0:n_max_kp, :])

def nms_np(
    pts:np.ndarray, 
    sals:np.ndarray,
    nms_radius:float,
    n_max_kp:int,
    ) -> Tuple[
        np.ndarray,     # kps
        np.ndarray      # kp_sals
        ]:
    '''
    2D or 3D nms
    Args:
        pts: (n, 3) xyz or (n, 2) xy
        sals: (n)
        nms_radius:
        n_max_kp:
    
    Returns:
        kps     (n', 3) or (n', 2) n' < n_max_kp
        kp_sals (n')
    '''
    valid_kp_counter = 0
    valid_kps = np.zeros(pts.shape, dtype=pts.dtype)
    valid_sals = np.zeros(sals.shape, dtype=sals.dtype)

    while pts.shape[0] > 0 and valid_kp_counter < n_max_kp:
        min_idx = np.argmax(sals, axis=0)
        valid_kps[valid_kp_counter, :] = pts[min_idx, :]
        valid_sals[valid_kp_counter] = sals[min_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = np.linalg.norm(
            (valid_kps[valid_kp_counter:valid_kp_counter + 1, :] - pts), 
            axis=1, 
            keepdims=False)  # n
        mask = distance_array > nms_radius  # n

        pts = pts[mask, ...]
        sals = sals[mask]

        # increase counter
        valid_kp_counter += 1

    if n_max_kp >= valid_kp_counter:
        return (valid_kps[0:valid_kp_counter, :], 
                valid_sals[0:valid_kp_counter])
    else:
        return (valid_kps[0:n_max_kp, :], 
                valid_sals[0:n_max_kp])