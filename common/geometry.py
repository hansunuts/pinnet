import torch
import numpy as np

def get_skew_symmetric_mat(
    vec:torch.Tensor
    ) -> torch.Tensor:
    '''
    Get the skew symmetric matrix of the vector
    Args:
        vec (3)
    Returns:
        skew_symmetric (3, 3)
    '''
    skew_symmetric = torch.zeros(3, 3)
    skew_symmetric[0, 1] = -1 * vec[2]
    skew_symmetric[0, 2] = vec[1]
    skew_symmetric[1, 0] = vec[2]
    skew_symmetric[1, 2] = -1 * vec[0]
    skew_symmetric[2, 0] = -1 * vec[1]
    skew_symmetric[2, 1] = vec[0]
    
    skew_symmetric.to(vec.device)
    return skew_symmetric

def get_fundamental_mat(
    pose1:torch.Tensor,
    pose2:torch.Tensor,
    K:torch.Tensor,
    ) -> torch.Tensor:
    '''
    Get fundamental matrix between two poses.
    Args:
        pose1 (4, 4) cam to world
        pose2 (4, 4) cam to world
        K   (3, 3)
    Returns:
        F   (3, 3)
    '''
    K = K[0:3, 0:3] # just in case input is (4, 4)
    K_inv = torch.linalg.inv(K)
    K_inv_t = torch.transpose(K_inv, 0, 1)
    T_2_1 = torch.matmul(torch.linalg.inv(pose2), pose1) # (4, 4)
    R = T_2_1[0:3, 0:3]         # (3, 3)
    t = T_2_1[0:3, 3]           # (3)
    t_skew = get_skew_symmetric_mat(t)
    F = torch.matmul(K_inv_t, torch.matmul(t_skew,torch.matmul(R, K_inv)))
    return F

def intersection_of_multi_lines(strt_points:np.ndarray, 
                                directions:np.ndarray
                                ) -> np.ndarray:  
    ''' get nearest point to all lines.
        https://zhuanlan.zhihu.com/p/146190385
    Args:
        strt_points (n, 3) line start points
        directions (n, 3) list dierctions
    Returns:
        inter   (3) the coordinate of the nearest point to n lines
        s  (n)  p = o + s * dir, can be used to get points on each line
    '''
    
    n, dim = strt_points.shape

    G_left = np.tile(np.eye(dim), (n, 1))  
    G_right = np.zeros((dim*n, n))  

    for i in range(n):
        G_right[i*dim:(i+1)*dim, i] = -directions[i, :]

    G = np.concatenate([G_left, G_right], axis=1)  
    d = strt_points.reshape((-1, 1)) 

    m = np.linalg.inv(np.dot(G.T, G)).dot(G.T).dot(d)   # [0:3] the coordinate of the nearest point
                                                        # [3:] put in the line equations, get the point on lines.
    inter = np.squeeze(m[0:dim], axis=1)
    scales = m[dim:]
    return inter, scales

def point_line_dist_2d(p:torch.Tensor,      # (n, 2) x,y     
                        A:float,       
                        B:float, 
                        C:float
                        ) -> torch.Tensor:    # (n)
    ''' distance between a point and a line in 2d
    Args:
        p: (n, 2) points 
        A:  
        B:
        C: line parameters
    Return:
        dist: (n) distance
    '''
    dist = torch.abs(A * p[:, 0] + B * p[:, 1] + C) / torch.sqrt(A * A + B * B)
    return dist

def points_line_dist_3d(p:torch.Tensor,  # (n, 3)
                       o:torch.Tensor,  # (3)
                       dir:torch.Tensor # (3)
                       ) -> torch.Tensor: # (n)
    ''' distance between points and a line in 3d
    Args:
        p: (3) points 
        o:  
        dir: line parameters
    Return:
        dist: (n) distance
    '''
    a = dir[0]
    b = dir[1]
    c = dir[2]
    xo = o[0]
    yo = o[1]
    zo = o[2]  
    xp = p[:, 0].unsqueeze(0)
    yp = p[:, 1].unsqueeze(0)
    zp = p[:, 2].unsqueeze(0)
    
    t = (a * (xp - xo) + b * (yp - yo) + c * (zp -zo)) / (a * a + b * b + c * c) # (n)
    p_ = o.unsqueeze(0) + t * dir.unsqueeze(0) # (n, 3) p's cloest point on the line 
    dist = torch.linalg.norm(p_ - p)
    
    return dist

def k_after_scale(
    k:torch.Tensor, 
    scale:float
    ) -> torch.Tensor:
    '''
    Args:
        k: (4, 4) color cam K
        scale: 
    Returns:
        k_scaled: (4, 4) k after scale
    '''
    k_scaled = k * scale 
    k_scaled[2, 2] = 1
    k_scaled[3, 3] = 1
    return k_scaled

if __name__ == '__main__':
    
    '''
    pose1 = torch.Tensor(
        [[-0.913237, 0.243608, -0.326578, 2.618314],
         [0.407408, 0.537971, -0.737974, 3.106410],
         [-0.004086, -0.806996, -0.590543, 1.296575],
         [0.000000, 0.000000, 0.000000, 1.000000],
        ])
    
    pose2 = torch.Tensor(
        [[-0.878439, 0.294809, -0.376075, 2.594427],
         [0.476927, 0.491877, -0.728422, 3.212729],
         [-0.029763, -0.819234, -0.572686, 1.307344],
         [0.000000, 0.000000, 0.000000, 1.000000],
        ])
    
    K = torch.Tensor(
        [[1169.621094, 0.000000, 646.295044],
         [0.000000, 1167.105103, 489.927032],
         [0.000000, 0.000000, 1.000000 ],
        ])
    
    F = get_fundamental_mat(pose1, pose2, K)
    '''
    
    strt_point = np.zeros((2, 3))
    strt_point[0, :] = np.array([0, 0, -10])
    strt_point[1, :] = np.array([-10, 0, 0])

    directions = np.zeros((2, 3))
    directions[0, :] = np.array([1, 0, 0])
    directions[1, :] = np.array([0, 1, 0])  

    inters = intersection_of_multi_lines(strt_point, directions)
    

    