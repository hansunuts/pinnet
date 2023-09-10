'''
https://github.com/vitalemonate/depth2Cloud/blob/main/depth2Cloud.py
'''

import os
import sys
import numpy as np
import torch
import cv2
from path import Path
from tqdm import tqdm

def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()

def pixel_to_point3d(
    point2d:torch.Tensor,
    Z:torch.Tensor,
    pose:torch.Tensor, 
    K:torch.Tensor,
    remove_invalid:bool=True,
    ) -> torch.Tensor:
    '''
    pixel to world points
    Args:
        point2d:    (n_points, 2)   int xy
        Z:          (n_points)      float in mm
        pose:       (4, 4) c2w
        K:          (4, 4)      
    Returns:
        points:     (n_points, 3)  or (n_points', 3) world coordinates
        valid_mask: (n_points) 
    '''
    X = (point2d[:, 0] - K[0, 2]) * Z / K[0, 0]
    Y = (point2d[:, 1] - K[1, 2]) * Z / K[1, 1]

    X = torch.ravel(X)
    Y = torch.ravel(Y)
    Z = torch.ravel(Z)

    valid_mask = Z > sys.float_info.epsilon

    if remove_invalid:
        X = X[valid_mask]  # (n_points')
        Y = Y[valid_mask]
        Z = Z[valid_mask]
    
        point_cam = torch.vstack((X, Y, Z, torch.ones(len(X)).to(point2d.device)))
        point_world = (torch.matmul(pose, point_cam)[0:3, :]).T # (n_points', 3)
        
        return point_world
    else:
        point_cam = torch.vstack((X, Y, Z, torch.ones(len(X)).to(point2d.device)))
        point_world = (torch.matmul(pose, point_cam)[0:3, :]).T # (n_points', 3)
        
        return point_world, valid_mask

def depth_image_to_point_cloud(rgb, depth, scale, K, pose):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()

    return points

def point3d_to_pixel_multiple_poses(
                    points:torch.Tensor,
                    K:torch.Tensor,
                    poses:torch.Tensor
                    ) -> torch.Tensor:
    '''
    Args:
        points: (n, 3) world coordinate, meter
        K: color camera K (n, 4, 4)
        poses: camera to world (n, 4, 4)
    Returns:
        (n, 2)
    '''
    pose_w2c = torch.linalg.inv(poses) # (n, 4, 4)
    points = torch.hstack((points, torch.ones((points.shape[0], 1)).to(points.device))).unsqueeze(-1) # (n, 4, 1)
    points_cam = torch.matmul(pose_w2c[:, :3, :], points) # (n, 3, 1)
    points_cam = points_cam.squeeze(-1) # (n, 3)

    X = points_cam[:, 0] #(n)
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    u = (X / Z) * K[:, 0, 0] + K[:, 0, 2]
    v = (Y / Z) * K[:, 1, 1] + K[:, 1, 2] # (n, )

    pixels = torch.vstack((u, v)).T

    return pixels

def point3d_to_pixel(points:torch.Tensor,
                     K:torch.Tensor,
                     pose:torch.Tensor
                     ) -> torch.Tensor:
    '''
    Args:
        points: (n, 3) world coordinate, meter
        K: color camera K (4, 4)
        pose: camera to world (4, 4)
    Returns:
        (n, 2)
    '''
    pose_w2c = torch.linalg.inv(pose)
    points = torch.hstack((points, torch.ones((points.shape[0], 1)).to(points.device))) # (n, 4)
    points_cam = torch.matmul(pose_w2c[:3, :], points.T) # (n, 3)
    points_cam = points_cam.T

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    u = (X / Z) * K[0, 0] + K[0, 2]
    v = (Y / Z) * K[1, 1] + K[1, 2]

    pixels = torch.vstack((u, v)).T

    return pixels

def point3d_to_pixel_np(points, K, pose) -> np.ndarray:
    '''
    Args:
        points: (n, 3) world
        K: depth camera K (4, 4)
        pose: camera to world (4, 4)
    Returns:
        (n, 2)
    '''
    pose_w2c = np.linalg.inv(pose)
    points = np.hstack((points, np.ones((points.shape[0], 2)))) # (n, 4)
    points_cam = np.dot(pose_w2c[:3, :], np.transpose(points)) # (n, 3)
    points_cam = np.transpose(points_cam)

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    u = (X / Z) * K[0, 0] + K[0, 2]
    v = (Y / Z) * K[1, 1] + K[1, 2]

    pixels = np.transpose(np.vstack((u, v)))

    """ points = np.hstack((points, np.ones((points.shape[0], 1)))) # (n, 4)
    points = np.transpose(points) # (4, n)
    points_cam = np.dot(pose_w2c, points)

    pixels = np.dot(K[0:3, 0:3], np.dot(pose_w2c[:3, :], points)[0:3, :]) # () """

    return pixels

def transform_world_points(points_w, pose_c2w):
    '''
    Args:
        points_w: (n, 3) world coordinates
        pose_c2w: (4, 4) cam to world

    Return:
        (n, 3)
    '''
    pose_w2c = np.linalg.inv(pose_c2w)
    points = np.dot(pose_w2c, points_w)
    return points

def is_point_visible(
    points:torch.Tensor, 
    pose:torch.Tensor,
    k:torch.Tensor
    ) -> torch.Tensor:
    '''
    Argrs:
        points      (n, 3)
        pose        (4, 4)
        k           (4, 4)
    Returns:
        vis_mask    (n)
    '''
    # Ensure inputs are torch tensors
    points = torch.tensor(points)
    pose = torch.tensor(pose)
    k = torch.tensor(k)

    # Perform the projection: 3D points -> Camera coordinates -> Image coordinates
    camera_coords = torch.matmul(pose[:3, :3], points.t()) + pose[:3, 3].unsqueeze(1)
    image_coords_homogeneous = torch.matmul(k[:3, :3], camera_coords)
    image_coords = image_coords_homogeneous[:2, :] / image_coords_homogeneous[2, :]

    # Get the image dimensions (the principal point is at (k[0, 2], k[1, 2]))
    image_width = k[0, 2] * 2
    image_height = k[1, 2] * 2

    # Check if the points are within the image boundaries
    is_visible = ((image_coords[0] >= 0) & (image_coords[0] < image_width) &
                  (image_coords[1] >= 0) & (image_coords[1] < image_height))

    return is_visible

def depth_to_point_cloud_torch(depth:torch.Tensor, 
                               scale:float, 
                               K:torch.Tensor,
                               pose:torch.Tensor
                               ) -> torch.Tensor:
    '''
    Args:
        depth: (h, w)
        scale: float
        K: (4, 4)
        pose: (4, 4) c2w
    Returns:
        position: (h*w*mask, 3)
    '''
    u = torch.arange(0, depth.shape[1])
    v = torch.arange(0, depth.shape[0])

    u, v = torch.meshgrid(u, v)
    u = u.type(torch.float).to(depth.device)
    v = v.type(torch.float).to(depth.device)

    Z = depth.type(torch.float).to(depth.device) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = torch.ravel(X)
    Y = torch.ravel(Y)
    Z = torch.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = torch.vstack((X, Y, Z, torch.ones(len(X))))
    position = torch.transpose(torch.dot(pose, position)[0:3, :])

    return position

def depth_to_point_cloud(depth, scale, K, pose):
    '''
    Args:
        depth: (h, w)
        scale: float
        K: (4, 4)
        pose: (4, 4) c2w
    Returns:
        position: (h*w*mask, 3)
    '''
    u = range(0, depth.shape[1])
    v = range(0, depth.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.transpose(np.dot(pose, position)[0:3, :])

    return position

# image_files: XXXXXX.png (RGB, 24-bit, PNG)
# depth_files: XXXXXX.png (16-bit, PNG)
# poses: camera-to-world, 4×4 matrix in homogeneous coordinates
def build_point_cloud(dataset_path, scale, view_ply_in_world_coordinate):
    K = np.fromfile(os.path.join(dataset_path, "K.txt"), dtype=float, sep="\n ")
    K = np.reshape(K, newshape=(3, 3))
    image_files = sorted(Path(os.path.join(dataset_path, "images")).files('*.png'))
    depth_files = sorted(Path(os.path.join(dataset_path, "depth_maps")).files('*.png'))

    if view_ply_in_world_coordinate:
        poses = np.fromfile(os.path.join(dataset_path, "poses.txt"), dtype=float, sep="\n ")
        poses = np.reshape(poses, newshape=(-1, 4, 4))
    else:
        poses = np.eye(4)

    for i in tqdm(range(0, len(image_files))):
        image_file = image_files[i]
        depth_file = depth_files[i]

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)

        if view_ply_in_world_coordinate:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[i])
        else:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses)
        save_ply_name = os.path.basename(os.path.splitext(image_files[i])[0]) + ".ply"
        save_ply_path = os.path.join(dataset_path, "point_clouds")

        if not os.path.exists(save_ply_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(save_ply_path)
        write_point_cloud(os.path.join(save_ply_path, save_ply_name), current_points_3D)

if __name__ == '__main__':
    dataset_folder = Path("dataset")
    scene = Path("hololens")
    # 如果view_ply_in_world_coordinate为True,那么点云的坐标就是在world坐标系下的坐标，否则就是在当前帧下的坐标
    view_ply_in_world_coordinate = False
    # 深度图对应的尺度因子，即深度图中存储的值与真实深度（单位为m）的比例, depth_map_value / real depth = scale_factor
    # 不同数据集对应的尺度因子不同，比如TUM的scale_factor为5000， hololens的数据的scale_factor为1000, Apollo Scape数据的scale_factor为200
    scale_factor = 1000.0
    build_point_cloud(os.path.join(dataset_folder, scene), scale_factor, view_ply_in_world_coordinate)