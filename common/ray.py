from typing import Tuple
import torch
import torch.nn.functional as F
from .image import generate_patches

def generate_rays_at_poses(
    coords:torch.Tensor,
    poses:torch.Tensor,
    K:torch.Tensor,
    opengl_cam:bool=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Args:
        coord: (n, 2) 2D sample point coordinate (xy) on the camera projection plane
        pose: (n, 4, 4), camera to world
        K: (n, 4, 4) color cam K
    Returns:
        origin  (n, 3)
        viewdir (n, 3)
    '''
    device = coords.device

    x = coords[:, 0]
    y = coords[:, 1]
    
    # generate rays
    camera_dirs = F.pad(
        torch.stack(
            [(x - K[:, 0, 2] + 0.5) / K[:, 0, 0],
             (y - K[:, 1, 2] + 0.5) / K[:, 1, 1] * (-1.0 if opengl_cam else 1.0),],
            dim=-1,
        ), # (n, 2)
        (0, 1),
        value=(-1.0 if opengl_cam else 1.0),
    )  # (n, 3)

    c2w = poses[:, :3, :]                # (n, 3, 4)
    direction = (camera_dirs * c2w[:, :3, :3]).sum(dim=-1) # (n, 3)
    origins = c2w[:, :3, -1]    # (n, 3)
    dirs = direction / torch.linalg.norm(direction, dim=-1, keepdims=True) # (3)
    
    return origins, dirs

def generate_ray_at(coord: torch.Tensor, pose:torch.Tensor, K:torch.Tensor, opengl_cam:bool=False
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Args:
        coord: (2) 2D sample point coordinate (xy) on the camera projection plane
        pose: (4, 4), camera to world
        K: (4, 4) color cam K
        
    Returns:
        origin  (3)
        viewdir (3)
    '''
    x = coord[0]
    y = coord[1]
    
    # generate rays
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if opengl_cam else 1.0),
            ],
            dim=-1,
        ), # (2)
        (0, 1),
        value=(-1.0 if opengl_cam else 1.0),
    )  # (3)

    c2w = pose[:3, :]                # (3, 4)
    direction = (camera_dirs * c2w[:3, :3]).sum(dim=-1) # (3)
    origin = c2w[:3, -1]    # (3)
    viewdir = direction / torch.linalg.norm(direction, dim=-1, keepdims=True) # (3)
    
    return origin, viewdir

def generate_ray_patches_at(
    coords: torch.Tensor, 
    pose:torch.Tensor, 
    K:torch.Tensor, 
    img_h:int,
    img_w:int,
    patch_size:int=1,
    opengl_cam:bool=False
    ) -> Tuple[
        torch.Tensor,   # origins 
        torch.Tensor,   # dirs
        torch.Tensor,   # patched_coords
        torch.Tensor,   # mask
        ]:
    '''
    Args:
        coords:     (n_rays, 2) xy
        pose:       (4, 4), camera to world
        K:          (4, 4) color cam K
        patch_size: should be an odd number
        img_h:      
        img_w:
        opengl_cam
        
    Returns:
        origins:        (n_rays*patch_size^2, 3) 
        dirs:           (n_rays*patch_size^2, 3)
        patched_coords  (n_rays*patch_size^2, 2)
        mask            (n_rays*patch_size^2) mask rays outside the image
    '''        
    patched_coords, patch_coord_mask = generate_patches(
        coords, 
        img_h,
        img_w,
        patch_size)
    
    origins, dirs = generate_rays_at(
        patched_coords,
        pose,
        K,
        opengl_cam
    )
    
    return origins, dirs, patched_coords, patch_coord_mask
    
def generate_rays_at(
    coords: torch.Tensor, 
    pose:torch.Tensor, 
    K:torch.Tensor, 
    opengl_cam:bool=False
    ) -> Tuple[
        torch.Tensor,   # origins 
        torch.Tensor    # dirs
        ]:
    '''
    Args:
        coords:     (n_rays, 2) xy
        pose:       (4, 4), camera to world
        K:          (4, 4) color cam K
        opengl_cam
        
    Returns:
        origins:    (n_rays, 3)
        dirs:       (n_rays, 3)
    '''
    device = coords.device
    origins = torch.zeros(0, 3).to(device)
    dirs = torch.zeros(0, 3).to(device)
    
    for coord in coords:
        origin, dir = generate_ray_at(coord, pose, K, opengl_cam)
        origins = torch.cat((origins, origin.unsqueeze(0)), dim=0).to(device)
        dirs = torch.cat((dirs, dir.unsqueeze(0)), dim=0).to(device)
    
    return origins, dirs

def generate_per_pixel_rays(
    image:torch.Tensor, 
    pose:torch.Tensor, 
    K:torch.Tensor, 
    edge_buffer:int=0, 
    opengl_cam:bool=False
    ) -> Tuple[
        torch.Tensor,  # origins 
        torch.Tensor]: # dirs 
    '''
    Args:
        image: (h, w, 3)
        pose: (4, 4), camera to world
        K: (4, 4) color cam K
        edge_buffer: int. skip pixels on edges
        opengl_cam: 
    
    Returns:
        origins (n_ray, 3)  n_ray = h * w
        dirs (n_ray, 3)
    '''
    img_h = image.shape[0]
    img_w = image.shape[1]
    num_rays = (img_h - 2 * edge_buffer) * (img_w - 2 * edge_buffer)
    
    yy, xx = torch.meshgrid(
        torch.arange(edge_buffer, img_h - edge_buffer),
        torch.arange(edge_buffer, img_w - edge_buffer))
    coords = torch.stack([xx.flatten(), yy.flatten()]).T.to(image.device)    # (n_rays, 2)

    x = coords[:, 0].squeeze()  # (n_rays)
    y = coords[:, 1].squeeze()  # (n_rays)
    
    # generate rays
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if opengl_cam else 1.0),
            ],
            dim=-1,
        ), # [num_rays, 2]
        (0, 1),
        value=(-1.0 if opengl_cam else 1.0),
    )  # [num_rays, 3]

    c2w = pose[:3, :].unsqueeze(0).expand(num_rays, -1, -1)             # (num_rays, 3, 4)
    # [n_cams, height, width, 3]
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1) # (num_rays, 3)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)      # (num_rays, 3)
    viewdirs = directions / torch.linalg.norm(                          # (num_rays, 3)
        directions, dim=-1, keepdims=True
    )

    origins = torch.reshape(origins, (num_rays, 3))
    viewdirs = torch.reshape(viewdirs, (num_rays, 3))
    
    return origins, viewdirs



class Point:
    def __init__(self, point_x, point_y, point_z):
        self.coord = [point_x, point_y, point_z]


# self.origin 为线段起始点坐标，坐标等同于 point_start
# self.direction 可视为线段的方向向量
class LineSegment:
    def __init__(self, point_start, point_end):
        origin = []
        direction = []
        for index in range(3):
            origin.append(point_start.coord[index])
            direction.append(point_end.coord[index] - point_start.coord[index])

        self.origin = origin
        self.direction = direction
    
    # 通过系数 t 获得其对应的线段上的点
    # 0 <= t <= 1 意味着点在线段上
    def get_point(self, coefficient):
        point_coord = []
        for index in range(3):
            point_coord.append(self.origin[index] + coefficient * self.direction[index])
        return Point(point_coord[0], point_coord[1], point_coord[2])


# point_a, point_b 为平行于坐标轴的立方体处于对角位置的两个顶点
class Box:
    def __init__(self, point_a, point_b):
        self.pA = point_a
        self.pB = point_b

    # 获得立方体与线段 line_segment 的两个交点
    def get_intersect_point(self, line_segment):
        # 线段 direction 分量存在 0  预处理
        for index, direction in enumerate(line_segment.direction):
            if direction == 0:
                box_max = max(self.pA.coord[index], self.pB.coord[index])
                box_min = min(self.pA.coord[index], self.pB.coord[index])
                if line_segment.origin[index] > box_max or line_segment.origin[index] < box_min:
                    return None, None, None, None

        # 常规处理
        t0, t1 = 0., 1.
        for index in range(3):
            if line_segment.direction[index] != 0.:
                inv_dir = 1. / line_segment.direction[index]
                t_near = (self.pA.coord[index] - line_segment.origin[index]) * inv_dir
                t_far = (self.pB.coord[index] - line_segment.origin[index]) * inv_dir
                if t_near > t_far:
                    t_near, t_far = t_far, t_near
                t0 = max(t_near, t0)
                t1 = min(t_far, t1)
                #if t0 > t1:
                    #return None, None, None, None
        intersection_point_near = line_segment.get_point(t0)
        intersection_point_far = line_segment.get_point(t1)

        return intersection_point_near, intersection_point_far, t0, t1,

    # 获得立方体与线段的交线长度
    def get_intersect_length(self, line_segment):
        near_point, far_point = self.get_intersect_point(line_segment)
        if near_point is None:
            return 0.
        length = 0.
        for index in range(3):
            length += (far_point.coord[index] - near_point.coord[index]) ** 2
        return length ** 0.5

def get_ray_point_in_cube(
    cube_pa:list,
    cube_pb:list,
    line_s:list,
    line_e:list,
    t:float,
    ) -> list:
    ''' Get a point on the line and in the cube.
    Args:
        cube_pa:    (3)
        cube_pb:    (3)
        line_s:     (3)
        line_e:     (3)
        t:          [0, 1]
    Return
        pt:         (3)
    '''
    box = Box(Point(cube_pa[0], cube_pa[1], cube_pa[2]), Point(cube_pb[0], cube_pb[1], cube_pb[2]))
    line = LineSegment(Point(line_s[0], line_s[1], line_s[2]), Point(line_e[0], line_e[1], line_e[2]))
    intersection_point_near, intersection_point_far, t0, t1 = box.get_intersect_point(line)
    t_p = (t1-t0) * t + t0
    p = line.get_point(t_p)
    return p.coord



# self.origin 为线段起始点坐标，坐标等同于 point_start
# self.direction 可视为线段的方向向量
class LineSegment_torch:
    def __init__(self, 
                 point_start:torch.Tensor,  #(n, 3)
                 point_end:torch.Tensor,    #(n, 3)
                 ):
        self.origin = point_start # (n, 3)
        self.direction = point_end - point_start # (n, 3)
    
    # 通过系数 t 获得其对应的线段上的点
    # 0 <= t <= 1 意味着点在线段上
    def get_point(self, 
                  coefficient:torch.Tensor # (n)
                  ):
        point_coord = self.origin + coefficient.unsqueeze(-1) * self.direction
        return point_coord # (n, 3)

# point_a, point_b 为平行于坐标轴的立方体处于对角位置的两个顶点
class Box_torch:
    def __init__(self, 
                 point_a:torch.Tensor, # (3)
                 point_b:torch.Tensor, # (3)
                 ):
        self.pA = point_a
        self.pB = point_b

    # 获得立方体与线段 line_segment 的两个交点
    def get_intersect_point(self, line_segment:LineSegment_torch):
        # 常规处理
        inv_dir = 1. / line_segment.direction   # (n, 3)
        t_near = (self.pA.unsqueeze(0) - line_segment.origin) * inv_dir # (n, 3)
        t_far = (self.pB.unsqueeze(0) - line_segment.origin) * inv_dir # (n, 3)
        t_near_c = t_near.clone()
        t_far_c = t_far.clone()
        t_near = torch.min(t_near_c, t_far_c)
        t_far = torch.max(t_near_c, t_far_c)
        
        t_near_max = t_near.max(dim=1).values # (n)
        t0 = torch.zeros_like(t_near_max).to(t_near_max.device)
        t0 = torch.max(t_near_max, t0)  # (n)
        
        t_far_min = t_far.min(dim=1).values # (n)
        t1 = torch.ones_like(t_far_min).to(t_far_min.device)
        t1 = torch.min(t_far_min, t1)   # (n)
        
        intersection_point_near = line_segment.get_point(t0)
        intersection_point_far = line_segment.get_point(t1)

        return intersection_point_near, intersection_point_far, t0, t1, # (n, 3) (n, 3) (n) (n)

    # 获得立方体与线段的交线长度
    def get_intersect_length(self, line_segment):
        near_point, far_point = self.get_intersect_point(line_segment)
        if near_point is None:
            return 0.
        length = 0.
        for index in range(3):
            length += (far_point.coord[index] - near_point.coord[index]) ** 2
        return length ** 0.5

def get_ray_point_in_cube_torch(
    cube_pa:torch.Tensor,
    cube_pb:torch.Tensor,
    line_s:torch.Tensor,
    line_e:torch.Tensor,
    t:torch.Tensor,
    ) -> torch.Tensor:
    ''' Get a point on the line and in the cube.
    Args:
        cube_pa:    (3)
        cube_pb:    (3)
        line_s:     (n, 3)
        line_e:     (n, 3)
        t:          (n) [0, 1]
    Return
        pt:         (n, 3)
    '''
    box = Box_torch(cube_pa, cube_pb)
    line = LineSegment_torch(line_s, line_e)
    intersection_point_near, intersection_point_far, t0, t1 = box.get_intersect_point(line)
    t_p = (t1-t0) * t + t0
    p = line.get_point(t_p)
    return p