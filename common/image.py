from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from PIL import Image as im
from PIL import ImageDraw

from common.depth2Cloud import point3d_to_pixel, pixel_to_point3d

def is_point_visible(
    points_2d:torch.Tensor,
    img_h:int,
    img_w:int) -> torch.Tensor:
    '''
    Args:
        points_2d: (n, 2)
        img_h
        img_w
    Return:
        vis_mask:  (n,)
    '''
        # Extract x and y coordinates of the 2D points
    x_coords = points_2d[:, 0]
    y_coords = points_2d[:, 1]

    # Check if the points are within the image boundaries
    is_visible = ((x_coords >= 0) & (x_coords < img_w) &
                  (y_coords >= 0) & (y_coords < img_h))

    return is_visible

def render_kp_on_img(
    image:np.ndarray,       # (h, w, 3) [0, 255]
    keypoints:np.ndarray,   # (n, 2)
    color=(255, 0, 0, 255),
    kp_size:int=2
    ) -> np.ndarray:
    
    (h, w, _) = image.shape
    kp_image = im.fromarray(image.astype(np.uint8), "RGB")    
    draw = ImageDraw.Draw(kp_image)
    for kp in keypoints:
        draw.ellipse((kp[0]-kp_size, 
                        kp[1]-kp_size, 
                        kp[0]+kp_size, 
                        kp[1]+kp_size),
                        outline=color,
                        width=1)
    
    kp_image_np = np.array(kp_image, dtype=np.uint8)
    return kp_image_np   

def generate_patches(
    coords: torch.Tensor, 
    img_h:int,
    img_w:int,
    patch_size:int=1,
    ) -> Tuple[
        torch.Tensor,   # patched_coords
        torch.Tensor,   # mask
        ]:
    '''
    Args:
        coords:     (n_kp, 2) xy
        patch_size: should be an odd number
        img_h:      
        img_w:
        
    Returns:
        patched_coords  (n_kp*patch_size^2, 2)
        coords_mask     (n_kp*patch_size^2) mask patch points outside the image
    '''
    device = coords.device
    patch_radius = (patch_size - 1) / 2 
    patched_coords = torch.zeros(0, 2).to(device)
    coords_mask = torch.zeros(0).to(device)
    for patch_idx, coord in enumerate(coords):
        yy, xx = torch.meshgrid(torch.arange(coord[1]-patch_radius, coord[1]+patch_radius+1), # height 
                                torch.arange(coord[0]-patch_radius, coord[0]+patch_radius+1)) # weight
        patched_coord = torch.stack([xx.flatten(), yy.flatten()]).T.to(device)
        patched_coords = torch.cat((patched_coords, patched_coord))
        mask = (patched_coord[:, 0] > 0) * (patched_coord[:, 0] < img_w) * (patched_coord[:, 1] > 0) * (patched_coord[:, 1] < img_h)
        coords_mask = torch.cat((coords_mask, mask), dim=0)
    
    return patched_coords, coords_mask.type(torch.bool)

def get_values_by_coords(
    map:torch.Tensor,
    coords:torch.Tensor,
    ) -> torch.Tensor:
    ''' get values from a map according to the coordinates.
    Args:
        map:    (h, w, ...)
        coords: (n, 2) coordinates. xy
    Returns:
        values: (n, ...)
    '''
    n_coords = coords.shape[0]
    if map.dim  == 2:
        map = map.unsqueeze(-1)
    shapes = list(map.shape)[1:]
    shapes[0] = n_coords
    values = torch.zeros(shapes).to(map.device)
    for idx, coord in enumerate(coords):
        x = int(coord[0])
        y = int(coord[1])
        values[idx] = map[y, x]
    
    if len(shapes) > 1 and values.shape[-1] == 1:
        values.squeeze(-1)
        
    return values

def get_depth_by_coord(
    depth:torch.Tensor,
    coords:torch.Tensor,
    ) -> torch.Tensor:
    '''get depth values from the depth map by coordinates
    Args:
        depth:  (h, w) depth map.
        coords: (n, 2) coordinates. xy
    Reutrn:
        Z:      (n)
    '''
    n_coords = coords.shape[0]
    Z = torch.zeros(n_coords).to(depth.device)
    for idx, coord in enumerate(coords):
        x = coord[0]
        y = coord[1]
        Z[idx] = depth[y, x]
    return Z
        
def resize_image_np(image_np:np.ndarray,
              resize_h:int,
              resize_w:int
              ) -> np.ndarray:
    '''
    Args:
        image (h, w, 3) range [0, 255) int8
        resize_h
        resize_w
    Returns:
        image (resize_h, resize_w, 3)
    '''
    image = im.fromarray(image_np.astype(np.uint8), "RGB").resize((resize_w, resize_h))
    image_np = np.array(image, dtype=np.float32)
    return image_np

def resize_image(image:torch.Tensor,
           resize_h:int,
           resize_w:int
           ) -> torch.Tensor:
    '''
    resize tensor image
    Args:
        image (h, w, 3) range [0, 255) int
        resize_h
        resize_w
    Returns:
        image (resize_h, resize_w, 3)
    '''    
    device = image.device
    image_np = image.cpu().detach().numpy()
    resized_image_np = resize_image_np(image_np, resize_h, resize_w)
    image = torch.from_numpy(resized_image_np).to(device)
    return image

def reproj_validation(
    src_kp:torch.Tensor,
    src_kp_z:torch.Tensor,
    src_pose:torch.Tensor,
    dest_kp:torch.Tensor,
    dest_pose:torch.Tensor,
    K:torch.Tensor,
    reporj_err_thre:float=2.0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
    '''
    recover 3D position of source 2d kp, reproject to dest coordinate, 
    if the cloest dest kp to the reprojected src kp is less than the threshold, the dest kp is valid.
    
    Args:
        src_kp              (n_src_kp, 2)       xy
        src_kp_z            (n_src_kp)          mm, cam coordinate 
        src_pose            (4, 4)              c2w
        dest_kp             (n_dest_kp, 2)
        dest_pose           (4, 4)
        K                   (4, 4)              color cam
        reporj_err_thre:                        max pixel distance to be validated
    
    Returns:
        dest_valid_mask     (n_dest_kp)         valid kp in src_kp
        reproj_src_kp       (n_src_kp, 2)       reprojected coordinates of source kp on dest pose
        corr_src_idx        (n_dest_kp)         the index in source kp corresponding to the dest kp. 
                                                if source kp has no correspondence in dest kp, set index to -1
    '''
    src_kp_world = pixel_to_point3d(src_kp, src_kp_z, src_pose, K, True)    # (n_src_kp, 3), (n_src_kp)
    reproj_src_kp = point3d_to_pixel(src_kp_world, K, dest_pose)      # (n_src_kp, 2)
    
    (n_dest_kp, _) = dest_kp.shape
    dest_valid_mask = torch.ones(n_dest_kp).to(src_kp.device)
    corr_src_idx = torch.full((n_dest_kp,), -1).to(src_kp.device)
    for index, p in enumerate(dest_kp):
        p = p.unsqueeze(0)    # (1, 2)
        dist = torch.norm(reproj_src_kp - p, dim=1, p=2)
        knn = torch.topk(dist, k=1, largest=False)
        if knn.values[0] > reporj_err_thre:
            dest_valid_mask[index] = False
        else:
            corr_src_idx[index] = knn.indices[0]
    
    return dest_valid_mask, reproj_src_kp, corr_src_idx
    
def check_kp_repeatbility(
    kp1:torch.Tensor,
    Z:torch.Tensor,
    pose1:torch.Tensor,
    kp2:torch.Tensor,
    pose2:torch.Tensor,
    K:torch.Tensor,
    tolerance:int=2,
    ) -> torch.Tensor:
    '''
    Check if keypoints in kp1 can be found in kp2. 
    keypoint coordinates and poses are original scales.
    Args:
        kp1         (n_kp1, 2)   xy
        Z           (n_kp1)      mm, cam coordinate 
        pose1       (4, 4)       c2w
        kp2         (n_kp2, 2)
        pose2       (4, 4)
        K           (4, 4)      color cam
        tolerance:  max pixel distance to be validated
    
    Returns:
        valid_mask (n_kp1)
    '''    
    kp1_world, valid_mask = pixel_to_point3d(kp1, Z, pose1, K, False)    # (n_kp1, 3), (n_kp1)
    kp1_reproj = point3d_to_pixel(kp1_world, K, pose2)      # (n_kp1, 2)
    
    # if kp2 has kp near kp1_reproj within tolerance, the original kp1 is valid
    for index, p1 in enumerate(kp1_reproj):
        if not valid_mask[index]:
            continue
        p1 = p1.unsqueeze(0)    # (1, 2)
        dist = torch.norm(kp2 - p1, dim=1, p=2)
        knn = torch.topk(dist, k=1, largest=False)
        if knn.values[0] > tolerance:
            valid_mask[index] = False
    
    return valid_mask, kp1_reproj

def optimize_depth(
    kp:torch.Tensor,
    depth_map:torch.Tensor,
    radius:int,
    diff_thre:float=0.1
    ) -> torch.Tensor:
    ''' Find smallest depth (>0) around each kp
        if all surrounded depths of a kp are 0, return 0. otherwise, return the smallest depth > 0
    Args:
        kp:             (n_kp, 2)   xy
        depth_map       (h, w)      in meter
        radius          range
        diff_thre       if the difference between optimized kp depth and the original depth is less than the threshold, then keep the original depth
    Returns:
        kp_depth        (n_kp)
    '''
    (h, w) = depth_map.shape
    (n_kp, _) = kp.shape
    kp_depth = torch.zeros(n_kp).to(kp.device)
    for index, coord in enumerate(kp):
        x = (int)(coord[0])
        y = (int)(coord[1])
        ori_depth = depth_map[y, x]
        x_l = max(x-radius, 0)
        x_h = min(w-1, x+radius)
        y_l = max(y-radius, 0)
        y_h = min(h-1, y+radius)
        if torch.max(depth_map[y_l:y_h, x_l:x_h]) == 0:
            kp_depth[index] = 0
        else:
            t = depth_map[y_l:y_h, x_l:x_h]
            t = t[t>0].min()
            if abs(t-ori_depth) < diff_thre:
                t = ori_depth
            kp_depth[index] = t
    
    return kp_depth

def get_gradient(map:torch.Tensor) -> torch.Tensor:
    '''Get gradient map of a single value map 
    Args:
        map         (h, w)      
    Return:
        gradient    (h, w)
    '''
    (h, w) = map.shape
    gradient = torch.zeros(h, w)
    edge_mask = torch.BoolTensor(h, w)
    for y in range(0, h-2):
        for x in range(0, w-2):
            if (map[y, x] > 0 and (map[y, x+1] == 0 or map [y+1, x] == 0)) or (map[y, x] == 0 and (map[y, x+1] > 0 or map[y+1, x] > 0)):
                edge_mask[y, x] = True
                continue
            edge_mask[y, x] = False
            gx = abs(map[y, x+1] - map[y, x])
            gy = abs(map[y+1, x] - map[y, x])
            gradient[y, x] = gx + gy
    
    gradient[edge_mask] = gradient.max()
    
    return gradient

def remove_plane_kp_lst(
    kp:torch.Tensor,
    depth_map:torch.Tensor,
    pc:torch.Tensor,
    K:torch.Tensor,
    pose:torch.Tensor,
    residual_thre:float,
    radius:int=3
    ) -> torch.Tensor:
    '''Remove surface keypoints via line fitting
    Args:
        kp:             (n_kp, 2)   xy
        depth_map       (h, w)      in meter
        pc              (n, 3)      point cloud
        K               (4, 4)      camera intrinsic matrix
        pose            (4, 4)      camera extrinsic matrix
        residual_thre   threhold of residual to determin if all surounding points are on the same surface.
        radius:         detection radius  
    Returns:
        kp              (n_kp', 2)
        edges           (h, w, 1)
    '''
    (h, w) = depth_map.shape
    (n_kp, _) = kp.shape
    
    det_patch_size = radius * 2 + 1
    kp_patches, patch_mask = generate_patches(kp, h, w, det_patch_size)
    
    kp_patches_depth = torch.zeros(n_kp, det_patch_size*det_patch_size).to(kp.device) # (n_kp, det_patch_size^2)
    for index, coord in enumerate(kp):
        x = int(coord[0])
        y = int(coord[1])
        kp_patches[index] = kp[det_patch_size * det_patch_size]
        kp_patches_depth[index] = depth_map[y-radius:y+radius+1, x-radius:x+radius+1].reshape(-1)
        
    kp_patches_3d, valid_mask = pixel_to_point3d(kp_patches, kp_patches_depth.reshape(-1), pose, K, False) # (n_kp*det_patch_size^2, 3), (n_kp*det_patch_size^2)
    kp_patches_3d = kp_patches_3d.reshape(n_kp, -1, 3) #(n_kp, det_patch_size^2, 3)
    valid_mask = valid_mask.reshape(n_kp, -1) #(n_kp, det_patch_size^2)
    
    A = torch.cat((kp_patches_3d[:, :, 0:2], torch.ones(kp_patches_3d.shape[0], kp_patches_3d.shape[1], 1).to(kp.device)), dim=2) # (n_kp, det_patch_size^2, 3)
    B = kp_patches_3d[:, :, 2].unsqueeze(-1) # (n_kp, det_patch_size^2, 1)
    
    residuals = torch.zeros(n_kp).to(kp.device)
    for idx_kp, a in enumerate(A): # a (det_patch_size^2, 3)
        b = B[idx_kp] # (det_patch_size^2, 1)
        mask = valid_mask[idx_kp] # (det_patch_size^2)
        a = a[mask]
        b = b[mask]
        
        if mask[mask==True].shape[0] < 4:
            residuals[idx_kp] = 0
            continue
        
        x = torch.linalg.lstsq(a, b)
        solution = x.solution
        residuals[idx_kp] = x.residuals
    
    filtered_kp = kp[residuals>residual_thre]
    return filtered_kp

def remove_plane_kp(
        kp:torch.Tensor,
        depth_map:torch.Tensor,
        minVal:float=1,
        maxVal:float=20,
        radius:int=3
        ) -> torch.Tensor:
    '''Remove keypoints which are on flat surface
    Args:
        kp:             (n_kp, 2)   xy
        depth_map       (h, w)      in meter
        minVal          min value for canny edge detection
        maxVal          max value for canny edge detection
        radius:         detection radius  
    Returns:
        kp              (n_kp', 2)
        edges           (h, w, 1)
    '''
    (h, w) = depth_map.shape
    filtered_kp = torch.zeros(0, 2).type(torch.int32).to(kp.device)
    
    depth_norm = (depth_map / depth_map.max()) * 255
    depth_norm = depth_norm.type(torch.uint8) 
    edges = torch.from_numpy(cv.Canny(depth_norm.cpu().detach().numpy(), minVal, maxVal, apertureSize=3, L2gradient=True))
    
    for index, coord in enumerate(kp):
        x = coord[0]
        y = coord[1]
        x_l = max(x-radius, 0).type(torch.int)
        x_h = min(w-1, x+radius).type(torch.int)
        y_l = max(y-radius, 0).type(torch.int)
        y_h = min(h-1, y+radius).type(torch.int)
        patch = edges[y_l:y_h, x_l:x_h]
        if patch.max() > 0:
            filtered_kp = torch.cat((filtered_kp, coord.unsqueeze(0)), dim=0)
    
    edges = edges.unsqueeze(-1)
    return filtered_kp, edges

def remove_plane_kp__(
        kp:torch.Tensor,
        depth_map:torch.Tensor,
        gradient_thre:float,
        radius:int):
    '''Remove keypoints which are on flat surface
    Args:
        kp:             (n_kp, 2)   xy
        depth_map       (h, w)      in meter
        gradient_thre   if gradient is smaller than the threshold, the kp is on the flat surface
        radius:         the radius of the gradient 
    Returns:
        kp              (n_kp', 2)
    '''
    (h, w) = depth_map.shape
    gradient = get_gradient(depth_map)
    gradient_2 = get_gradient(gradient)
    filtered_kp = torch.zeros(0, 2).type(torch.int32)
    
    for index, coord in enumerate(kp):
        x = coord[0]
        y = coord[1]
        x_l = max(x-radius, 0)
        x_h = min(w-2, x+radius)
        y_l = max(y-radius, 0)
        y_h = min(h-2, y+radius)
        patch_gradient = gradient_2[y_l:y_h, x_l:x_h]
        if patch_gradient.max() > gradient_thre:
            filtered_kp = torch.cat((filtered_kp, coord.unsqueeze(0)), dim=0)
    
    return filtered_kp
    
def remove_plane_kp_(
        kp:torch.Tensor,
        depth_map:torch.Tensor,
        gradient_thre:float,
        radius:int):
    '''Remove keypoints which are on flat surface
    Args:
        kp:             (n_kp, 2)   xy
        depth_map       (h, w)      in meter
        gradient_thre   if gradient is smaller than the threshold, the kp is on the flat surface
        radius:         the radius of the gradient 
    Returns:
        kp              (n_kp', 2)
    '''
    (h, w) = depth_map.shape
    filtered_kp = torch.zeros(0, 2).type(torch.int32)
    
    for index, coord in enumerate(kp):
        x = coord[0]
        y = coord[1]
        x_l = max(x-radius, 0)
        x_h = min(w-1, x+radius+1)
        y_l = max(y-radius, 0)
        y_h = min(h-1, y+radius+1)
        patch = depth_map[y_l:y_h, x_l:x_h]
        patch_gradient = get_gradient(get_gradient(patch))
        if patch_gradient.max() > gradient_thre:
            filtered_kp = torch.cat((filtered_kp, coord.unsqueeze(0)), dim=0)
    
    return filtered_kp

def optimize_kp_by_depth(
    kp_coords:torch.Tensor,
    depth:torch.Tensor,
    search_radius:int=2,
    depth_diff_thr:float=0.02,
    img_h:int=480,
    img_w:int=640,
    ) -> torch.Tensor:
    '''
    Invalid depth is 65535
    Args:
        kp_coords:  (n, 2)
        depth:      (h, w)
        search_radius:
    Returns:
        refined_kp: (n, 2)
    '''
    kps_optim = kp_coords.clone()
    for kp_idx, kp_coord in enumerate(kp_coords):
        kp_depth = get_values_by_coords(depth, kp_coord.unsqueeze(0))
        for offset in range(1, search_radius+1):
            x = kp_coord[0]
            y = kp_coord[1]
            yy, xx = torch.meshgrid(torch.arange(max(y-offset, 0), min(y+offset+1, img_h)), # height 
                                                torch.arange(max(x-offset, 0), min(x+offset+1, img_w))) # width
            search_coords = torch.stack([xx.flatten(), yy.flatten()]).T.to(kp_coords.device)
            search_depths = get_values_by_coords(depth, search_coords)
            diff_mask = search_depths - kp_depth > depth_diff_thr
            if diff_mask.sum() == 0:
                continue
            
            search_depths = search_depths[diff_mask]
            search_coords = search_coords[diff_mask]
            replace_idx = torch.topk(search_depths, 1).indices[0]
            kps_optim[kp_idx] = search_coords[replace_idx]
            break
    return kps_optim

def check_visibility(
    kps3d:torch.Tensor,
    pose:torch.Tensor,
    depth:torch.Tensor,
    k:torch.Tensor,
    vis_dist_thre:float,
    depth_optimize_radius:int=2,
    ) -> torch.Tensor:
    '''
    Args:
        kps3d:  [n_kps3d, 3]
        poses:  [4, 4]
        depths: [h, w]
        k:      [4, 4]
    Returns:
        kps3d_vis_mask: [n_kps3d]
    '''
    # project kps3d to 2d
    proj_kps3d = point3d_to_pixel(kps3d, k, pose)
    
    proj_kps3d_vis_mask = is_point_visible(proj_kps3d, depth.shape[0], depth.shape[1])
    
    proj_kps3d = proj_kps3d[proj_kps3d_vis_mask]
    kps3d = kps3d[proj_kps3d_vis_mask]
    # convert the proj kps3d depth to 3D points
    proj_kps3d_depth = optimize_depth(proj_kps3d, depth, radius=2)
    kps3d_vis, kps3d_vis_valid_mask = pixel_to_point3d(proj_kps3d, proj_kps3d_depth, pose, k, remove_invalid=False)
    
    # check dist between kps3d_vis and kps3d
    vis_mask = torch.linalg.norm(kps3d_vis - kps3d, dim=1) < vis_dist_thre
    
    recovered_vis_mask = torch.zeros_like(proj_kps3d_vis_mask)
    recovered_vis_mask[proj_kps3d_vis_mask] = vis_mask * kps3d_vis_valid_mask
    
    return recovered_vis_mask