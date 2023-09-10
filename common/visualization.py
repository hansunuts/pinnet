import sys
import os
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.append(project_root)

from posixpath import split
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
import copy
from PIL import Image as im

from sqlalchemy import null

def visualize_pc(np_pts):
    '''
    input:
        np_pts    [n, 3]
    '''
    original_pointcloud = o3d.geometry.PointCloud()

    kp_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    kp_mesh.paint_uniform_color([1, 0, 0])

    original_pointcloud.points = o3d.utility.Vector3dVector(np_pts)
    original_pointcloud.paint_uniform_color([0.2, 1, 0])

    spheres, cylinders, box_lines, mesh_frame = get_unit_box()
    o3d.visualization.draw_geometries([original_pointcloud, spheres, cylinders, box_lines, mesh_frame])

def visualize_pc_kp(np_pts, np_key_pts, origin=[-0.5, -0.5, -0.5]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pts)
    pcd.paint_uniform_color([0.2, 1, 0])

    kp = o3d.geometry.PointCloud()
    kp.points = o3d.utility.Vector3dVector(np_key_pts)
    kp.paint_uniform_color([1, 0, 0])

    spheres, cylinders, box_lines, mesh_frame = get_unit_box()
    o3d.visualization.draw_geometries([pcd, kp, mesh_frame, spheres, cylinders, box_lines])

def visualize_pc_kp_compare(np_pts_1, np_kp_1, np_pts_2, np_kp_2):
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(np_pts_1)
    pcd_1.paint_uniform_color([0.2, 1, 0])

    kp_1 = o3d.geometry.PointCloud()
    kp_1.points = o3d.utility.Vector3dVector(np_kp_1)
    kp_1.paint_uniform_color([1, 0, 0])
    
    spheres_1, cylinders_1, box_lines_1, mesh_frame_1 = get_unit_box()

    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(np_pts_2)
    pcd_2.paint_uniform_color([0.2, 1, 0])
    pcd_2.translate([1, 0, 0])

    kp_2 = o3d.geometry.PointCloud()
    kp_2.points = o3d.utility.Vector3dVector(np_kp_2)
    kp_2.paint_uniform_color([1, 0, 0])
    kp_2.translate([1, 0, 0])

    spheres_2, cylinders_2, box_lines_2, mesh_frame_2 = get_unit_box([0.5, -0.5, -0.5])

    o3d.visualization.draw_geometries([pcd_1, kp_1, pcd_2, kp_2, 
        spheres_1, cylinders_1, box_lines_1, mesh_frame_1,
        spheres_2, cylinders_2, box_lines_2, mesh_frame_2])

def visualize_occ(np_pts, np_occ, occ_vis_thr=0.9):
    occ_shperes = o3d.geometry.TriangleMesh()
    for index, point in enumerate(np_pts):
        occ = np_occ[index]
        if occ > occ_vis_thr:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003 + 0.007 * (occ-occ_vis_thr)/(1-occ_vis_thr))
            #sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate([point[0], point[1], point[2]])
            sphere.paint_uniform_color([(occ-occ_vis_thr)/(1-occ_vis_thr), 0, 0])
            occ_shperes += sphere

    spheres, cylinders, box_lines, mesh_frame = get_unit_box()
    o3d.visualization.draw_geometries([occ_shperes, mesh_frame, spheres, cylinders, box_lines])

def visualize_occ_sal(np_pts, np_occ, np_sal, occ_vis_thr=0.9, sal_vis_thr=0.5):
    occ_shperes = o3d.geometry.TriangleMesh()
    for index, point in enumerate(np_pts):
        occ = np_occ[index]
        sal = np_sal[index]
        if occ > occ_vis_thr:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003 + 0.007 * (occ-occ_vis_thr)/(1-occ_vis_thr))
            #sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate([point[0], point[1], point[2]])
            if sal < sal_vis_thr:
                sal = 0
            else:
                sal = sal-sal_vis_thr
            sphere.paint_uniform_color([sal/(1-sal_vis_thr), 0, 0])
            occ_shperes += sphere

    spheres, cylinders, box_lines, mesh_frame = get_unit_box()
    o3d.visualization.draw_geometries([occ_shperes, mesh_frame, spheres, cylinders, box_lines])

def visualize_occ_sal_pcd(np_pcd, np_pts, np_occ, np_sal, occ_vis_thr=0.9, sal_vis_thr=0.5):
    occ_shperes = o3d.geometry.TriangleMesh()
    for index, point in enumerate(np_pts):
        occ = np_occ[index]
        sal = np_sal[index]
        if occ > occ_vis_thr:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003 + 0.007 * (occ-occ_vis_thr)/(1-occ_vis_thr))
            #sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate([point[0], point[1], point[2]])
            if sal < sal_vis_thr:
                sal = 0
            else:
                sal = sal-sal_vis_thr
            sphere.paint_uniform_color([sal/(1-sal_vis_thr), 0, 0])
            occ_shperes += sphere

    spheres, cylinders, box_lines, mesh_frame = get_unit_box()

    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(np_pcd)
    pcd_2.paint_uniform_color([0.2, 1, 0])
    pcd_2.translate([1, 0, 0])

    spheres_2, cylinders_2, box_lines_2, mesh_frame_2 = get_unit_box([0.5, -0.5, -0.5])

    o3d.visualization.draw_geometries([occ_shperes, mesh_frame, spheres, cylinders, box_lines,
                                        pcd_2, spheres_2, cylinders_2, box_lines_2, mesh_frame_2])

def visualise_occ_sal_with_cam(
    # camera
    view_w_px, view_h_px, intrinsic, extrinsic, scale,
    np_pts, np_occ, np_sal, occ_vis_thr=0.9, sal_vis_thr=0.5):
    
    camera = get_camera(view_w_px, view_h_px, intrinsic, extrinsic, scale)
    occ_shperes = o3d.geometry.TriangleMesh()
    for index, point in enumerate(np_pts):
        occ = np_occ[index]
        sal = np_sal[index]
        if occ > occ_vis_thr:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003 + 0.007 * (occ-occ_vis_thr)/(1-occ_vis_thr))
            #sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate([point[0], point[1], point[2]])
            if sal < sal_vis_thr:
                sal = 0
            else:
                sal = sal-sal_vis_thr
            sphere.paint_uniform_color([sal/(1-sal_vis_thr), 0, 0])
            occ_shperes += sphere

    spheres, cylinders, box_lines, mesh_frame = get_unit_box()
    
    o3d.visualization.draw_geometries([occ_shperes, mesh_frame, spheres, cylinders, box_lines, camera])

def visualize_pc_kp_with_cam(
    # camera
    view_w_px, view_h_px, intrinsic, extrinsic, scale,
    # objects
    np_pts, np_key_pts, origin=[-0.5, -0.5, -0.5]
    ):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pts)
    pcd.paint_uniform_color([0.2, 1, 0])

    kp = o3d.geometry.PointCloud()
    kp.points = o3d.utility.Vector3dVector(np_key_pts)
    kp.paint_uniform_color([1, 0, 0])

    spheres, cylinders, box_lines, mesh_frame = get_unit_box()
    camera = get_camera(view_w_px, view_h_px, intrinsic, extrinsic, scale)
    
    o3d.visualization.draw_geometries([pcd, 
                                       kp, 
                                       mesh_frame, 
                                       spheres, 
                                       cylinders, 
                                       box_lines, 
                                       camera])
def visualize_image(image):
    '''pixel value range [0, 1], brighter is higher
    Args:
        image: (h, w, channel) np
    '''
    img = o3d.geometry.Image(image)
    o3d.visualization.draw_geometries([img])
    
def visualize_score_on_image(image, score_image, color='g'):
    '''
    Args:
        image (h_img, w_img, 3) np, RGB [0, 255]
        scroe_image (h, w)  np [0, 1]
        color: 'r', 'g', 'b'
    '''
    (h_img, w_img, _) = image.shape
    (h, w) = score_image.shape
    if color == 'r':
        color_index = 0
    elif color == 'g':
        color_index = 1
    elif color == 'b':
        color_index = 2
    
    # expand score image to RGBA
    score_image = np.expand_dims(score_image, axis=-1) # (h, w, 1)
    score_image_rgba = np.full((h, w, 4), (0, 0, 0, 0)).astype(np.float32) # (h, w, rgba)
    score_image_rgba[:, :, color_index] += score_image[:, :, 0]
    score_image_rgba[:, :, 3] += score_image[:, :, 0]
    max = score_image_rgba.max()
    score_image_rgba /= max
    score_image_rgba *= 255
    
    vis_image = im.fromarray(image.astype(np.uint8), "RGB").resize((w, h))
    #vis_image.show()
    vis_sal2d = im.fromarray(score_image_rgba.astype(np.uint8), "RGBA")
    #vis_sal2d.show()
    vis_image.paste(vis_sal2d, (0, 0), vis_sal2d)
    #vis_image.show()
    
    img = o3d.geometry.Image(np.array(vis_image))
    o3d.visualization.draw_geometries([img,])
    
def get_score_image(scores:np.ndarray, 
                    color=(255, 0, 0)
                  ) -> np.ndarray:
    '''
    Args:
        scores (h, w)  np [0, 1]
        color: 'r', 'g', 'b'
    Returns:
        score_image (h, w, 4) rgba
    '''
    (h_img, w_img, _) = scores.shape
    score_image_rgb = np.full((h_img, w_img, 3), (0, 0, 0)).astype(np.int8) # (h, w, rgb)
    for h in range(0, h_img):
        for w in range(0, w_img):
            r, g, b = get_heatmap_value(scores[h, w], offset, amplifier)
            score_image_rgb[h, w] = np.array([r, g, b])
    
    return score_image_rgb
    
def visualize_heatmap(image, score_image, offset=0.0, amplifier=1.0):
    '''
    Args:
        image (h_img, w_img) np [0, 255]
        scroe_image (h, w)  np [0, 1]
    '''
    (h_img, w_img, _) = image.shape
    (h_score, w_score) = score_image.shape
    
    # expand score image to RGBA
    score_image_rgba = np.full((h_score, w_score, 4), (0, 0, 0, 0)).astype(np.float32) # (h, w, rgba)
    
    for h in range(0, h_score):
        for w in range(0, w_score):
            r, g, b = get_heatmap_value(score_image[h, w], offset, amplifier)
            score_image_rgba[h, w] = np.array([r, g, b, 200])
    
    vis_image = im.fromarray(image.astype(np.uint8), "RGB").resize((w_score, h_score))
    #vis_image.show()
    vis_sal2d = im.fromarray(score_image_rgba.astype(np.uint8), "RGBA")
    #vis_sal2d.show()
    vis_image.paste(vis_sal2d, (0, 0), vis_sal2d)
    #vis_image.show()
    
    img = o3d.geometry.Image(np.array(vis_image))
    o3d.visualization.draw_geometries([img,])

def visualize_heatmap_highlight_ray(image, score_image, ray_coord, offset=0.0, amplifier=1.0):
    '''
    Args:
        image (h_img, w_img) np [0, 255]
        scroe_image (h, w)  np [0, 1]
        ray_coord (2) np h, w
    '''
    (h_img, w_img, _) = image.shape
    (h_score, w_score) = score_image.shape
    
    # expand score image to RGBA
    score_image_rgba = np.full((h_score, w_score, 4), (0, 0, 0, 0)).astype(np.float32) # (h, w, rgba)
    
    for h in range(0, h_score):
        for w in range(0, w_score):
            r, g, b = get_heatmap_value(score_image[h, w], offset, amplifier)
            score_image_rgba[h, w] = np.array([r, g, b, 200])
            
    score_image_rgba[ray_coord[0], ray_coord[1]] = np.array([255, 255, 255, 255])
    
    vis_image = im.fromarray(image.astype(np.uint8), "RGB").resize((w_score, h_score))
    #vis_image.show()
    vis_sal2d = im.fromarray(score_image_rgba.astype(np.uint8), "RGBA")
    #vis_sal2d.show()
    vis_image.paste(vis_sal2d, (0, 0), vis_sal2d)
    #vis_image.show()
    
    img = o3d.geometry.Image(np.array(vis_image))
    o3d.visualization.draw_geometries([img,])
    
def get_heatmap(scores:np.ndarray, offset=0.0, amplifier=1.0
                ) -> np.ndarray:
    '''
    Args:
        scores: (h, w, 1), value range [0, 1]
        offset:
        amplifier:
        
    Returns:
        heatmap: (h, w, 3), value range [0, 255]
    '''
    (h_img, w_img, _) = scores.shape
    score_image_rgb = np.full((h_img, w_img, 3), (0, 0, 0)).astype(np.int8) # (h, w, rgb)
    for h in range(0, h_img):
        for w in range(0, w_img):
            r, g, b = get_heatmap_value(scores[h, w], offset, amplifier)
            score_image_rgb[h, w] = np.array([r, g, b])
    
    return score_image_rgb

def get_heatmap_value(value, offset=0.0, amplifier=1.0):
    '''
    Args:
        value: [0, 1]
        offset: [0, 1]
        
    Returns:
        r   [0, 255]
        g   [0, 255]
        b   [0, 255]
    '''
    value = (value * amplifier + offset) * 2.0
    r = int(max(0, (value - 1) * 255))
    r = min(255, r)
    g = int((1 - abs(value - 1)) * 255)
    g = min(255, g)
    b = int(max(0, (1 - value) * 255))
    b = min(255, b)
    
    return r, g, b

def get_heatmap_values(values:torch.Tensor, min=0.0, max=1.0):
    '''
    Args:
        values (n): [0, 1]
        offset: [0, 1]
        
    Returns:
        colors (n, 3): rgb range [0, 1]
    '''
    values[values<min] = min
    values[values>max] = max
    values = (values - min) / (max - min)
    
    values = values * 2.0
    
    r = values - 1
    r[r<0] = 0
    
    g = 1 - abs(values - 1)
    g[g > 1] = 1
    
    b = 1 - values
    b[b < 0] = 0
    
    colors = torch.cat((r[:, None], g[:, None], b[:, None]), dim=1)
    
    return colors

def get_camera(view_w_px, view_h_px, intrinsic, extrinsic, scale):
    '''
    Args:
        view_w_px: int
        view_h_px: int
        intrinsic: np (3, 3)
        extrinsic: np (4, 4) world to camera
        scale: float
    '''
    camera_lineset = o3d.geometry.LineSet.create_camera_visualization(
        view_w_px, 
        view_h_px,
        intrinsic,
        extrinsic,
        scale,)
    
    return camera_lineset

def get_unit_box(o=[-0.5, -0.5, -0.5]):
    spheres = o3d.geometry.TriangleMesh()
    sphere_o = o3d.geometry.TriangleMesh.create_sphere(radius=0.0001)
    sphere_o.translate([0, 0, 0])
    sphere_o.paint_uniform_color([0.5, 0.5, 0.5])
    sphere_x = o3d.geometry.TriangleMesh.create_sphere(radius=0.0001)
    sphere_x.translate([1, 0, 0])
    sphere_x.paint_uniform_color([1, 0, 0])
    sphere_y = o3d.geometry.TriangleMesh.create_sphere(radius=0.0001)
    sphere_y.translate([0, 1, 0])
    sphere_y.paint_uniform_color([0, 1, 0])
    sphere_z = o3d.geometry.TriangleMesh.create_sphere(radius=0.0001)
    sphere_z.translate([0, 0, 1])
    sphere_z.paint_uniform_color([0, 0, 1])
    spheres += sphere_o
    spheres += sphere_x
    spheres += sphere_y
    spheres += sphere_z

    cylinders = o3d.geometry.TriangleMesh()

    cylinder_x = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=1.0)
    R = cylinder_x.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
    cylinder_x.rotate(R, center=(0, 0, 0))
    cylinder_x.translate([0.5, 0, 0])
    cylinder_x.paint_uniform_color([1, 0, 0])

    cylinder_y = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=1.0)
    R = cylinder_y.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    cylinder_y.rotate(R, center=(0, 0, 0))
    cylinder_y.translate([0.0, 0.5, 0])
    cylinder_y.paint_uniform_color([0, 1, 0])

    cylinder_z = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=1.0)
    cylinder_z.translate([0, 0, 0.5])
    cylinder_z.paint_uniform_color([0, 0, 1])

    cylinders += cylinder_x
    cylinders += cylinder_y
    cylinders += cylinder_z

    box_lines = o3d.geometry.TriangleMesh()
    line_0 = copy.deepcopy(cylinder_x)
    line_0.translate([0, 1, 0])
    line_0.paint_uniform_color([0.5, 0.5, 0.5])
    line_1 = copy.deepcopy(cylinder_x)
    line_1.translate([0, 0, 1])
    line_1.paint_uniform_color([0.5, 0.5, 0.5])
    line_2 = copy.deepcopy(cylinder_x)
    line_2.translate([0, 1, 1])
    line_2.paint_uniform_color([0.5, 0.5, 0.5])
    line_3 = copy.deepcopy(cylinder_y)
    line_3.translate([1, 0, 0])
    line_3.paint_uniform_color([0.5, 0.5, 0.5])
    line_4 = copy.deepcopy(cylinder_y)
    line_4.translate([0, 0, 1])
    line_4.paint_uniform_color([0.5, 0.5, 0.5])
    line_5 = copy.deepcopy(cylinder_y)
    line_5.translate([1, 0, 1])
    line_5.paint_uniform_color([0.5, 0.5, 0.5])
    line_6 = copy.deepcopy(cylinder_z)
    line_6.translate([1, 0, 0])
    line_6.paint_uniform_color([0.5, 0.5, 0.5])
    line_7 = copy.deepcopy(cylinder_z)
    line_7.translate([0, 1, 0])
    line_7.paint_uniform_color([0.5, 0.5, 0.5])
    line_8 = copy.deepcopy(cylinder_z)
    line_8.translate([1, 1, 0])
    line_8.paint_uniform_color([0.5, 0.5, 0.5])

    box_lines += line_0
    box_lines += line_1
    box_lines += line_2
    box_lines += line_3
    box_lines += line_4
    box_lines += line_5
    box_lines += line_6
    box_lines += line_7
    box_lines += line_8

    spheres.translate(o)
    cylinders.translate(o)
    box_lines.translate(o)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[o[0]+0.5, o[1]+0.5, o[2]+0.5])

    return spheres, cylinders, box_lines, mesh_frame

def keypoints_to_spheres(keypoints):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.translate(keypoint)
        spheres += sphere
    return spheres

def visualize_kp_in_pcd(key_pcd, original_pcd):
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    key_spheres = keypoints_to_spheres(key_pcd)
    key_spheres.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([original_pcd, key_spheres])

def visualize_kp_only(key_pcd):
    key_spheres = keypoints_to_spheres(key_pcd)
    key_spheres.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([key_spheres])
    
# =================================
# inject vis
from common.utils import to_device, torch_to_np, np_to_torch
from analyze.analyzer import Analyzer
from analyze.imageview import ImageView
from analyze.pointcloudview import PointCloudView
from analyze.unitboxview import UnitBoxView
from analyze.coloredshpereview import ColoredSphereView
from analyze.cameraview import CameraView
from analyze.raysview import RaysView
from analyze.matchview2d3d import MatchView2D3D
from common.image import render_kp_on_img
from common.depth2Cloud import point3d_to_pixel_multiple_poses

def vis_pc(
    pc:torch.Tensor,        # (n, 3)
    ):
    (pc_np,) = torch_to_np((pc,))

    analyzer = Analyzer()
    unit_box = UnitBoxView()
    pc_view = PointCloudView()
    pc_view.cfg.color = [0.5, 0.5, 0.5]
    analyzer.add_view(unit_box)
    analyzer.add_view(pc_view)
    pc_view.set(pc_np)

    analyzer.run()

def vis_pc_kp(
    pc:torch.Tensor,        # (n, 3)
    kps3d,                  # (m', 3)
    ):
    pc_np, kps3d_np = torch_to_np((pc, kps3d))
    kp_sphere_size_max = 0.01
    
    analyzer = Analyzer()
    unit_box = UnitBoxView()
    pc_view = PointCloudView()
    pc_view.cfg.color = [0.5, 0.5, 0.5]
    analyzer.add_view(unit_box)
    analyzer.add_view(pc_view)
    pc_view.set(pc_np)
    
    sphere_view_kp = ColoredSphereView()
    sphere_view_kp.cfg.size_max = kp_sphere_size_max
    sphere_view_kp.cfg.size_min = 0
    sphere_view_kp.cfg.color_max = [1.0, 0.0, 0.0]
    analyzer.add_view(sphere_view_kp)
    sphere_view_kp.set(kps3d_np, np.ones(kps3d_np.shape[0]), np.ones(kps3d_np.shape[0]))

    analyzer.run()

def vis_pc_q(
    pc:torch.Tensor,        # (n, 3)
    queries:torch.Tensor,   # (m, 3)
    sal:torch.Tensor,       # (m)
    norm_sal:bool=True,
    multi_color:bool=True
    ):
    if norm_sal:
        sal = (sal-sal.min()) / (sal.max()-sal.min())
    if multi_color:
        colors = get_heatmap_values(sal.squeeze(0), min=0.3, max=0.7)
    else:
        colors = sal
    pc_np, queries_np, sal_np, colors_np = torch_to_np((pc, queries, sal, colors))
    
    kp_sphere_size_max = 0.005
    query_shpere_size_max = 0.004
    
    analyzer = Analyzer()
    unit_box = UnitBoxView()
    pc_view = PointCloudView()
    pc_view.cfg.color = [0.5, 0.5, 0.5]
    analyzer.add_view(unit_box)
    analyzer.add_view(pc_view)
    pc_view.set(pc_np)
    
    sphere_view_query = ColoredSphereView()
    sphere_view_query.cfg.size_max = query_shpere_size_max
    sphere_view_query.cfg.size_min = 0
    sphere_view_query.cfg.size_vis_thr = -0.1
    analyzer.add_view(sphere_view_query)
    if multi_color:
        sphere_view_query.set_with_color(queries_np, sal_np, colors_np)
    else:
        sphere_view_query.set(queries_np, sal_np, sal_np)

    analyzer.run()
    
def vis_pc_q_kp(
    pc:torch.Tensor,        # (n, 3)
    queries:torch.Tensor,   # (m, 3)
    sal:torch.Tensor,       # (m)
    kps3d,                  # (m', 3)
    norm_sal:bool=True,
    show_query:bool=True
    ):
    if norm_sal:
        sal = sal / sal.max()
    colors = get_heatmap_values(sal.squeeze(0), min=0, max=1.0)
    pc_np, queries_np, sal_np, colors_np, kps3d_np = torch_to_np((pc, queries, sal, colors, kps3d))
    
    kp_sphere_size_max = 0.005
    query_shpere_size_max = 0.002
    
    analyzer = Analyzer()
    unit_box = UnitBoxView()
    pc_view = PointCloudView()
    pc_view.cfg.color = [0.5, 0.5, 0.5]
    analyzer.add_view(unit_box)
    analyzer.add_view(pc_view)
    pc_view.set(pc_np)
    
    sphere_view_kp = ColoredSphereView()
    sphere_view_kp.cfg.size_max = kp_sphere_size_max
    sphere_view_kp.cfg.size_min = 0
    sphere_view_kp.cfg.color_max = [0.0, 0.0, 0.0]
    analyzer.add_view(sphere_view_kp)
    sphere_view_kp.set(kps3d_np, np.ones(kps3d_np.shape[0]), np.ones(kps3d_np.shape[0]))

    if show_query:
        sphere_view_query = ColoredSphereView()
        sphere_view_query.cfg.size_max = query_shpere_size_max
        sphere_view_query.cfg.size_min = 0
        sphere_view_query.cfg.size_vis_thr = -0.1
        analyzer.add_view(sphere_view_query)
        sphere_view_query.set_with_color(queries_np, sal_np, colors_np)

    analyzer.run()

def vis_pc_q_kp_cam(
    pc:torch.Tensor,        # (n, 3)
    queries:torch.Tensor,   # (m, 3)
    sal:torch.Tensor,       # (m)
    k:torch.Tensor,         # (4, 4)
    pose:torch.Tensor,      # (4, 4)
    image:torch.Tensor,      # (h, w, 3)
    norm_sal:bool=True
    ):
    if norm_sal:
        sal = sal / sal.max()
    colors = get_heatmap_values(sal.squeeze(0), min=0, max=1.0)
    pc_np, queries_np, sal_np, colors_np, k_np, pose_np, image_np = torch_to_np((pc, queries, sal, colors, k, pose, image))
    
    kp_sphere_size_max = 0.005
    query_shpere_size_max = 0.0025
    
    analyzer = Analyzer()
    unit_box = UnitBoxView()
    pc_view = PointCloudView()
    pc_view.cfg.color = [0.5, 0.5, 0.5]
    analyzer.add_view(unit_box)
    analyzer.add_view(pc_view)
    pc_view.set(pc_np)

    sphere_view_query = ColoredSphereView()
    sphere_view_query.cfg.size_max = query_shpere_size_max
    sphere_view_query.cfg.size_min = 0
    sphere_view_query.cfg.size_vis_thr = -0.1
    analyzer.add_view(sphere_view_query)
    sphere_view_query.set_with_color(queries_np, sal_np, colors_np)
    
    camera_view = CameraView()
    analyzer.add_view(camera_view)
    camera_view.set(
        pose_np,
        k_np, 
        image_np,
        scale=0.5
        )

    analyzer.run()
    
def vis_pc_cam(
    pc:torch.Tensor,        # (n, 3)
    k:torch.Tensor,         # (4, 4)
    pose:torch.Tensor,      # (4, 4)
    image:torch.Tensor,      # (h, w, 3)
    ):
    pc_np, k_np, pose_np, image_np = torch_to_np((pc, k, pose, image))
    
    kp_sphere_size_max = 0.005
    query_shpere_size_max = 0.0025
    
    analyzer = Analyzer()
    unit_box = UnitBoxView()
    pc_view = PointCloudView()
    pc_view.cfg.color = [0.5, 0.5, 0.5]
    analyzer.add_view(unit_box)
    analyzer.add_view(pc_view)
    pc_view.set(pc_np)
    
    camera_view = CameraView()
    analyzer.add_view(camera_view)
    camera_view.set(
        pose_np,
        k_np, 
        image_np,
        scale=0.5
        )

    analyzer.run()
    
def vis_pc_cams(
    pc:torch.Tensor,        # (n, 3)
    k:torch.Tensor,         # (4, 4)
    poses,      # list (4, 4)
    images,      # list (h, w, 3)
    ):
    cam_n = len(poses)
    pc_np, k_np = torch_to_np((pc, k))
    
    kp_sphere_size_max = 0.005
    query_shpere_size_max = 0.0025
    
    analyzer = Analyzer()
    unit_box = UnitBoxView()
    pc_view = PointCloudView()
    pc_view.cfg.color = [0.5, 0.5, 0.5]
    analyzer.add_view(unit_box)
    analyzer.add_view(pc_view)
    pc_view.set(pc_np)
    
    for i in range(0, cam_n):
        pose_np, image_np = torch_to_np((poses[i], images[i]))
        camera_view = CameraView()
        if i == 4:
            camera_view.cfg.color = [1.0, 0.0, 0.0]
        analyzer.add_view(camera_view)
        camera_view.set(
            pose_np,
            k_np, 
            image_np,
            scale=0.2
            )

    analyzer.run()

def vis_proj_err(
    proj_kps3d:torch.Tensor,    # (m, 2)
    kps2d:torch.Tensor,         # (m, 2)
    image:torch.Tensor,         # (h, w, 3) [0, 255]
    ):
    
    # =======
    kp_sphere_size_max = 0.005
    # =======
    
    proj_kps3d_np, kps2d_np, image_np = torch_to_np((proj_kps3d, kps2d, image))
    
    analyzer = Analyzer()
    unit_box = UnitBoxView()
    image_view = ImageView()
    analyzer.add_view(unit_box)
    analyzer.add_view(image_view, 'image')
    
    image_np = render_kp_on_img(image_np, kps2d_np, color=(255, 0, 0, 100), kp_size=1)
    image_np = render_kp_on_img(image_np, proj_kps3d_np, color=(255, 255, 0, 100), kp_size=1)
    image_view.set(image_np, 0, 1)
    
    analyzer.run()
    
def vis_desc_match_sphere(
    pc:torch.Tensor,
    kps3d:torch.Tensor,
    descs:torch.Tensor,
    center:torch.Tensor,    
    match_radius:float,
    sample1_idx:int=-1,
    sample2_idx:int=-1,
    focus_idx:int=-1
    ):
    '''
    Args:
        pc:     (n, 3)
        kps3d   (n_kps, 3)
        descs   (n_kps, dim)
        center  (3)
        match_radius:
    '''
    kp_sphere_size_max = 0.001
    upscale = 0.00001
    
    # find kps in the sphere
    mask = torch.linalg.norm(kps3d - center[None, :], dim=1) < match_radius # (n')
    kps3d_masked = kps3d[mask]
    descs_masked = descs[mask]
    
    pc_np, kps3d_np, kps3d_masked_np, center_np = torch_to_np((pc, kps3d, kps3d_masked, center.unsqueeze(0)))
    
    analyzer = Analyzer()
    unit_box = UnitBoxView()
    pc_view = PointCloudView()
    pc_view.cfg.color = [0.5, 0.5, 0.5]
    analyzer.add_view(unit_box)
    analyzer.add_view(pc_view)
    pc_view.set(pc_np)
    
    sphere_view_center = ColoredSphereView()
    sphere_view_center.cfg.size_max = kp_sphere_size_max+20*upscale
    sphere_view_center.cfg.size_min = 0
    sphere_view_center.cfg.color_max = [0.0, 1.0, 1.0]
    analyzer.add_view(sphere_view_center)
    sphere_view_center.set(center_np, np.ones(center_np.shape[0]), np.ones(center_np.shape[0]))
    
    if kps3d_masked.shape[0] == 0:
        print('No kps detected..')
        analyzer.run()
    
    cos = nn.CosineSimilarity(dim=2)
    match_mat = cos(descs_masked[:, None, :], descs_masked[None, :, :]) # (n', n')
    print(match_mat)

    sphere_view_kp = ColoredSphereView()
    sphere_view_kp.cfg.size_max = kp_sphere_size_max
    sphere_view_kp.cfg.size_min = 0
    sphere_view_kp.cfg.color_max = [0.0, 0.0, 0.0]
    analyzer.add_view(sphere_view_kp)
    sphere_view_kp.set(kps3d_np, np.ones(kps3d_np.shape[0]), np.ones(kps3d_np.shape[0]))
    
    sphere_view_kp_masked = ColoredSphereView()
    sphere_view_kp_masked.cfg.size_max = kp_sphere_size_max+2*upscale
    sphere_view_kp_masked.cfg.size_min = 0
    sphere_view_kp_masked.cfg.color_max = [0.0, 1.0, 0.0]
    analyzer.add_view(sphere_view_kp_masked)
    sphere_view_kp_masked.set(kps3d_masked_np, np.ones(kps3d_masked_np.shape[0]), np.ones(kps3d_masked_np.shape[0]))
    
    if sample1_idx >= 0:
        sphere_view_kp1 = ColoredSphereView()
        sphere_view_kp1.cfg.size_max = kp_sphere_size_max+3*upscale
        sphere_view_kp1.cfg.size_min = 0
        sphere_view_kp1.cfg.color_max = [0.0, 0.0, 1.0]
        analyzer.add_view(sphere_view_kp1)
        sphere_view_kp1.set(np.expand_dims(kps3d_masked_np[sample1_idx], axis=0), np.ones(1), np.ones(1))
        
    if sample2_idx >= 0:
        sphere_view_kp2 = ColoredSphereView()
        sphere_view_kp2.cfg.size_max = kp_sphere_size_max+3*upscale
        sphere_view_kp2.cfg.size_min = 0
        sphere_view_kp2.cfg.color_max = [1.0, 1.0, 0.0]
        analyzer.add_view(sphere_view_kp2)
        sphere_view_kp2.set(np.expand_dims(kps3d_masked_np[sample2_idx], axis=0), np.ones(1), np.ones(1))
        
    if focus_idx >=0:
        colors_np = get_heatmap_values(match_mat[focus_idx]).cpu().detach().numpy()
        sphere_view_match = ColoredSphereView()
        sphere_view_match.cfg.size_max = kp_sphere_size_max+4*upscale
        sphere_view_match.cfg.size_min = 0
        analyzer.add_view(sphere_view_match)
        sphere_view_match.set_with_color(kps3d_masked_np, np.ones(kps3d_masked_np.shape[0]), colors_np)
        
        sphere_view_focus = ColoredSphereView()
        sphere_view_focus.cfg.size_max = kp_sphere_size_max+5*upscale
        sphere_view_focus.cfg.size_min = 0
        sphere_view_focus.cfg.color_max = [1.0, 0.0, 1.0]
        analyzer.add_view(sphere_view_focus)
        sphere_view_focus.set(np.expand_dims(kps3d_masked_np[focus_idx], axis=0), np.ones(1), np.ones(1))
    
    analyzer.run()
    
def vis_2d3d_match(
    pc:torch.Tensor,
    kps3d:torch.Tensor,
    image:torch.Tensor,
    kps2d:torch.Tensor,
    correspondences:torch.Tensor,
):
    pc_np, image_np, kps3d_np, kps2d_np, correspondences_np = torch_to_np((pc,
                                                                           image,
                                                                           kps3d,
                                                                           kps2d,
                                                                           correspondences))
    
    analyzer = Analyzer()
    matchview2d3d = MatchView2D3D()
    analyzer.add_view(matchview2d3d)
    matchview2d3d.set(pc_np, image_np, kps3d_np, kps2d_np, correspondences_np)
    analyzer.run()

if __name__ == "__main__":
    np_pts = 1
#    np_pts = ReadPointCloudFromFile_ModelNet40("/home/han/Projects/Datasets/modelnet40_normal_resampled/airplane/airplane_0001.txt")
 #   choice = np.random.choice(len(np_pts[0]), 50, replace=True) # replace = True?
 #   np_key_pts = np_pts[0:3, choice]
#    Visualize(np_pts, np_key_pts)