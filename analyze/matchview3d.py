import sys
import os
project_root = os.path.abspath('.')
sys.path.append(project_root)

from dataclasses import dataclass
from dataclasses import field
import numpy as np
import open3d as o3d

from analyze.baseview import ConfigBaseView, BaseView
from common.visualization import get_unit_box

@dataclass
class ConfigMatchView3D(ConfigBaseView):
    name:str='pointcloud'
    dist:float=1.0
    show_unit_cubes:bool=True
    show_gizmos:bool=True
    kp_size:float=0.005
    pc1_color:list=field(default_factory=lambda:[0.2, 1.0, 0.0])
    pc2_color:list=field(default_factory=lambda:[0.3, 0.5, 0.0])
    kp1_color:list=field(default_factory=lambda:[1.0, 0.0, 0.0])
    kp2_color:list=field(default_factory=lambda:[0.0, 0.0, 1.0])
    line_color:list=field(default_factory=lambda:[0.0, 0.0, 0.0])
    
class MatchView3D(BaseView):   
    def __init__(self, cfg:ConfigMatchView3D=None):
        super().__init__(cfg)
        if cfg == None:
            cfg = ConfigMatchView3D()
        self.cfg = cfg
    
    def setup(self, vis):
        super().setup(vis)        
        spheres, cylinders, box_lines, mesh_frame = get_unit_box(o=[-0.5, -self.cfg.dist/2-0.5, -0.5])
        self.vis.add_geometry(spheres)
        self.vis.add_geometry(cylinders)
        self.vis.add_geometry(box_lines)
        self.vis.add_geometry(mesh_frame)
        
        spheres, cylinders, box_lines, mesh_frame = get_unit_box(o=[-0.5, self.cfg.dist/2-0.5, -0.5])
        self.vis.add_geometry(spheres)
        self.vis.add_geometry(cylinders)
        self.vis.add_geometry(box_lines)
        self.vis.add_geometry(mesh_frame)

        self.pc_o3d_1 = o3d.geometry.PointCloud()
        self.pc_o3d_2 = o3d.geometry.PointCloud()
        self.lines = o3d.geometry.LineSet()
        self.kp1 = o3d.geometry.TriangleMesh()
        self.kp2 = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.kp1)
        self.vis.add_geometry(self.kp2)
        self.vis.add_geometry(self.lines)
        self.vis.add_geometry(self.pc_o3d_1)
        self.vis.add_geometry(self.pc_o3d_2)
        
    def update(self):
        if self.dirty:
            self.dirty = False
            self.vis.update_geometry(self.kp1)
            self.vis.update_geometry(self.kp2)
            self.vis.update_geometry(self.lines)
            self.vis.update_geometry(self.pc_o3d_1)
            self.vis.update_geometry(self.pc_o3d_2)

    def set(
        self, 
        pc1:np.ndarray, 
        pc2:np.ndarray, 
        kp1:np.ndarray,
        kp2:np.ndarray,
        correspondences:np.ndarray,
        ):
        '''
        Args:
            pc1:            (n, 3)
            pc2:            (n, 3)
            kp1:            (n_kp1, 3)
            kp2:            (n_kp2, 3)
            correspondences: [{'kp1':kp1, 'desc1':desc1, 'kp2':kp2, 'desc2':desc2}, ...] all elements are np.ndarray
        '''
        self.dirty = True
        self.pc1 = pc1
        self.pc1[:, 1]  = pc1[:, 1] - self.cfg.dist / 2
        self.pc2 = pc2
        self.pc2[:, 1]  = pc2[:, 1] + self.cfg.dist / 2
        self.pc_o3d_1.points = o3d.utility.Vector3dVector(self.pc1)
        self.pc_o3d_1.paint_uniform_color(self.cfg.pc1_color)
        self.pc_o3d_2.points = o3d.utility.Vector3dVector(self.pc2)
        self.pc_o3d_2.paint_uniform_color(self.cfg.pc2_color)
        
        for index, point in enumerate(kp1):
            sphere = o3d.geometry.TriangleMesh.create_sphere(self.cfg.kp_size)
            sphere.translate([point[0], point[1]-self.cfg.dist / 2, point[2]])
            sphere.paint_uniform_color(self.cfg.kp1_color)
            self.kp1 += sphere
            
        for index, point in enumerate(kp2):
            sphere = o3d.geometry.TriangleMesh.create_sphere(self.cfg.kp_size)
            sphere.translate([point[0], point[1]+self.cfg.dist / 2, point[2]])
            sphere.paint_uniform_color(self.cfg.kp2_color)
            self.kp2 += sphere
        
        points = np.zeros((0, 3))
        for correspondence in correspondences:
            kp1 = np.expand_dims(correspondence['kp1'], axis=0)
            kp2 = np.expand_dims(correspondence['kp2'], axis=0)
            kp1[0, 1] = kp1[0, 1] - self.cfg.dist / 2
            kp2[0, 1] = kp2[0, 1] + self.cfg.dist / 2
            points = np.concatenate((points, kp1[:, :3], kp2[:, :3]), axis=0) # (2*n_lines, 3)
            
        self.lines.points = o3d.utility.Vector3dVector(points)
        n_lines = len(correspondences)
        lines = np.ndarray((n_lines, 2))
        for idx, line in enumerate(lines):
            line[0] = 2*idx
            line[1] = 2*idx + 1
        self.lines.lines = o3d.utility.Vector2iVector(lines) 
        colors = np.array([self.cfg.line_color])
        colors = np.repeat(colors, n_lines, axis=0)
        self.lines.colors = o3d.utility.Vector3dVector(colors)