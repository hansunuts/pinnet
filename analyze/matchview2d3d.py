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
from common.image import render_kp_on_img

@dataclass
class ConfigMatchView2D3D(ConfigBaseView):
    name:str='pointcloud'
    dist:float=1.0
    show_unit_cubes:bool=True
    show_gizmos:bool=True
    show_mismatch:bool=True
    kp_pc_size:float=0.005
    pc_color:list=field(default_factory=lambda:[0.5, 0.5, 0.5])
    kp_pc_color:list=field(default_factory=lambda:[1.0, 0.9, 0.0])
    kp_img_color:list=field(default_factory=lambda:[0.0, 0.0, 1.0])
    correct_line_color:list=field(default_factory=lambda:[0.1, 1.0, 0.0])
    incorrect_line_color:list=field(default_factory=lambda:[1.0, 0.1, 0.0])
    
class MatchView2D3D(BaseView):   
    def __init__(self, cfg:ConfigMatchView2D3D=None):
        super().__init__(cfg)
        if cfg == None:
            cfg = ConfigMatchView2D3D()
        self.cfg = cfg
    
    def setup(self, vis):
        super().setup(vis)        
        spheres, cylinders, box_lines, mesh_frame = get_unit_box()
        self.vis.add_geometry(spheres)
        #self.vis.add_geometry(cylinders)
        #self.vis.add_geometry(box_lines)
        #self.vis.add_geometry(mesh_frame)

        self.pc_o3d = o3d.geometry.PointCloud()
        self.lines = o3d.geometry.LineSet()
        self.kp_pc = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.kp_pc)
        self.vis.add_geometry(self.lines)
        self.vis.add_geometry(self.pc_o3d)
        
        self.image_mesh = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.image_mesh)
        
    def update(self):
        if self.dirty:
            self.dirty = False
            self.vis.update_geometry(self.kp_pc)
            self.vis.update_geometry(self.lines)
            self.vis.update_geometry(self.pc_o3d)
            self.vis.update_geometry(self.image_mesh)

    def set(
        self, 
        pc:np.ndarray, 
        image:np.ndarray, 
        kps3d:np.ndarray,
        kps2d:np.ndarray,
        correspondences:np.ndarray,
        ):
        '''
        Args:
            pc:                 (n, 3)
            image:              (h, w, 3)
            kp3d:               (n_kp3d, 3)
            kp2d:               (n_kp2d, 2)
            correspondences:    (n_corr, 3) kp3d idx, kp2d idx, is correct(0/1).
        '''
        self.dirty = True
        self.pc = pc
        self.pc_o3d.points = o3d.utility.Vector3dVector(self.pc)
        self.pc_o3d.paint_uniform_color(self.cfg.pc_color)
        
        H = image.shape[0]
        W = image.shape[1]
        
        if not self.cfg.show_mismatch:
            correspondences = correspondences[correspondences[:, 2] == 1]
        
        for index, point in enumerate(kps3d):
            sphere = o3d.geometry.TriangleMesh.create_sphere(self.cfg.kp_pc_size)
            sphere.translate([point[0], point[1], point[2]])
            sphere.paint_uniform_color(self.cfg.kp_pc_color)
            self.kp_pc += sphere
            
        image = render_kp_on_img(image, kps2d, kp_size=1)
        
        # image    
        vertex_array = np.zeros(shape=(4, 3))
        face_array = np.zeros(shape=(2, 3))
        uv_array = np.zeros(shape=(6, 2))
        
        '''
        vertex_array[0] = [0.5,         0.5,        -0.5]
        vertex_array[1] = [0.5+0.64,    0.5,        -0.5]
        vertex_array[2] = [0.5+0.64,    0.5-0.48,  -0.5]
        vertex_array[3] = [0.5,         0.5-0.48,  -0.5]
        '''
        
        vertex_array[0] = [0.5,         -0.1,        0.5-0.48*2,   ]
        vertex_array[1] = [0.5+0.64*2,  -0.1,        0.5-0.48*2,   ]
        vertex_array[2] = [0.5+0.64*2,  -0.1,        0.5,        ]
        vertex_array[3] = [0.5,         -0.1,        0.5,        ]

        face_array[0] = [0, 3, 1]
        face_array[1] = [1, 3, 2]

        
        uv_array[0] = [0, 1] 
        uv_array[1] = [0, 0]
        uv_array[2] = [1, 1]
        uv_array[3] = [1, 1] 
        uv_array[4] = [0, 0]
        uv_array[5] = [1, 0]
        
        self.image_mesh.clear()
        self.image_mesh.vertices = o3d.utility.Vector3dVector(vertex_array)
        self.image_mesh.triangles = o3d.utility.Vector3iVector(face_array)
        
        img = o3d.geometry.Image(image)
        self.image_mesh.textures = [img]
        self.image_mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_array)
        self.image_mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
        
        # correspondence lines
        points = np.zeros((0, 3))
        colors = np.zeros((0, 3))
        for correspondence in correspondences:
            kp3d = np.expand_dims(kps3d[correspondence[0]], axis=0)
            kp2d = np.expand_dims(np.hstack([kps2d[correspondence[1]]*2, -0.1]), axis=0) # (1, 3)
            kp2d[:, [2, 1]] = kp2d[:, [1, 2]]
            kp2d[0, 0] = kp2d[0, 0] / 1000 + 0.5
            kp2d[0, 2] = 0.5 - kp2d[0, 2] / 1000
            points = np.concatenate((points, kp3d[:, :3], kp2d[:, :3]), axis=0) # (2*n_lines, 3)
            if correspondence[2] == 0:
                colors = np.concatenate((colors, np.expand_dims(self.cfg.incorrect_line_color, axis=0)), axis=0) # (2*n_lines, 3)
            else:
                colors = np.concatenate((colors, np.expand_dims(self.cfg.correct_line_color, axis=0)), axis=0) # (2*n_lines, 3)
            
        self.lines.points = o3d.utility.Vector3dVector(points)
        n_lines = len(correspondences)
        lines = np.ndarray((n_lines, 2))
        for idx, line in enumerate(lines):
            line[0] = 2*idx
            line[1] = 2*idx + 1
        self.lines.lines = o3d.utility.Vector2iVector(lines) 
        self.lines.colors = o3d.utility.Vector3dVector(colors)