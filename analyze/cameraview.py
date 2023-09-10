import sys
import os
project_root = os.path.abspath('.')
sys.path.append(project_root)

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import open3d as o3d
import matplotlib.image as mpimg

from analyze.baseview import ConfigBaseView, BaseView
from common.visualization import get_camera

@dataclass
class ConfigCameraView(ConfigBaseView):
    name:str='camera'
    color:list=field(default_factory=lambda:[0.0, 0.0, 0.0])
    
class CameraView(BaseView):   
    def __init__(self, cfg:ConfigCameraView=None):
        super().__init__(cfg)
        if cfg == None:
            cfg = ConfigCameraView()
        self.cfg = cfg
    
    def setup(self, vis):
        super().setup(vis)
        
        self.mesh_proj_plane = o3d.geometry.TriangleMesh()
        self.camera_lineset = o3d.geometry.LineSet()
        self.vis.add_geometry(self.camera_lineset)
        self.vis.add_geometry(self.mesh_proj_plane)
        
    def update(self):
        if self.dirty:
            self.dirty = False
            self.vis.update_geometry(self.camera_lineset)
            self.vis.update_geometry(self.mesh_proj_plane)

    def set(self, pose:np.ndarray, k:np.ndarray, image:np.ndarray, scale:float=1.0):
        '''
        Args:
            pose:   (4, 4) c2w, need to inverse 
            k:      (4, 4) intrinsic matrix
            view_w  resolution w of the projection image
            view_h  resolution h of the projection image
            image   (h, w, 3) needs to be int8
            scale   scale of the camera lineset. scale = 1.0 shows the normalized projection plane.
        '''
        self.dirty = True
        k = k[0:3, 0:3]
        pose_w2c = np.linalg.inv(pose)
        self.camera_lineset.clear()
        self.camera_lineset += get_camera(image.shape[1], image.shape[0], k, pose_w2c, scale)
        self.camera_lineset.paint_uniform_color(self.cfg.color)
        
        vertex_array = np.zeros(shape=(4, 3))
        face_array = np.zeros(shape=(2, 3))
        uv_array = np.zeros(shape=(6, 2))

        vertex_array[0] = self.camera_lineset.get_line_coordinate(0)[1]
        vertex_array[1] = self.camera_lineset.get_line_coordinate(1)[1]
        vertex_array[2] = self.camera_lineset.get_line_coordinate(2)[1]
        vertex_array[3] = self.camera_lineset.get_line_coordinate(3)[1]

        face_array[0] = [0, 3, 1]
        face_array[1] = [1, 3, 2]

        uv_array[0] = [0, 0] 
        uv_array[1] = [0, 1]
        uv_array[2] = [1, 0]
        uv_array[3] = [1, 0] 
        uv_array[4] = [0, 1]
        uv_array[5] = [1, 1]
        
        self.mesh_proj_plane.clear()
        self.mesh_proj_plane.vertices = o3d.utility.Vector3dVector(vertex_array)
        self.mesh_proj_plane.triangles = o3d.utility.Vector3iVector(face_array)
        
        img = o3d.geometry.Image(image)
        self.mesh_proj_plane.textures = [img]
        self.mesh_proj_plane.triangle_uvs = o3d.utility.Vector2dVector(uv_array)
        self.mesh_proj_plane.triangle_material_ids = o3d.utility.IntVector([0, 0])

