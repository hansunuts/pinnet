import sys
import os
project_root = os.path.abspath('.')
sys.path.append(project_root)

from dataclasses import dataclass
from dataclasses import field
import numpy as np
import open3d as o3d

from analyze.baseview import ConfigBaseView, BaseView

@dataclass
class ConfigRaysView(ConfigBaseView):
    name:str='raysview'
    ray_max_dist:float=2.0
    proj_point:bool=True
    
class RaysView(BaseView):   
    def __init__(self, cfg:ConfigRaysView=None):
        super().__init__(cfg)
        if cfg == None:
            cfg = ConfigRaysView()
        self.cfg = cfg
    
    def setup(self, vis):
        super().setup(vis)
        self.rays = o3d.geometry.LineSet()
        self.vis.add_geometry(self.rays)
        
    def update(self):
        if self.dirty:
            self.dirty = False
            self.vis.update_geometry(self.rays)

    def set(self, origins:np.ndarray, dirs:np.ndarray, colors:np.ndarray=np.zeros((0, 3)), use_dir_length:bool=False):
        '''
        Args:
            origins:    (n_lines, 3)
            dirs:       (n_lines, 3)
            colors:     (n_lines, 3) [0, 1]
        '''
        self.dirty = True
        self.rays.clear()
        
        if use_dir_length:
            endpoints = origins + dirs
        else:
            endpoints = origins + dirs * self.cfg.ray_max_dist
        
        points = np.concatenate((origins, endpoints), axis=0) # (2*n_lines, 3)
        self.rays.points = o3d.utility.Vector3dVector(points)
        
        n_lines = origins.shape[0]
        lines = np.ndarray((n_lines, 2))
        for idx, line in enumerate(lines):
            line[0] = idx
            line[1] = idx + n_lines
        self.rays.lines = o3d.utility.Vector2iVector(lines) 
        
        if colors.shape[0] == 0:
            colors = np.random.rand(n_lines, 3)
        self.rays.colors = o3d.utility.Vector3dVector(colors)