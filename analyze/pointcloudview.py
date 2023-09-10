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
class ConfigPointCloudView(ConfigBaseView):
    name:str='pointcloud'
    color:list=field(default_factory=lambda:[0.2, 1.0, 0.0])
    
class PointCloudView(BaseView):   
    def __init__(self, cfg:ConfigPointCloudView=None):
        super().__init__(cfg)
        if cfg == None:
            cfg = ConfigPointCloudView()
        self.cfg = cfg
    
    def setup(self, vis):
        super().setup(vis)
        self.pc_o3d = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pc_o3d)
        
    def update(self):
        if self.dirty:
            self.dirty = False
            self.vis.update_geometry(self.pc_o3d)

    def set(self, pc:np.ndarray):
        self.dirty = True
        self.pc = pc
        self.pc_o3d.points = o3d.utility.Vector3dVector(self.pc)
        self.pc_o3d.paint_uniform_color(self.cfg.color)