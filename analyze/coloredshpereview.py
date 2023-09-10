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
class ConfigColoredSphereView(ConfigBaseView):
    name:str='coloredshpere'
    size_vis_thr:float=0.0 # sphere size [0, 1]
    size_min:float=0.003
    size_max:float=0.01
    color_vis_thr:float=0.0 # color [0, 1]
    color_min:list=field(default_factory=lambda:[0.0, 0.0, 0.0])
    color_max:list=field(default_factory=lambda:[1.0, 0.0, 0.0])
    
class ColoredSphereView(BaseView):   
    '''
    Privide visualization of two [0, 1] values
    '''
    def __init__(self, cfg:ConfigColoredSphereView=None):
        super().__init__(cfg)
        if cfg == None:
            cfg = ConfigColoredSphereView()
        self.cfg = cfg
    
    def setup(self, vis):
        super().setup(vis)
        self.shperes = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.shperes)
        
    def update(self):
        if self.dirty:
            self.dirty = False
            self.vis.update_geometry(self.shperes)
            
    def set_with_color(self, pts:np.ndarray, sizes:np.ndarray, colors:np.ndarray):
        '''
        Args:
            pts:    (n, 3)
            size:   (n) [0, 1] for shpere size
            color:  (n, 3) color of each sphere, color in rgb, range [0, 1]
        '''
        self.dirty = True
        self.shperes.clear()
        
        for index, point in enumerate(pts):
            size = sizes[index] # (1)
            color = colors[index] # (3)
            if size > self.cfg.size_vis_thr:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.cfg.size_max)
                sphere.translate([point[0], point[1], point[2]])
                sphere.paint_uniform_color([color[0], color[1], color[2]])
                self.shperes += sphere
            
    def set(self, pts:np.ndarray, vals1:np.ndarray, vals2:np.ndarray):
        '''
        Args:
            pts:    (n, 3)
            vals1:  (n) [0, 1] for shpere size
            vals2:  (n) [0, 1] for shpere color
        '''
        self.dirty = True
        self.shperes.clear()
        
        for index, point in enumerate(pts):
            val1 = vals1[index]
            val2 = vals2[index]
            if val1 > self.cfg.size_vis_thr:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.cfg.size_min + self.cfg.size_max * (val1-self.cfg.size_vis_thr)/(1-self.cfg.size_vis_thr))
                sphere.translate([point[0], point[1], point[2]])
                if val2 < self.cfg.color_vis_thr:
                    val2 = 0
                else:
                    val2 = val2-self.cfg.color_vis_thr
                color_factor = val2/(1-self.cfg.color_vis_thr)
                sphere.paint_uniform_color([color_factor*self.cfg.color_max[0], color_factor*self.cfg.color_max[1], color_factor*self.cfg.color_max[2]])
                self.shperes += sphere