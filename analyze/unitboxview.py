import sys
import os
project_root = os.path.abspath('.')
sys.path.append(project_root)

from dataclasses import dataclass
import open3d as o3d

from analyze.baseview import ConfigBaseView, BaseView
from common.visualization import get_unit_box

@dataclass
class ConfigUnitBoxView(ConfigBaseView):
    name:str='unitbox'
    
class UnitBoxView(BaseView):   
    def __init__(self, cfg:ConfigUnitBoxView=None):
        super().__init__(cfg)
        if cfg == None:
            cfg = ConfigUnitBoxView()
        self.cfg = cfg
    
    def setup(self, vis):
        super().setup(vis)
        self.pc_o3d = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pc_o3d)
        
        spheres, cylinders, box_lines, mesh_frame = get_unit_box()
        spheres.scale(1, center=spheres.get_center())
        self.vis.add_geometry(spheres)
        
        '''
        self.vis.add_geometry(cylinders)
        self.vis.add_geometry(box_lines)
        self.vis.add_geometry(mesh_frame)
        '''
        
            
    def update(self):
        pass