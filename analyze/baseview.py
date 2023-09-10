import sys
import os
project_root = os.path.abspath('.')
sys.path.append(project_root)

from dataclasses import dataclass
from common.config import BaseConfig

from typing import Tuple
import torch
import torch.nn as nn

import open3d as o3d

@dataclass
class ConfigBaseView(BaseConfig):
    name:str
    
class BaseView():   
    def __init__(self, cfg:ConfigBaseView):
        '''
        Setup contents to update
        
        Args:
            cfg
        '''
        self.cfg = cfg
        
    def setup(self, vis):
        '''
        setup vis and other geometries 
        Args:
            vis: o3d.visualization.Visualizer()
        '''
        self.vis = vis
        
    def update(self):
        '''
        update in self.vis_3d and plot
        '''
        raise NotImplementedError