import sys
import os
project_root = os.path.abspath('.')
sys.path.append(project_root)
print(project_root)

from math import *
from matplotlib import pyplot as plt
import numpy as np
print(plt.get_backend())

from functools import partial

from dataclasses import dataclass
from common.config import BaseConfig

import open3d as o3d
import torch
import torch.nn as nn

from analyze.baseview import ConfigBaseView, BaseView

@dataclass 
class ConfigAnalyzer(BaseConfig):
    name:str='analyzer'
    line_width:float=3.0

class Analyzer():
    def __init__(self, cfg:ConfigAnalyzer=None) -> None:
        if cfg == None:
            cfg = ConfigAnalyzer()
        self.views = []
        self.cfg = cfg
        
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name='world')
        vis.get_render_option().load_from_json('analyze/render_option.json')
        self.vis_dict = {}
        self.vis_dict['world'] = vis
        
        vis.register_key_callback(ord("Q"), partial(self.close))
        self.activate = True
        
    def close(self, vis):
        self.activate = False
        vis.close()
        
    def add_view(self, view:BaseView, vis_name='world', width:int=1920, height:int=1080):
        if vis_name not in self.vis_dict.keys():
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name=vis_name, width=width, height=height)
            self.vis_dict[vis_name] = vis
            
        view.setup(self.vis_dict[vis_name])
        self.views.append(view)
        
    def get_view(self, vis_name:str) -> BaseView:
        if vis_name not in self.vis_dict.keys():
            return None
        else:
            return self.vis_dict[vis_name]
        
    def update(self):
        for view in self.views:
            view.update()
            
        for vis in self.vis_dict.values():
            vis.poll_events()
            vis.update_renderer()
        
    def run(self, pre_update:callable=None, args=None):
        while self.activate:
            if pre_update != None:
                if args != None:
                    pre_update(*args)
                else:
                    pre_update()
            self.update()
            
        
        
        
    




