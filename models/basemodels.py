import sys
import os
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.append(project_root)

from dataclasses import dataclass
from common.config import BaseConfig

from typing import Tuple
import torch
import torch.nn as nn

@dataclass
class ConfigBaseModel(BaseConfig):
    pass

@dataclass
class ConfigBaseModel2D(BaseConfig):
    desc_feat:int=128
    
@dataclass
class ConfigBaseModel3D(ConfigBaseModel):
    dim_desc:int=128

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def load_weight(self, weights_path:str):
        '''
        path: pth path
        '''
        checkpoint = torch.load(weights_path)
        self.load_state_dict(checkpoint['state_dict'])

class BaseModel2D(BaseModel):   
    def __init__(self):
        super(BaseModel2D, self).__init__()
    
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            x (b, h, w, 3)  batch, height, width, rgb
            
        Returns:
            det_score (b, h, w) detection score of each input pixel, score range [0, 1]
            desc (b, h, w, dim_desc) descriptor of each input pixel
        '''
        raise NotImplementedError
        
class BaseModel3D(BaseModel):
    def __init__(self) -> None:
        super(BaseModel3D, self).__init__()
    
    def forward(self, 
                pcd:torch.Tensor, 
                query:torch.Tensor
                ) -> Tuple[
                    torch.Tensor, 
                    torch.Tensor,
                    torch.Tensor]:
        '''
        Args:
            pcd (b, n, 3)  batch, point num, position. Target point cloud.
            query (b, q_n, 3) batch, query num, query position. query points in the space of target point.
            
        Returns:
            occ (b, q_n) detection score of each input pixel, score range [0, 1]
            sal (b, q_n)
            desc (b, q_n, dim_desc) descriptor of each input pixel
        '''
        raise NotImplementedError 