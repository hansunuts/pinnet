from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist

from dataclasses import dataclass
from common.config import BaseConfig
from models.basemodels import ConfigBaseModel3D, BaseModel3D
from models.layers.pointnet import LocalPoolPointnet
from models.layers.decoder3d import LocalDecoder

@dataclass
class ConvONetConfig():
    name:str='convonet'
    encoder_input_dim:int=3
    encoder_hidden_dim:int=256
    encoder_output_dim:int=128
    scatter_type:str='max'
    unet:bool=False
    unet_3d:bool=True
    plane_resolution:int=32
    grid_resolution:int=32
    plane_type:str='grid' #xy, xz, yz, grid
    padding:float=0.1
    n_blocks:int=5
    
@dataclass
class ConfigPinNetDet(ConfigBaseModel3D):
    name:str='pinnet_det'
    convonet_cfg:ConvONetConfig=ConvONetConfig()
    decoder_sal_out_dim:int=1
    decoder_desc_out_dim:int=256*3
    decoder_hidden_dim:int=256
    decoder_block_n:int=5
    decoder_leaky:bool=False
    decoder_desc_type:str='field' # field, occp
    deocder_sigmoid:bool=True
    
class PinNetDet(BaseModel3D):
    def __init__(self, exp_name:str):
        super(PinNetDet, self).__init__()
        self.cfg = ConfigPinNetDet().load_from_exp(exp_name)
        self.encoder = LocalPoolPointnet(
                                c_dim=self.cfg.convonet_cfg.encoder_output_dim, 
                                dim=self.cfg.convonet_cfg.encoder_input_dim,
                                hidden_dim=self.cfg.convonet_cfg.encoder_hidden_dim,
                                scatter_type=self.cfg.convonet_cfg.scatter_type,
                                unet=self.cfg.convonet_cfg.unet,
                                unet3d=self.cfg.convonet_cfg.unet_3d,
                                plane_resolution=self.cfg.convonet_cfg.plane_resolution,
                                grid_resolution=self.cfg.convonet_cfg.grid_resolution,
                                plane_type=self.cfg.convonet_cfg.plane_type,
                                padding=self.cfg.convonet_cfg.padding,
                                n_blocks=self.cfg.convonet_cfg.n_blocks
                                ) # [b, x, y, z, feat]
        self.sal_decoder = LocalDecoder(dim=3,                                          # point coord dim
                                        c_dim=self.cfg.convonet_cfg.encoder_output_dim,  # grid feat dim
                                        out_dim=self.cfg.decoder_sal_out_dim,                   
                                        hidden_size=self.cfg.decoder_hidden_dim, 
                                        n_blocks=self.cfg.decoder_block_n, 
                                        leaky=self.cfg.decoder_leaky,)
        
    def forward(
        self, 
        pc: torch.Tensor, 
        query: torch.Tensor
        ) -> Tuple[
            torch.Tensor, 
            torch.Tensor]:
        '''
        Args:
            pc:         (b, n, 3) point cloud
            query:      (b, n_q, 3) query points location (3), view origin (3), view dir (3)
            
        Returns:
            sal:    (b, n_q) salience
        '''        
        grid_feat = self.encoder(pc)
        sal, sample_feat = self.decode_saliency(query, grid_feat)

        return sal 

    def decode_saliency(self, query, grid_feat):
        ''' Returns saliency probabilities for the sampled points.

        Args:
            query          (b, n_q, 3): points
            grid_feat      (b, feat_dim, res, res, res): latent conditioned code c

        Returns:
            sal         (b, n_q, 1)
            sample_feat (b, n_q, feat_dim)
        '''
        logits, sample_feat = self.sal_decoder(query, grid_feat, True)
        sal = F.softplus(logits).squeeze(-1)
        return sal / (sal + 1), sample_feat