import os
from typing import Tuple
from dataclasses import dataclass
from common.config import BaseConfig
import torch
from torch.utils.data import Dataset

@dataclass
class ConfigBaseDataset(BaseConfig):
    data_path:str 
    train_file:str 
    val_file:str 
    test_file:str
    
@dataclass
class ConfigBaseDataset3D(ConfigBaseDataset):
    # train
    n_max_pt:int
    n_occ_pt:int
    n_non_occ_pt:int
    n_rep_pt:int
    # test
    n_query:int     

@dataclass
class ConfigBaseDataset2D(ConfigBaseDataset):
    img_h:int
    img_w:int

class BaseDataset(Dataset):
    def __init__(self, cfg:ConfigBaseDataset, mode):
        Dataset.__init__(self)
        self.mode = mode

        if mode == 'train':
            self.data_file = self.cfg.train_file
        elif mode == 'val':
            self.data_file = self.cfg.val_file
        elif mode == 'test':
            self.data_file = self.cfg.test_file
        else:
            raise Exception("Unkonwn dataset mode")
    
    def get_item_by_key(self, key:dict, with_batch=True):
        ''' get a single item by key. Used in analysis. '''
        raise NotImplementedError()
    
class BaseDataset3D(BaseDataset):
    def __init__(self, cfg:ConfigBaseDataset3D, mode):
        super(BaseDataset3D, self).__init__(cfg, mode)
    
    def _get_train_item(self, idx
                        ) -> Tuple[
                            torch.Tensor,       # pc1               (n_max_pt, 3)
                            torch.Tensor,       # rep1_pts          (n_rep_pt, 3)
                            torch.Tensor,       # occ1_pts          (n_occ_pt + n_non_occ_pt, 3)
                            torch.Tensor,       # occ1_gt           (n_occ_pt + n_non_occ_pt) 0 or 1
                            torch.Tensor,       # pc2               (n_max_pt, 3)
                            torch.Tensor,       # rep2_pts          (n_rep_pt, 3)
                            torch.Tensor,       # occ2_pts          (n_occ_pt + n_non_occ_pt, 3)
                            torch.Tensor,       # occ2_gt           (n_occ_pt + n_non_occ_pt) 0 or 1
                        ]:
        raise NotImplementedError()
    
    def _get_val_item(self, idx
                        ) -> Tuple[
                            torch.Tensor,       # pc1               (n_max_pt, 3)
                            torch.Tensor,       # rep1_pts          (n_rep_pt, 3)
                            torch.Tensor,       # occ1_pts          (n_occ_pt + n_non_occ_pt, 3)
                            torch.Tensor,       # occ1_gt           (n_occ_pt + n_non_occ_pt) 0 or 1
                            torch.Tensor,       # pc2               (n_max_pt, 3)
                            torch.Tensor,       # rep2_pts          (n_rep_pt, 3)
                            torch.Tensor,       # occ2_pts          (n_occ_pt + n_non_occ_pt, 3)
                            torch.Tensor,       # occ2_gt           (n_occ_pt + n_non_occ_pt) 0 or 1
                        ]:
        raise NotImplementedError()
    
    def _get_test_item(self, idx
                    ) -> Tuple[
                        torch.Tensor,       # pc                 (n_max_pt, 3)
                        torch.Tensor,       # query_pts          (n_query, 3)
                    ]:
        raise NotImplementedError()    
        
class BaseDataset2D(BaseDataset):
    def __init__(self, cfg:ConfigBaseDataset2D, mode):
        super(BaseDataset2D, self).__init__(cfg)
        
    def _get_train_item(self, idx
                        ) -> Tuple[
                            torch.Tensor,       # image         (h, w, 3)
                            torch.Tensor,       # k             (4, 4)
                            torch.Tensor,       # pose          (4, 4)
                            ]:
        raise NotImplementedError()
    
    def _get_val_item(self, idx
                        ) -> Tuple[
                            torch.Tensor,       # image         (h, w, 3)
                            torch.Tensor,       # k             (4, 4)
                            torch.Tensor,       # pose          (4, 4)
                            ]:
        raise NotImplementedError()
    
    def _get_test_item(self, idx
                        ) -> Tuple[
                            torch.Tensor,       # image         (h, w, 3)
                            torch.Tensor,       # k             (4, 4)
                            torch.Tensor,       # pose          (4, 4)
                            ]:
        raise NotImplementedError()
