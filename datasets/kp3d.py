import sys
import os
proj_root = os.path.abspath('.')
if proj_root not in sys.path:
    sys.path.append(proj_root)

from typing import Tuple
from dataclasses import dataclass
from datasets.augmentations import pc_remove_volumes, pc_density_change, random_rotation, pc_noise
from common.pointcloud import random_select
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image as im
import random

from datasets.basedatasets import ConfigBaseDataset, BaseDataset

@dataclass
class ConfigKp3dDetDataset(ConfigBaseDataset):
    name:str='dataset_kp3d_det'
    
    data_path:str='/home/han/Projects/Datasets/Kp3d/scenes_shapenet/test_scenes_image36_depth/categories/'
    train_file:str='/home/han/Projects/Datasets/Kp3d/scenes_shapenet/test_scenes_image36_depth/train.txt'
    val_file:str='/home/han/Projects/Datasets/Kp3d/scenes_shapenet/test_scenes_image36_depth/val.txt'
    test_file:str='/home/han/Projects/Datasets/Kp3d/scenes_shapenet/test_scenes_image36_depth/test.txt'
    
    # resolutions
    res_h:int=480
    res_w:int=640
    
class Kp3dDetDataset(BaseDataset):
    def __init__(self, exp_name:str, mode='train'):
        self.cfg = ConfigKp3dDetDataset()
        self.cfg = self.cfg.load_from_exp(exp_name)
        
        super(Kp3dDetDataset, self).__init__(self.cfg, mode)
        
        self.mode = mode
        self.scene_names = []
        with open(os.path.join(proj_root, self.data_file), 'r') as f:
            self.scene_names = f.read().splitlines()
                
    def __len__(self):
        return len(self.scene_names)
        
    def __getitem__(
        self, 
        idx:int
        ):
        '''
        Args:
            idx
        Returns:
            'train' and 'val' mode
            pc              (n, 3)
            k               (4, 4)
            kps3d           (n_kp, 3)
            kps2d           (n_kp, 2) 
            kps2d_det       (n_kp)
            kps2d_desc      (n_kp, 256)
            poses           (n_kp, 4, 4)
            kps_frame_id    (n_kp)
            images          (n_img, h, w, 3)
            depths          (n_img, h, w)
            poses           (n_img, 4, 4)

            'test' mode
            pc          (n, 3)
        '''
        print(f'Loading {self.scene_names[idx]}...')
        
        split = self.scene_names[idx].split('-')
        cat_name = split[0]
        scene_name = split[1]
        
        scene_path = os.path.join(self.cfg.data_path, cat_name, scene_name)
        kpinfo_path = os.path.join(scene_path, 'kpinfo')
        
        pc = torch.from_numpy(np.loadtxt(os.path.join(scene_path, 'pc.txt'), dtype=np.float32)) # (n, 3)
        pc_colored = torch.from_numpy(np.loadtxt(os.path.join(scene_path, 'pc_colored_2048.txt'), dtype=np.float32)) # (n, 3)
        
        if self.mode == 'test': 
            return pc, pc_colored, cat_name, scene_name
        
        K = torch.from_numpy(np.loadtxt(os.path.join(scene_path, 'k.txt'), dtype=np.float32)) # (4, 4)
        kps2d = torch.from_numpy(np.loadtxt(os.path.join(kpinfo_path, 'kps2d.txt'), dtype=np.int32)) # (n_kp, 2)
        kps2d_det = torch.from_numpy(np.loadtxt(os.path.join(kpinfo_path, 'kps2d_det.txt'), dtype=np.float32)) # (n_kp)
        kps2d_desc = torch.from_numpy(np.loadtxt(os.path.join(kpinfo_path, 'kps2d_desc.txt'), dtype=np.float32)) # (n_kp, 256)
        kps2d_pose = torch.from_numpy(np.loadtxt(os.path.join(kpinfo_path, 'kps2d_pose.txt'), dtype=np.float32)).reshape(-1, 4, 4) # (n_kp, 4, 4)
        kps3d = torch.from_numpy(np.loadtxt(os.path.join(kpinfo_path, 'kps3d.txt'), dtype=np.float32)) # (n_kp, 3)
        kps_frame_id = torch.from_numpy(np.loadtxt(os.path.join(kpinfo_path, 'kps2d_frame_id.txt'), dtype=np.int32)) # (n_kp)
        kps3d_match_score = torch.from_numpy(np.loadtxt(os.path.join(kpinfo_path, 'kps3d_match_score.txt'), dtype=np.float32)) # (n_kp)
        
        image_path = os.path.join(scene_path, 'image')
        depth_path = os.path.join(scene_path, 'depth')
        pose_path = os.path.join(scene_path, 'pose')
        images = torch.zeros(0, self.cfg.res_h, self.cfg.res_w, 3)
        depths = torch.zeros(0, self.cfg.res_h, self.cfg.res_w)
        poses = torch.zeros(0, 4, 4)
        blender_to_open3d = torch.from_numpy(np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]])).type(torch.float32)
        
        image_count = len(os.listdir(image_path))
        for image_idx in range(0, image_count):
            image = torch.from_numpy(np.array(im.open(os.path.join(image_path, f'{image_idx}.jpg'))))
            images = torch.cat((images, image.unsqueeze(0)), dim=0)
            depth = torch.from_numpy(np.array(im.open(os.path.join(depth_path, f'{image_idx}.png')))) # (h, w)
            depth = depth/10000.0
            depths = torch.cat((depths, depth.unsqueeze(0)), dim=0)
            pose = torch.from_numpy(np.loadtxt(os.path.join(pose_path, f'{image_idx}.txt'), dtype=np.float32)) # (4, 4)
            pose = pose @ blender_to_open3d   
            poses = torch.cat((poses, pose.unsqueeze(0)), dim=0)
            
        # data augmentation
        
        #pc = pc_density_change(pc, 10000, 500)
        pc, R = random_rotation(pc)
        R_4x4 = torch.eye(4)
        R_4x4[:3, :3] = R
        R_4x4_inv = torch.linalg.inv(R_4x4)
        
        sigma = random.uniform(0.0, 0.02)
        pc = pc_noise(pc=pc, sigma=sigma)
        
        kps3d = kps3d @ R
        poses = torch.linalg.inv(torch.linalg.inv(poses) @ R_4x4_inv)
        kps2d_pose = torch.linalg.inv(torch.linalg.inv(kps2d_pose) @ R_4x4_inv)
        
        return pc, K, kps3d, kps3d_match_score, kps2d, kps2d_det, kps2d_desc, kps2d_pose, kps_frame_id, images, depths, poses
        
        