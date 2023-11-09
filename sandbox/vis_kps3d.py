import sys
import os
project_root = os.path.abspath(".")
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.optim as optim
from tqdm import tqdm
from common.utils import to_device
from common.visualization import vis_pc_kp
from common.nms import nms
from common.pointcloud import random_select
from models.pinnet_det import PinNetDet
from datasets.kp3d import Kp3dDetDataset
from datasets.augmentations import random_rotation, add_gaussian_noise

# ==============================
# Parameters:
exp_name = 'plane'
model_name = 'plane.pth' 

# pc
point_num = 10000
noise_sigma = 0.000
downsample_rate = 1.0

# ---- kp
kp3d_det_thr=0.55
kp3d_nms_radius = 0.1

optimize_query = True
optim_max_iter = 10
optim_lr = 0.00001
kp3d_nms_max_num = 100

kp_pc_max_dist = 0.05

# ==============================

point_num = int(point_num / downsample_rate)

device = 'cuda:0'
model = PinNetDet(exp_name=exp_name).to(device)
model.load_weight(f"{project_root}/experiments/{exp_name}/{model_name}")
model.eval()

ds = Kp3dDetDataset(exp_name=exp_name, mode='test')
for data in tqdm(ds):
    pc, pc_colored, cat_name, scene_name = data
    
    pc = random_select(pc, point_num)
    #pc, R = random_rotation(pc)
    pc = add_gaussian_noise(point_cloud=pc, sigma=noise_sigma)
    
    (pc, pc_colored) = to_device((pc, pc_colored))
    
    sal = torch.zeros(0)
    querys = pc

    if optimize_query:
        for i in range(0, optim_max_iter):
            querys.requires_grad_()
            optimizer = optim.Adam([querys,], lr=optim_lr)
            sal = model.forward(pc.unsqueeze(0), querys.unsqueeze(0))
            loss = 1 - sal.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            sal = sal.squeeze(0)
            sal_mask = sal > kp3d_det_thr
            
            querys = querys.detach()
            querys = querys[sal_mask]
            sal = sal[sal_mask]
    else:
        sal = model.forward(pc.unsqueeze(0), querys.unsqueeze(0))

    kps3d, kps3d_sal = nms(querys[:, :3], sal.squeeze(0), kp3d_nms_radius, kp3d_nms_max_num)

    kps3d_cube_mask = (kps3d[:, 0] < 0.5) * (kps3d[:, 0] > -0.5) * \
        (kps3d[:, 1] < 0.5) * (kps3d[:, 1] > -0.5) * (kps3d[:, 2] < 0.5) * (kps3d[:, 2] > -0.5)
    dist = torch.norm(pc[None, :, :] - kps3d[:, None, :3], dim=2).min(dim=1).values
    kps3d_dist_mask = dist < kp_pc_max_dist
    kps3d_det_mask = kps3d_sal > kp3d_det_thr
    kps3d = kps3d[kps3d_cube_mask * kps3d_dist_mask * kps3d_det_mask]
    kps3d_sal = kps3d_sal[kps3d_cube_mask * kps3d_dist_mask * kps3d_det_mask]
    
    vis_pc_kp(pc, kps3d)