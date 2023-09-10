import os
import sys
import numpy as np
import torch
import cv2
from path import Path
from tqdm import tqdm

def find_kps_in_frame(
    kps:torch.Tensor,   
    kps_frame_id:torch.Tensor,
    frame_id:int,
    ) -> torch.Tensor:
    '''
    Args:
        kps: (n_kp, 3) or (n_kp, 2)
        kps_frame_id: (n_kp)
        frame_id: int
    Return:
        frame_kps: (n_kp', 3) or (n_kp', 2)
    '''
    frame_kps = kps[kps_frame_id == frame_id]
    return frame_kps
    