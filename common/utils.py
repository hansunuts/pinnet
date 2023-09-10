import torch
import numpy as np
from typing import Tuple

def to_device(tensors:Tuple):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ret = []
    for i, tensor in enumerate(tensors):
        ret.append(tensor.to(device))
    return tuple(ret)

def torch_to_np(tensors:Tuple):
    ret = []
    for i, tensor in enumerate(tensors):
        ret.append(tensor.cpu().detach().numpy())
    return tuple(ret)

def np_to_torch(arrays:Tuple, to_device=True):
    ret = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, array in enumerate(arrays):
        tensor = torch.from_numpy(array)
        if to_device:
            tensor = tensor.to(device)
        ret.append(tensor)
    return tuple(ret)