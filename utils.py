from prettytable import PrettyTable
import torch 
import os 
import random 
import numpy as np
from tqdm import tqdm
import deepwave
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tools import (SaveResults, rock_properties, 
                   load_checkpoint, awgn)
from PyFWI.rock_physics import pcs2dv_gassmann
from pyfwi_tools import model_resizing
from typing import List, Tuple, Optional

PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))

def earth_model(name, smooth=0, device="cpu"):
    
    if name == "marmousi_30":
        vp = torch.load(PATH + "/data_model/marmousi_30.bin")
    elif name == "marmousi_bl":
        vp = torch.load(PATH + "/data_model/marmousi_bl.bin")
        
    vp0 = torch.tensor(gaussian_filter(vp, sigma=smooth))
    
    return vp.to(device=device), vp0.to(device=device)
    

def count_parameters(model):
    '''
    Function to count parameters of a network
    '''
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def data_normalization(data):
    '''
    Normalize before squeezing for batch and along t-direction.
    It means, we measure the max of data for each receivers
    '''
    data_max, _ = data.max(dim=1, keepdim=True)
    
    return data / (data_max.abs() + 1e-10)


def seed_everything(seed=42):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 

def save_checkpoint(model, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        # "optimizer": optimizer.state_dict(),
        
        # "inpa": inpa
    }
    
    torch.save(checkpoint, filename)