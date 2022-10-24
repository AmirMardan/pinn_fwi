from prettytable import PrettyTable
import torch 
import PyFWI.model_dataset as md
import os 
import random 
import numpy as np
from tqdm import tqdm
import deepwave
import torch.nn as nn
import PyFWI.wave_propagation as wave
import PyFWI.processing as process
import matplotlib.pyplot as plt


PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))


def earth_model(name, smooth=0, device="cpu"):
    if name == "marmousi_30":
        vp = torch.load(PATH + "/data_model/marmousi_30.bin")
    elif name == "marmousi_bl":
        vp = torch.load(PATH + "/data_model/marmousi_bl.bin")
         
    vp0 = torch.tensor(md.model_smoother({"vp":vp}, smoothing_value=10)['vp'])
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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


def load_checkpoint(checkpoint_file, model, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    