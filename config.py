import matplotlib.pyplot as plt 
import os 
import torch
import numpy as np
from utils import * 
from pyfwi_tools import show_earth_model

PACKAGE = "deepwave"
from train import train_deepwave
from networks import Physics_deepwave
    
Physics = Physics_deepwave
train_fun = train_deepwave

rec_in_well = False  # If you want to have geophones in well

LR_MILESTONE = 50
MODEL = "marmousi_bl" 
LOAD_CHP = True
DEVICE = ("cpu", "cuda")[torch.cuda.is_available()]
# DEVICE =  (("cpu", "cuda")[torch.cuda.is_available()],
#           "mps")[torch.backends.mps.is_available()]
ITERATION = 800
PRINT_FREQ = np.ceil(ITERATION/20)

N_BLOCKS_ENCODER = 5
N_BLOCKS_DECODER = 4
BATCH_SIZE = 1
VP_MIN = 1450.0
VP_MAX = 4550.0 

LAM_PRIOR = 0.0  # 1e-5
rp_properties = None
if MODEL == "marmousi":
    model_shape = [100, 310]
elif MODEL in ["marmousi_bl", "np_marmousi"]:
    model_shape = [116, 227]

DECODER_INITIAL_SHAPE = torch.div(torch.tensor(model_shape), (2 ** (N_BLOCKS_DECODER - 1)), rounding_mode='floor')
FINAL_SIZE_ENCODER = BATCH_SIZE * DECODER_INITIAL_SHAPE[0] * DECODER_INITIAL_SHAPE[1]
# print(FINAL_SIZE_ENCODER)

DT = 0.004
F_PEAK = 25
DH = 5

N_SHOTS = 18
MINI_BATCHES = 3  # Number of mini batches
INV_FREQS = [12, 25, 60] # 
N_SOURCE_PER_SHOT = 1

inpa = {
    'ns': N_SHOTS,  # Number of sources
    'sdo': 4,  # Order of FD
    'fdom': F_PEAK,  # Central frequency of source
    'dh': DH,  # Spatial sampling rate
    'dt': DT,  # Temporal sampling rate
    'acq_type': 2,  # Type of acquisition (0: crosswell, 1: surface, 2: both)
    't': 0.6, #8,  # Length of operation
    'npml': 20,  # Number of PML 
    'pmlR': 1e-5,  # Coefficient for PML (No need to change)
    'pml_dir': 2,  # type of boundary layer
    'device': 2, # The device to run the program. Usually 0: CPU 1: GPU
    'seimogram_shape': '3d',
    'energy_balancing': False,
    "chpr": 70,
    "f_inv": INV_FREQS
}
NT = int(inpa['t'] // inpa["dt"] + 1)

inpa['rec_dis'] =  1 * inpa['dh']  # Define the receivers' distance

offsetx = inpa['dh'] * model_shape[1]
depth = inpa['dh'] * model_shape[0]

surface_loc_x = np.arange(2*inpa["dh"], offsetx-2*inpa["dh"], inpa['dh'], np.float32)
n_surface_rec = len(surface_loc_x)
surface_loc_z = 4 * inpa["dh"] * np.ones(n_surface_rec, np.float32)
surface_loc = np.vstack((surface_loc_x, surface_loc_z)).T

if rec_in_well:
    well_z = np.arange(2*inpa["dh"], depth-2*inpa["dh"], inpa['dh'], np.float32)
    n_well_rec = len(well_z)
    well_left = np.vstack((4 * inpa["dh"] * np.ones(n_well_rec, np.float32),
                        well_z)).T
    well_right = np.vstack((offsetx - 4 * inpa["dh"] * np.ones(n_well_rec, np.float32),
                        well_z)).T

    rec_loc_temp = np.vstack((
        well_left,
        surface_loc,
        well_right
    ))
else:
    rec_loc_temp = surface_loc
    n_well_rec = 0

src_loc_temp = np.vstack((
    np.linspace(4*inpa["dh"], offsetx-4*inpa["dh"], N_SHOTS, np.float32),
    2 * inpa["dh"] * np.ones(N_SHOTS, np.float32)
    )).T
 
# src_loc_temp = np.array([[  20.,   20.],
#                       [ 555.,   20.],
#                       [1090.,   20.]], dtype=np.float32)
src_loc_temp[:, 1] -= 2 * inpa['dh']

# Create the source
N_RECEIVERS = n_surface_rec + 2 * n_well_rec
inpa["n_well_rec"] = n_well_rec
    
# Shot 1 source located at cell [0, 1], shot 2 at [0, 2], shot 3 at [0, 3]
src_loc = torch.zeros(N_SHOTS, N_SOURCE_PER_SHOT, 2,
                        dtype=torch.long, device=DEVICE)
src_loc[:, 0, :] = torch.Tensor(np.flip(src_loc_temp) // DH)


# Receivers located at [0, 1], [0, 2], ... for every shot
rec_loc = torch.zeros(N_SHOTS, N_RECEIVERS, 2,
                        dtype=torch.long, device=DEVICE)
rec_loc[:, :, :] = (
    torch.Tensor(np.flip(rec_loc_temp)/DH)
    )
    
    
src = (
    wavelets.ricker(F_PEAK, NT, DT, 1.5 / F_PEAK)
    .repeat(N_SHOTS, N_SOURCE_PER_SHOT, 1)
    .to(DEVICE)
    )
