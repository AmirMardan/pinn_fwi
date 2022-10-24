import matplotlib.pyplot as plt 
import PyFWI.seiplot as splt
import PyFWI.acquisition as acq
import os 
from utils import * 

PACKAGE = "deepwave"

MODEL = "marmousi_bl" 
DEVICE = ('cpu', 'cuda')[torch.cuda.is_available()]
ITERATION = 500
N_BLOCKS_ENCODER = 5
N_BLOCKS_DECODER = 4
BATCH_SIZE = 1
VP_MIN = 1450.0
VP_MAX = 4550.0 

if MODEL == "marmousi":
    model_shape = (100, 310)
elif MODEL in ["marmousi_bl", "np_marmousi"]:
    model_shape = (116, 227)

DECODER_INITIAL_SHAPE = torch.div(torch.tensor(model_shape), (2 ** (N_BLOCKS_DECODER - 1)), rounding_mode='floor')
FINAL_SIZE_ENCODER = BATCH_SIZE * DECODER_INITIAL_SHAPE[0] * DECODER_INITIAL_SHAPE[1]
# print(FINAL_SIZE_ENCODER)

DT = 0.004
F_PEAK = 25
DH = 5

N_SHOTS = 60
MINI_BATCHES = 3  # Number of mini batches
INV_FREQS = [12, 25, 60] # 
N_SOURCE_PER_SHOT = 1

inpa = {
    'ns': N_SHOTS,  # Number of sources
    'sdo': 8,  # Order of FD
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

# Design the acquisition
src_loc_temp, rec_loc_temp, n_surface_rec, n_well_rec = acq.acq_parameters(
    ns=inpa['ns'], rec_dis=inpa['rec_dis'],
    depth=depth, offsetx=offsetx,
    acq_type=inpa['acq_type'], dh=inpa['dh'],
    sdo=inpa['sdo']
)
src_loc_temp[:, 1] -= 2 * inpa['dh']

# Create the source
N_RECEIVERS = n_surface_rec + 2 * n_well_rec
inpa["n_well_rec"] = n_well_rec

if PACKAGE == "pyfwi":
    print("======= Package pyfwi is used =========")
    from train import train_pyfwi
    from networks import Physics_pyfwi
    
    Physics = Physics_pyfwi
    train_fun = train_pyfwi
    
    src_loc = src_loc_temp
    rec_loc = rec_loc_temp
    src = acq.Source(src_loc_temp, inpa['dh'], inpa['dt'])
    src.Ricker(inpa['fdom'])
    
    
elif PACKAGE == "deepwave":
    print("======= Package deepwave is used =========")
    from train import train_deepwave
    from networks import Physics_deepwave
    
    Physics = Physics_deepwave
    train_fun = train_deepwave
    
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
    deepwave.wavelets.ricker(F_PEAK, NT, DT, 1.5 / F_PEAK)
    .repeat(N_SHOTS, N_SOURCE_PER_SHOT, 1)
    .to(DEVICE)
    )
