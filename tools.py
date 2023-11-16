from typing import Any
import torch 
from pathlib import Path
from datetime import datetime
import numpy as np
from torch.nn.functional import conv2d
from torch.distributions import Normal
from math import ceil
from matplotlib.figure import Figure 

def save_checkpoint(
    model,
    file:str) -> None:
    """
    saves the checkpoints

    Parameters
    ----------
    model : 
        _description_
    file : str
        _description_
    """
    
    torch.save(model.state_dict(), file)
    print("== Checkpoint is saved! ==")
    
    
class SaveResults:
    def __init__(self, path:str) -> None:
        now = datetime.now()
        unique_string = now.strftime("%b_%d_%Y_%H_%M_%S")
        self.path_to_save = f"{path}/results/{unique_string}"
        Path(self.path_to_save).mkdir(parents=True)
        
    def numpy(self, 
              array:np.ndarray,
              file_name:str) -> None:
        np.save(f"{self.path_to_save}/{file_name}.npy", array)
        print(f"File {file_name} is saved in {self.path_to_save}")
        
    def network(self, 
                model, 
                file_name: str):
        save_checkpoint(model=model, 
                        file=f"{self.path_to_save}/{file_name}.tar")
        
        print(f"Checkpoint {file_name} is saved in {self.path_to_save}")
    
    def fig(self, fig: Figure,
            file_name: str,
            **kwargs) -> None:
        fig.savefig(f"{self.path_to_save}/{file_name}",
                    **kwargs) 
        

def rock_properties():
    classic_rock_properties = {
        'k_q': 37,
        'k_c': 21,
        'k_w': 3.01,
        'k_h': 0.13,
    
        'mu_q': 44,
        'mu_c': 10,
        'mu_w': 0,
        'mu_h': 0,
    
        'rho_q': 2.7,
        'rho_c': 2.6,
        'rho_w': 1.055,
        'rho_h': 0.336,
        'cs': 20
    }
    return classic_rock_properties


def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.) -> torch.Tensor:
    
    radius = ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())


def gaussian_filter_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    device = img.device
    img = img.to(device="cpu")
    kernel_1d = gaussian_kernel_1d(sigma)  # Create 1D Gaussian kernel
    
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    # img = img.unsqueeze(0).unsqueeze_(0)  # Need 4D data for ``conv2d()``
    # Convolve along columns and rows
    img = conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    img = conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    img = img.squeeze_(0).squeeze_(0) 
    return img.to(device=device)


def load_checkpoint(model, 
                    file: str,
                    device: str) -> None:
    """
    Load checkpoint for selected model

    Parameters
    ----------
    model
        A DL network
    file : str
        Checkpoint's file
    device : str
        Name of device
    """
    try:
        state = torch.load(file, map_location=torch.device(device))
        model.load_state_dict(state) 
        print("=== Checkpoint is loaded! ===")
    except:
        raise RuntimeError("Please enter a valid checkpoint file.")
    
    
def awgn(x_volt, snr):
    """

    https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python

    """
    # np.random.rand(2)
    if snr != 0:
        # for param in x_volt:
            x_watts = x_volt ** 2
            sig_avg_watts = torch.mean(x_watts)

            sig_avg_db = 10 * torch.log10(sig_avg_watts)

            noise_avg_db = sig_avg_db - snr
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            mean_noise = 0

            noise = torch.normal(mean_noise, torch.sqrt(noise_avg_watts), x_watts.shape)
            x_volt += noise
            # Noise up the original signal

    return x_volt