PINN-FWI for estimating the Marmousi velocity model
====================================================

In this repository, I implemented the physics-informed neural network (PINN) for full-waveform inversion. 
This PINN can be implemented with or without attention block.
The architecture of their study is shown in the following figure. 
 
![architecture](/readme_files/architecture.png)

For running the code, you should use this [notebook](https://github.com/AmirMardan/piann_fwi/blob/main/pinn_fwi.ipynb).
The required parameters for running this notebook should be set in this [config](https://github.com/AmirMardan/piann_fwi/blob/main/config.py) file.

<span style='color:red; font-weight:bold;'>Note: </span> I have commented cell 3 in this notebook, you should run this cell whenever you change an acquisition parameter (and for the first time using the codes).

<span style='color:red; font-weight:bold;'>Note: </span> Please use the [requirements](https://github.com/AmirMardan/piann_fwi/blob/main/requirements.txt) file (written in the jupyter file) to install the packages with specified versions to be sure everything works.
```console
pip install -r requirements.txt
```

In this repo, there are four scripts for running FWI.
1. [`pinn_fwi.py`](https://github.com/AmirMardan/piann_fwi/blob/main/pinn_fwi.py) for performing PINN- or PIANN-FWI.
2. [`original_fwi.py`](https://github.com/AmirMardan/piann_fwi/blob/main/original_fwi.py) for running the conventional FWI.
3. [`pinn_for_init.py`](https://github.com/AmirMardan/piann_fwi/blob/main/pinn_for_init.py) for performing PINN- or PIANN-FWI to create an initial model and use that to perform the conventional FWI.
4. [`pinn_fwi.ipynb`](https://github.com/AmirMardan/piann_fwi/blob/main/pinn_fwi.ipynb) for performing PINN- or PIANN-FWI, but this notebook might not be updated.

The result of running this code for 22 shots with 2500 epochs on the Marmousi model is shown in the following figures. 

![res](/readme_files/marmousi_clean.png)
For a faster convergence (300 epochs), I considered geophones around the model and the results are
![with_init](/readme_files/image2024_marmousi_clean.png)
where the hybrid method is using the PIANN-FWI for creating only initial model.


Reference:
```
@inproceedings{mardan2024piann_eage,
	title = {Physics-informed attention-based neural networks for full-waveform inversion},
  	author = {Mardan, Amir and Fabien-Ouellet, Gabriel},
  	year = {2024},
  	booktitle = {85$^{th}$ {EAGE} Annual Conference \& Exhibition},
	publisher = {European Association of Geoscientists \& Engineers},
	pages = {1-5},
  	doi = {}
}
``` 