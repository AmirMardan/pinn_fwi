
from utils import *
import torchfwi
# from torchvision.transforms import GaussianBlur
from tools import gaussian_filter_2d

def deepwave_engine(Physics, 
                    dh, 
                    dt,
                    src_loc, 
                    rec_loc, 
                    src,
                    batch, 
                    mini_batches
                    ):
    src_loc_batch = src_loc[batch::mini_batches]
    rec_loc_batch = rec_loc[batch::mini_batches]
    src_batch = src[batch::mini_batches]
            
    physics = Physics(dh, dt, src_batch,
                      src_loc_batch, rec_loc_batch)
    return physics
        
        
def train_deepwave(Physics, 
                   autoencoder, 
                   d_obs,
                   optim_autoencoder, 
                   criteria, 
                    mini_batches,
                   src_loc, 
                   rec_loc, 
                   src, 
                   inpa,
                   freqs,
                   lam_prior: float = 1e-4,
                    well_locations: np.ndarray=None,
                    well_data: np.ndarray=None,
                   test=None):
    
    loss_data_minibatch = []
    loss_prior_minibatch = []
    # for batch in tqdm(range(mini_batches), leave=False):
    for batch in range(mini_batches):
        loss_freqs = []
        loss_priors = []
        for freq in freqs:
            optim_autoencoder.zero_grad()
            
            src_loc_batch = src_loc[batch::mini_batches]
            rec_loc_batch = rec_loc[batch::mini_batches]
            src_batch = src[batch::mini_batches]
                    
            physics = Physics(inpa['dh'], inpa['dt'], src=src_batch,
                            src_loc=src_loc_batch, rec_loc=rec_loc_batch
                            )

            loss_data, loss_prior, m = train_engine(autoencoder, physics,
                                   criteria, optim_autoencoder,
                                   d_obs, freq,
                                   batch, mini_batches,
                                   lam_prior=lam_prior,
                                   well_locations=well_locations,
                                   well_data=well_data,
                                   test=test
                                   )
            # plt.imshow(m.grad, cmap='jet')
            loss_freqs.append(loss_data)
            loss_priors.append(loss_prior)
        
        loss_data_minibatch.append(np.mean(loss_freqs))
        loss_prior_minibatch.append(np.mean(loss_priors))
    return np.mean(loss_data_minibatch), np.mean(loss_prior_minibatch), m, autoencoder


def train_engine(autoencoder, physics,
                 criteria, optim_autoencoder,
                 d_obs, freq,
                 batch, mini_batches,
                 lam_prior: float = 1e-4,
                 well_locations: np.ndarray=None,
                 well_data: np.ndarray=None,
                 test=None):
    # transfer = GaussianBlur(kernel_size=5, sigma=5)
    earth_model = autoencoder(d_obs)
    device = earth_model.device
    vp = earth_model[:, 0, ...].squeeze()
    
    if test is None:  
        if earth_model.shape[1] == 3:
            vs = earth_model[:, 1, ...].squeeze()
            rho = earth_model[:, 2, ...].squeeze()
        elif earth_model.shape[1] == 1:
            vs = 0 * torch.ones(vp.shape, 
                                  device=device,
                                  dtype=torch.float32)
            rho = torch.ones(vp.shape, 
                            device=device,
                            dtype=torch.float32)
        else: 
            raise Exception(f"Model should have either one parameter or 3 but got {earth_model.shape[1]}")
        
    else:
        vp = test.requires_grad_(True)
        plt.figure()
    
    m = vp
    
    taux_est = physics(m.squeeze())
    
    if freq:     
      taux_est_filtered, d_obs_filtered = torchfwi.lpass(
        taux_est, d_obs[:, batch::mini_batches], 
        freq, 800)
    else:
        taux_est_filtered = taux_est
        d_obs_filtered = d_obs[:, batch::mini_batches]

    loss_data = criteria(taux_est_filtered, d_obs_filtered)
    if well_data is not None:
        loss_prior = lam_prior * criteria(m[:well_data.shape[0], well_locations], well_data) 
        loss = loss_data + loss_prior
        # print(f"Loss is {loss_data.item()} for data and {loss_model.item()} for model")
    else:
        loss = loss_data
        print("entered no lam")
    
    loss.backward()
    optim_autoencoder.step()
    
    return loss_data.item(), loss_prior.item(), m