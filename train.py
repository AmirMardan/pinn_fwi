
from utils import *
import PyFWI.acquisition as acq
import torchfwi


def deepwave_engine(Physics, dh, dt,
                    src_loc, rec_loc, src,
                    batch, mini_batches
                    ):
    src_loc_batch = src_loc[batch::mini_batches]
    rec_loc_batch = rec_loc[batch::mini_batches]
    src_batch = src[batch::mini_batches]
            
    physics = Physics(dh, dt, src_batch,
                      src_loc_batch, rec_loc_batch)
    return physics


def pyfwi_engin(Physics, inpa, dh, dt, fpeak,
                src_loc, rec_loc,
                batch, mini_batches
                ):
    src_loc_batch = src_loc[batch::mini_batches]
        
    src_batch = acq.Source(src_loc_batch, dh, dt)
    src_batch.Ricker(fpeak)

    physics = Physics(inpa, src=src_batch, rec_loc=rec_loc,
                      f_show=False, b_show=False)
    return physics
        
        
def train_deepwave(Physics, autoencoder, d_obs,
                   optim_autoencoder, criteria, 
                    mini_batches,
                   src_loc, rec_loc, src, inpa,
                   freqs,
                   test=None):
    
    loss_minibatch = []
    for batch in tqdm(range(mini_batches), leave=False):
        loss_freqs = []
        for freq in freqs:
            optim_autoencoder.zero_grad()
            
            src_loc_batch = src_loc[batch::mini_batches]
            rec_loc_batch = rec_loc[batch::mini_batches]
            src_batch = src[batch::mini_batches]
                    
            physics = Physics(inpa['dh'], inpa['dt'], src_batch,
                            src_loc_batch, rec_loc_batch)

            loss, m = train_engine(autoencoder, physics,
                                   criteria, optim_autoencoder,
                                   d_obs, freq,
                                   batch, mini_batches,
                                   test=test)
            # plt.imshow(m.grad, cmap='jet')
            loss_freqs.append(loss)
        
        loss_minibatch.append(np.mean(loss_freqs))
        
    return np.mean(loss_minibatch), m, autoencoder


def train_pyfwi(Physics, autoencoder, d_obs,
                optim_autoencoder, criteria,
                mini_batches, freqs,
                src_loc, rec_loc, src, inpa,
                test=None):
    
    loss_minibatch = []
    for batch in tqdm(range(mini_batches), leave=True):
        loss_freqs = []
        for freq in freqs:
            optim_autoencoder.zero_grad()
        
            src_loc_batch = src_loc[batch::mini_batches]
            
            src_batch = acq.Source(src_loc_batch, inpa['dh'], inpa['dt'])
            src_batch.Ricker(inpa['fdom'])

            physics = Physics(inpa, src=src_batch, rec_loc=rec_loc,
                            f_show=False, b_show=False)
            
            loss, m = train_engine(autoencoder, physics,
                                    criteria, optim_autoencoder,
                                    d_obs, freq,
                                    batch, mini_batches,
                                    test=test)
            
            loss_freqs.append(loss)
        
        loss_minibatch.append(np.mean(loss_freqs))
                
    return np.mean(loss_minibatch), m, autoencoder


def train_engine(autoencoder, physics,
                 criteria, optim_autoencoder,
                 d_obs, freq,
                 batch, mini_batches,
                 test=None
                 ):
    
    if test is None:  
        m = autoencoder(d_obs)
        m = m.squeeze()
        # m.retain_grad()
    else:
        m = test.requires_grad_(True)
        plt.figure()
                
    taux_est = physics(m)
    if freq:     
      taux_est_filtered, d_obs_filtered = torchfwi.lpass(
        taux_est, d_obs[:, batch::mini_batches], 
        freq, 800)
    else:
        taux_est_filtered = taux_est
        d_obs_filtered = d_obs[:, batch::mini_batches]

                
    loss = criteria(taux_est_filtered, d_obs_filtered)
                    
    loss.backward()
    optim_autoencoder.step()
    
    return loss.item(), m