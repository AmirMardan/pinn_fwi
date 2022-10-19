
from utils import *
import PyFWI.acquisition as acq


def train_deepwave(Physics, autoencoder, d_obs,
                   optim_autoencoder, criteria, 
                    mini_batches,
                   src_loc, rec_loc, src, inpa,
                   test=None):
    
    loss_minibatch = []
    for batch in tqdm(range(mini_batches), leave=False):
        optim_autoencoder.zero_grad()
            
        src_loc_batch = src_loc[batch::mini_batches]
        rec_loc_batch = rec_loc[batch::mini_batches]
        src_batch = src[batch::mini_batches]
            
        physics = Physics(inpa['dh'], inpa['dt'], src_batch,
                                src_loc_batch, rec_loc_batch)
        if test is None:  
            m = autoencoder(d_obs)
            m = m.squeeze()
            # m.retain_grad()
        else:
            m = test.requires_grad_(True)
            plt.figure()

        # plt.figure()
        # plt.imshow(m.cpu().detach(), cmap='jet')
        # plt.title(f"Iteration {iter + 1}")
        # plt.colorbar()
        # # print(m.shape)
        # # print(m.min(), m.max())
        
        taux_est = physics(m)
        loss = criteria(taux_est, d_obs[:, batch::mini_batches])
            
        loss.backward()
        # optim_phys.step()
        optim_autoencoder.step()
        # plt.imshow(m.grad, cmap='jet')
        
        loss_minibatch.append(loss.item())
        
    return np.sum(loss_minibatch)/len(loss_minibatch), m, autoencoder


def train_pyfwi(Physics, autoencoder, d_obs,
                optim_autoencoder, criteria,
                mini_batches,
                src_loc, rec_loc, src, inpa,
                test=None):
    
    loss_minibatch = []
    for batch in tqdm(range(mini_batches), leave=True):
        optim_autoencoder.zero_grad()
    
        src_loc_batch = src_loc[batch::mini_batches]
        rec_loc_batch = rec_loc  # [batch::mini_batches]
        
        src_batch = acq.Source(src_loc_batch, inpa['dh'], inpa['dt'])
        src_batch.Ricker(inpa['fdom'])

        physics = Physics(inpa, src=src_batch, rec_loc=rec_loc_batch,
                          f_show=False, b_show=False)
        
        if test is None:   
            m = autoencoder(d_obs)
            m = m.squeeze()
            # m.retain_grad()
        else:
            m = test.requires_grad_(True)
        
        taux_est = physics(m)
        loss = criteria(taux_est, d_obs[:, batch::mini_batches])
            
        loss.backward()
        optim_autoencoder.step()
        loss_minibatch.append(loss.item())
        
    return np.sum(loss_minibatch)/len(loss_minibatch), m, autoencoder

        