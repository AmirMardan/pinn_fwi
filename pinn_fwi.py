from config import *
from networks import Autoencoder  #, Physics

seed_everything()

d_obs = torch.load(
    f= PATH + "/data_model/taux_obs_" + PACKAGE + "_" + MODEL + "_" + str(N_SHOTS)
    )

# TODO:  Normalize 
# d_obs = data_normalization(d_obs)
d_obs = d_obs.unsqueeze(0).to(device=DEVICE)
# N_RECEIVERS = d_obs.shape[3]
# d_obs = d_obs[:,:3,...]
# print(d_obs.shape)

vp, vp0 = earth_model("marmousi_bl", smooth=15, device=DEVICE)

# im = splt.earth_model({"Vp": vp, "$Vp_0$":vp0}, cmap="jet")
# plt.show(block=False)

criteria = torch.nn.MSELoss(reduction='sum')
# criteria = torch.nn.L1Loss(reduction='sum')

autoencoder = Autoencoder(batch_size=BATCH_SIZE, in_channels=N_SHOTS,
                  n_blocks_encoder=N_BLOCKS_ENCODER, n_blocks_decoder=N_BLOCKS_DECODER,
                  final_size_encoder=FINAL_SIZE_ENCODER, initial_shape_decoder=DECODER_INITIAL_SHAPE,
                  nt=NT, nr=N_RECEIVERS, final_spatial_shape=model_shape,
                  m_min=VP_MIN, m_max=VP_MAX,
                  final_out_channels=1
                  )
autoencoder = autoencoder.to(device=DEVICE)

# optim_phys = torch.optim.Adam(autoencoder.parameters(), lr=5, betas=(0.5, 0.9))
optim_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, betas=(0.5, 0.9))
scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(optim_autoencoder, 30, gamma=0.5)
    
all_loss = []

for iter in range(ITERATION):
    loss, m, autoencoder = train_fun(Physics=Physics, autoencoder=autoencoder,
                     d_obs=d_obs, freqs=INV_FREQS,
                     optim_autoencoder=optim_autoencoder, criteria=criteria,
                     mini_batches = MINI_BATCHES,
                     src_loc=src_loc, rec_loc=rec_loc, src=src,
                     inpa=inpa, test=None)
    
    all_loss.append(loss)
        
    if iter%1 == 0:
        print(f"Iteration {iter + 1} ===== loss: {all_loss[-1]}")
        
    scheduler_autoencoder.step()

plt.figure()
plt.imshow(m.cpu().detach(), cmap="jet")
plt.colorbar()

plt.figure()
plt.plot(all_loss)
plt.show()