from utils import seed_everything
seed_everything(42)

from config import *
from networks import Autoencoder  #, Physics
# from tools import SaveResults#, load_checkpoint

print(f"Running on {DEVICE}")
d_obs = torch.load(
    f= PATH + "/data_model/taux_obs_" + PACKAGE + "_" + MODEL + "_" + str(N_SHOTS)
    )

# TODO:  Normalize 
# d_obs = data_normalization(d_obs)
d_obs = d_obs.unsqueeze(0).to(device=DEVICE)
# N_RECEIVERS = d_obs.shape[3]
# d_obs = d_obs[:,:3,...]
# print(d_obs.shape)

vp, vp0 = earth_model(MODEL, smooth=15, 
                      device=DEVICE)

# im = show_earth_model({"Vp": vp, "$Vp_0$":vp0}, cmap="jet")
# plt.show(block=False)

criteria = torch.nn.MSELoss(reduction='sum')
# criteria = torch.nn.L1Loss(reduction='sum')

autoencoder = Autoencoder(
        batch_size=BATCH_SIZE, 
        in_channels=N_SHOTS,
        n_blocks_encoder=N_BLOCKS_ENCODER, 
        n_blocks_decoder=N_BLOCKS_DECODER,
        final_size_encoder=FINAL_SIZE_ENCODER, 
        initial_shape_decoder=DECODER_INITIAL_SHAPE,
        nt=NT, nr=N_RECEIVERS,
        final_spatial_shape=model_shape,
        m_min=VP_MIN, m_max=VP_MAX,
        final_out_channels=FINAL_OUT_CHANNEL
        )
# print(BATCH_SIZE, N_SHOTS, FINAL_SIZE_ENCODER,
#       DECODER_INITIAL_SHAPE, NT, N_RECEIVERS)
if LOAD_CHP:
    autoencoder_default = Autoencoder(
        batch_size=BATCH_SIZE, 
        in_channels=22,
        n_blocks_encoder=N_BLOCKS_ENCODER, # 5
        n_blocks_decoder=N_BLOCKS_DECODER, # 4
        final_size_encoder=336, 
        initial_shape_decoder=[12, 28],
        nt=801, 
        nr=223,
        final_spatial_shape=[96, 227],
        m_min=VP_MIN, m_max=VP_MAX,
        final_out_channels=FINAL_OUT_CHANNEL
        )
    load_checkpoint(model=autoencoder_default, 
           file=f"{PATH}/saved_checkpoint/default_chp.tar",
           device="cpu")
    
    autoencoder.reshape(autoencoder_default,
                d_obs=d_obs.to(device="cpu"),
                decoder_initial_shape=DECODER_INITIAL_SHAPE,
                decoder_final_shape=model_shape
                )
    
autoencoder = autoencoder.to(device=DEVICE)

save_results = SaveResults(path=PATH)

# optim_phys = torch.optim.Adam(autoencoder.parameters(), lr=5, betas=(0.5, 0.9))
optim_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, betas=(0.5, 0.9))
scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(optim_autoencoder, 30, gamma=0.5)
    
all_loss_data = []
all_loss_model = []
all_loss_prior = []

for iter in range(ITERATION):
    loss_data, loss_prior, m, autoencoder  = train_fun(
        Physics=Physics, 
        autoencoder=autoencoder,
        d_obs=d_obs, 
        freqs=INV_FREQS,
        optim_autoencoder=optim_autoencoder, 
        criteria=criteria,
        mini_batches = MINI_BATCHES,
        src_loc=src_loc, 
        rec_loc=rec_loc, 
        src=src,
        inpa=inpa, 
        lam_prior=LAM_PRIOR,
        well_locations=well_locations,
        well_data=well_data,
        test=None)
    
    all_loss_data.append(loss_data)
    all_loss_prior.append(loss_prior) 
    
    with torch.no_grad():
        all_loss_model.append(
            criteria(m, vp).item()
        )
    if iter%PRINT_FREQ == 0:
        print(f"Iteration {iter + 1} ===== loss: {all_loss_data[-1]} for data and {all_loss_model[-1]} for model")
    if iter%SAVE_FREQ == 0:
        save_results.numpy(np.array(m.cpu().detach()), file_name=f"m_{iter}")
    #     plt.figure()
    #     plt.imshow(np.array(m.cpu().detach()), cmap="jet")
    scheduler_autoencoder.step()

m = np.array(m.cpu().detach())

#%% Saving the results
save_results.numpy(all_loss_data, file_name="all_loss_data")
save_results.numpy(all_loss_prior, file_name="all_loss_prior")
save_results.numpy(all_loss_model, file_name="all_loss_model")
save_results.numpy(m, file_name="m")
save_results.network(model=autoencoder, file_name="autoencoder")
#%% Show results
plt.figure()
plt.imshow(m, cmap="jet")
plt.plot([well_locations, well_locations], 
         [0, depth], "r")
plt.colorbar()

plt.figure()
plt.plot(all_loss_data)
plt.show(block=False)

plt.figure()
plt.plot(all_loss_model)
plt.show(block=True)

