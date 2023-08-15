from config import *
# export CXX=/usr/local/opt/llvm/bin/clang

SAVE = 0

vp, _ = earth_model(MODEL, smooth=10, device=DEVICE)
vs = torch.zeros(1)
rho = torch.ones(1)

if not SAVE:
    fig = show_earth_model({"Vp": vp}, cmap="jet")
    fig.axes[0].plot(rec_loc_temp[:,0]/DH, rec_loc_temp[:, 1]/DH, 'k*', markersize=1)
    fig.axes[0].plot(src_loc_temp[:,0]/DH, src_loc_temp[:, 1]/DH, 'rv', markersize=4)


out = deepwave.scalar(vp, DH, DT,
                      source_amplitudes=src,
                      source_locations=src_loc,
                      receiver_locations=rec_loc)


taux = (out[-1].cpu()).permute(0, 2, 1)
ns, nt, nr = taux.shape
print(f"Number of \n{'-'*10}")
print(f"sources: {ns}\nreceivers: {nr}\ntime samples: {nt}")

vmin, vmax = torch.quantile(taux[N_SHOTS//2],
                            torch.tensor([0.01, 0.99]))
if SAVE:
    torch.save(taux, 
               f= PATH + "/data_model/taux_obs_" + PACKAGE + "_" + MODEL + "_" + str(N_SHOTS))
else:
    
    plt.figure()
    plt.imshow(taux[N_SHOTS//2], aspect='auto', cmap='gray',
            vmin=vmin, vmax=vmax)
    plt.xlabel("Receiver")
    plt.ylabel("Time sample")
    plt.show()

