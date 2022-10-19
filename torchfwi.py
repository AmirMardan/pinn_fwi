import numpy as np
import torch 
import PyFWI.processing as process
import PyFWI.wave_propagation as wave


class Fwi(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, inpa, vp, vs, rho,
                src, rec_loc, f_show=False, b_show=False):
        
        ctx.b_show = b_show
        model_shape = vp.shape
        model = {'vp': vp.cpu().numpy().astype(np.float32),
                 'vs': vs.cpu().numpy().astype(np.float32),
                 'rho': rho.cpu().numpy().astype(np.float32)
                 }
        
        w = wave.WavePropagator(inpa, src, rec_loc, model_shape,
                            n_well_rec=inpa["n_well_rec"], chpr=inpa["chpr"],
                            components=4
                            )
        # Call the forward modelling
        db_obs = w.forward_modeling(model, show=f_show)  # show=True can show the propagation of the wave

        
        ctx.WavePropagator = w
        
        return  torch.tensor(db_obs['vx']), torch.tensor(db_obs['vz']), \
                torch.tensor(db_obs['taux']), torch.tensor(db_obs['tauz']),\
                torch.tensor(db_obs['tauxz'])
                
    @staticmethod   
    def backward(ctx, adj_vx, adj_vz, adj_tx, adj_tz, adj_txz):
        """
        backward calculates the gradient of cost function with
        respect to model parameters

        Parameters
        ----------
        adj_tx : Tensor
            Adjoint of recoreded normal stress in x-direction
        adj_tz : Tensor
            Adjoint of recoreded normal stress in z-direction

        Returns
        -------
        Tensor
            Gradient of cost function w.r.t. model parameters
        """
        adj_vx = adj_vx.cpu().detach().numpy()
        adj_vz = adj_vz.cpu().detach().numpy()
        adj_taux = adj_tx.cpu().detach().numpy()
        adj_tauz = adj_tz.cpu().detach().numpy()
        adj_tauxz = adj_txz.cpu().detach().numpy()
        
        # If shape is (nt, nr, ns) switch it to (nt, nr * ns)
        if adj_tauz.ndim == 3:
            (nt, nr, ns) = adj_taux.shape
            adj_vx = np.reshape(adj_vx, (nt, nr * ns), 'F')
            adj_vz = np.reshape(adj_vz, (nt, nr * ns), 'F')
            adj_taux = np.reshape(adj_taux, (nt, nr * ns), 'F')
            adj_tauz = np.reshape(adj_tauz, (nt, nr * ns), 'F')
            adj_tauxz = np.reshape(adj_tauxz, (nt, nr * ns), 'F')
            
        adj_src = process.prepare_residual(
            {
                'vx':adj_vx,
                'vz': adj_vz,
                'taux': adj_taux,
                'tauz': adj_tauz,
                'tauxz': adj_tauxz,
             }, 1)
                
        result = ctx.WavePropagator.gradient(adj_src, ctx.b_show)
        
        return (None, 
                torch.tensor(result['vp']), 
                torch.tensor(result['vs']), 
                torch.tensor(result['rho']),
                None,
                None,
                None,
                None
                )