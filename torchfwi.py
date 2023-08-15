import numpy as np
import torch 
import pyfwi_tools
# import PyFWI.processing as process
from pyfwi_tools import lowpass, adj_lowpass


def data2d_to_3d(data1_2d, data2_2d, ns, nr):
    nt = data1_2d.shape[0]
    
    data1_3d = torch.empty((ns, nt, nr))
    data2_3d = torch.empty((ns, nt, nr))
    
    for i in range(ns):
        data1_3d[i, :, :] = data1_2d[:, i*nr:(i+1)*nr]
        data2_3d[i, :, :] = data2_2d[:, i*nr:(i+1)*nr]
    return data1_3d, data2_3d


def data3d_to_2d(data1_3d, data2_3d):
    ns, nt, nr = data2_3d.shape
    x1_2d = torch.empty((nt, ns*nr))
    x2_2d = torch.empty((nt, ns*nr))
    for i in range(ns):
        x1_2d[:, i*nr:(i+1)*nr] = data1_3d[i, ...]
        x2_2d[:, i*nr:(i+1)*nr] = data2_3d[i, ...]
        
    return x1_2d, x2_2d

            
def lpass(x1, x2, highcut, fn):
    x1_filtered, x2_filtered = Lfilter.apply(x1, x2, highcut, fn)
    return x1_filtered, x2_filtered


class Lfilter(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x1, x2, highcut, fn):
        ctx.lpass_highcut = highcut
        ctx.lpass_fn = fn
        
        nb, ns, nt, nr = x1.shape
        device = x1.device.type
        
        x1_np = x1.detach()
        x2_np = x2.detach()
        
        x1_np = x1_np.squeeze(dim=0) #.numpy()
        x2_np = x2_np.squeeze(dim=0) #.numpy()
        
        x1_np, x2_np = data3d_to_2d(x1_np , x2_np)
        
        x1_np = torch.unsqueeze(x1_np, 0)
        x2_np = torch.unsqueeze(x2_np, 0)
        
        filtered1 = lowpass(x1_np.numpy(), highcut=highcut, fn=fn,
                           order=3, axis=1)
        
        filtered2 = lowpass(x2_np.numpy(), highcut=highcut, fn=fn,
                           order=3, axis=1)
        
        filtered1_3d, filtered2_3d = data2d_to_3d(
            torch.Tensor(filtered1[0, ...]),
            torch.Tensor(filtered2[0, ...]),
            ns, nr)
        
        # filtered1 = torch.tensor(filtered1_3d, device=device)
        # filtered2 = torch.tensor(filtered2_3d, device=device)
                   
        return filtered1_3d.unsqueeze(0).to(device=device), filtered2_3d.unsqueeze(0).to(device=device)
    
    @staticmethod
    def backward(ctx, adj1, adj2):
        
        nb, ns, nt, nr = adj1.shape
        device = adj1.device.type
        
        x1_np = adj1.detach()
        x2_np = adj2.detach()
        
        x1_np = x1_np.squeeze(dim=0) # .numpy()
        x2_np = x2_np.squeeze(dim=0) # .numpy()
        
        x1_np, x2_np = data3d_to_2d(x1_np, x2_np)
        x1_np = torch.unsqueeze(x1_np, 0)
        x2_np = torch.unsqueeze(x2_np, 0)
        
        filtered1 = adj_lowpass(x1_np.numpy(), highcut=ctx.lpass_highcut,
                                fn=ctx.lpass_fn, order=3, axis=1)
        
        filtered2 = adj_lowpass(x2_np.numpy(), highcut=ctx.lpass_highcut,
                                fn=ctx.lpass_fn, order=3, axis=1)
        
        filtered1_3d, filtered2_3d = data2d_to_3d(
            torch.Tensor(filtered1[0, ...]),
            torch.Tensor(filtered2[0, ...]),
            ns, nr)
        
        # filtered1 = torch.tensor(filtered1_3d, device=device)
        # filtered2 = torch.tensor(filtered2_3d, device=device)
                   
        return filtered1_3d.unsqueeze(0).to(device=device), \
                filtered2_3d.unsqueeze(0).to(device=device),\
                    None,\
                    None
    

