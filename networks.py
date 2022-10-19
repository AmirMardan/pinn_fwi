import torch.nn as nn
import torch
from utils import *

# from torchfwi import Fwi
# from PyFWI.torchfwi import Fwi


class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SubBlock, self).__init__()
        self.conv = self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=stride, padding=1,
                      bias=True,
                    #   padding_mode="reflect"  # Help to reduce the artifacts
                      ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.conv(x)
        
        
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, operation, final_shape=None):
        super(Block, self).__init__()
        layers = [
            SubBlock(in_channels=in_channels,
                              out_channels=out_channels,
                              stride=stride),
            SubBlock(in_channels=out_channels,
                              out_channels=out_channels,
                              stride=stride)
            ]
        if operation == "down":
            layers.append(
                nn.MaxPool2d(kernel_size=2)
            )
        elif operation == "up":
            if not final_shape:
                layers.append(
                    nn.Upsample(scale_factor=2)
                )
            else:
                layers.append(
                    nn.Upsample(final_shape)
                )
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.layers(x)
        return out
        
        
class Encoder(nn.Module):
    def __init__(self, batch_size, in_channels,
                 n_blocks, nt, nr,
                 final_size):
        super(Encoder, self).__init__()
            
        
        self.batch_size = batch_size
        
        layers = []
        self.out_channels = [2 ** i for i in range(n_blocks + 1)]
        # print(self.out_channels)
        
        for layer_idx in range(n_blocks):
            layers.append(
                Block(in_channels=self.out_channels[layer_idx] * in_channels,
                      out_channels=self.out_channels[layer_idx + 1] * in_channels,
                      stride=1, operation="down"
                      )
                )
        self.conv_layers = nn.Sequential(*layers)
    
        fc_in_features = batch_size * in_channels * self.out_channels[-1] \
                    * torch.div(nt, self.out_channels[-1], rounding_mode="floor")\
                    * torch.div(nr, self.out_channels[-1], rounding_mode="floor")
        # print(fc_in_features, final_size) 
        # Fully connected layer is used to bring the results of conv layers to
        # an pproopriate size for upsacling. 
        self.final = nn.Sequential(
            nn.Linear(
                in_features=fc_in_features, 
                out_features=final_size 
                    )
        )
        
    def forward(self, x):
        out = self.conv_layers(x)
        # out = nn.Flatten()(out)
        out = out.view(-1)
        # print(out.shape)
        out = self.final(out)
        
        return out
        
        
class Decoder(nn.Module):
    def __init__(self, batch_size,
                 initial_shape, final_shape,
                 n_blocks,
                 m_min, m_max,
                 final_out_channels=1
                 ):
        
        super(Decoder, self).__init__() 
        #Shape to reconstruct output of the encoder 
        self.initial_shape = initial_shape
        self.batch_size = batch_size
        self.m_max = m_max
        self.m_min = m_min
        finalize = None
        
        layers = []
        self.out_channels = [2 ** i for i in range(n_blocks + 1)]
        
        for layer_idx in range(n_blocks):
            finalize = final_shape if  layer_idx== n_blocks-1 else None
                        
            layers.append(
                Block(in_channels=self.out_channels[layer_idx],
                      out_channels=self.out_channels[layer_idx + 1],
                      stride=1, operation="up",
                      final_shape=finalize
                      )
                )
            
        self.conv_layers = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels[-1],
                      out_channels=final_out_channels, 
                      kernel_size=3, padding=1, stride=1,
                      bias=True
                      ),
            nn.Sigmoid(),
            
            )
    def forward(self, x):
        x = x.reshape(self.batch_size, 1, self.initial_shape[0], self.initial_shape[1])
        out = self.conv_layers(x)
        out = self.final(out)
        out = self.m_min + out * (self.m_max - self.m_min)
        return out
    

class Autoencoder(nn.Module):
    def __init__(self, batch_size, in_channels,
                 n_blocks_encoder, n_blocks_decoder,
                 final_size_encoder, initial_shape_decoder,
                 nt, nr, final_spatial_shape,
                 m_min, m_max,
                 final_out_channels=1 
                 ):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(batch_size=batch_size, in_channels=in_channels,
                               n_blocks=n_blocks_encoder, 
                               nt=nt, nr=nr, final_size=final_size_encoder)
        self.decoder = Decoder(batch_size=batch_size, initial_shape=initial_shape_decoder,
                               final_shape=final_spatial_shape, n_blocks=n_blocks_decoder, 
                               m_min=m_min, m_max=m_max, 
                               final_out_channels=final_out_channels)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    

class Physics_deepwave(nn.Module):
    def __init__(self, dh, dt, src,
                 src_loc, rec_loc):
        super(Physics_deepwave, self).__init__()
        self.dh = dh
        self.dt = dt
        self.src = src
        self.src_loc = src_loc
        self.rec_loc = rec_loc
    
    def forward(self, vp):
        out = deepwave.scalar(vp, self.dh, self.dt,
                      source_amplitudes=self.src,
                      source_locations=self.src_loc,
                      receiver_locations=self.rec_loc)
        taux = out[-1]
        return taux.permute(0, 2, 1).unsqueeze(0)
           

class Physics_pyfwi(nn.Module):
    def __init__(self, inpa , src, rec_loc, f_show=False, b_show=False):
        super(Physics_pyfwi, self).__init__()
                
        self.b_show = b_show
        self.f_show = f_show
        self.src = src
        self.rec_loc = rec_loc
        self.inpa = inpa
        # To use less ram, I make the size 1, and torch uses 
        self.vs = torch.zeros(1)
        self.rho = torch.ones(1)
    
    def forward(self, vp):
        
        _, _, taux, tauz, _ = Fwi.apply(self.inpa, vp, self.vs, self.rho,
                               self.src, self.rec_loc, self.f_show, self.b_show)
        # TODO:  Normalize 
        return taux.permute(2, 0, 1).unsqueeze(0) # data_normalization(taux.permute(2, 0, 1)).unsqueeze(0)
    
       
# class Physics_pyfwi(nn.Module):
#     def __init__(self, wave, b_show=False, f_show=False):
#         super(Physics_pyfwi, self).__init__()
#         nz = wave.nz
#         nx = wave.nx 
        
#         self.b_show = b_show
#         self.f_show = f_show
        
#         self.w = wave
#         # To use less ram, I make the size 1, and torch uses 
#         self.vs = torch.zeros(1)
#         self.rho = torch.ones(1)
    
#     def forward(self, vp):
        
#         taux, tauz = Fwi.apply(self.w, vp, self.vs, self.rho,
#                                self.f_show, self.b_show)
        
#         return data_normalization(taux.permute(2, 0, 1)).unsqueeze(0)
        
         
if __name__ == "__main__":
    import torch 
    nb = 64
    nc = 6
    nz= 90
    nt = 150
    nx = 100
    
    data = torch.rand(nb, nc, nz, nx)
    sub_block = SubBlock(in_channels=nc,
                         out_channels=2*nc,
                         stride=1)
    data_shape = data.shape
    print(f"Shape of data is: {data_shape}")
    
    sub_block_data_shape = sub_block(data).shape
    assert sub_block_data_shape == (data_shape[0], data_shape[1]*2, data_shape[2], data_shape[3]),\
        f"SubBlock must only halves the number of channels! So, size should be {(data_shape[0], data_shape[1]*2, data_shape[2], data_shape[3])}, "\
        f"but got {sub_block_data_shape}"
        