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
                    nn.Upsample(scale_factor=2, mode="bilinear")
                )
            else:
                layers.append(
                    nn.Upsample(final_shape, mode="bilinear")
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
        self.nr = nr
        self.nt = nt
        
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
    
        self._set_fc_in_features(in_channels)
        # print(fc_in_features, final_size) 
        # Fully connected layer is used to bring the results of conv layers to
        # an pproopriate size for upsacling. 
        self.final_size = final_size
        
        self.final = nn.Sequential(
            nn.Linear(
                in_features=self.fc_in_features, 
                out_features=self.final_size 
                    )
        )
        
    def _set_fc_in_features(self, in_channels):
        self.fc_in_features =self.batch_size * in_channels * self.out_channels[-1] \
                    * torch.div(self.nt, self.out_channels[-1], rounding_mode="floor")\
                    * torch.div(self.nr, self.out_channels[-1], rounding_mode="floor")
        
    def _reshape_input(self, in_channels:int):
        input_layer = self.conv_layers[0].layers[0].conv[0] 
        stride = input_layer.stride
        out_channels = input_layer.out_channels
        
        # self._set_fc_in_features(in_channels)
        old_block = self.conv_layers[0].layers[0].conv[0]
        if old_block.in_channels == in_channels:
            pass
        else:
            self.conv_layers[0].layers[0].conv[0] = \
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=3, stride=stride, padding=1,
                        bias=True,
                        )
            self.conv_layers[0].layers[0].conv[0].bias = nn.Parameter(old_block.bias)
            if old_block.in_channels > in_channels:
                self.conv_layers[0].layers[0].conv[0].weight = nn.Parameter(old_block.weight[:,:in_channels, ...])
                
            else:
                self.conv_layers[0].layers[0].conv[0].weight[:, :in_channels, ...] = nn.Parameter(old_block.weight)
                
    def _reshape_final(self, 
                      fc_in_features: int,
                      final_size: int):
        old_in_features = self.final[0].in_features
        if old_in_features == fc_in_features:
            pass
        else:
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
                 initial_shape: Tuple, 
                 final_shape: Tuple,
                 n_blocks,
                 m_min, m_max,
                 final_out_channels=1
                 ):
        
        super(Decoder, self).__init__() 
        #Shape to reconstruct output of the encoder
        self._set_initial_shape(initial_shape) 
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

    def _set_initial_shape(self, initial_shape_decoder: Tuple):
        self.initial_shape = initial_shape_decoder
    
    def _set_finale_shape(self, final_shape_decoder: Tuple):
        self.conv_layers[-1].layers[2].size = final_shape_decoder
        
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
    
    def _reshape_encoder(self, trained_encoder, d_obs: torch.Tensor):
        new_in_channels: int = d_obs.shape[1]
        
        final_size_encoder = self.encoder.final_size
        
        self.encoder = trained_encoder
        self.encoder._reshape_input(in_channels=new_in_channels)

        a = self.encoder.conv_layers(d_obs)
        self.encoder._reshape_final(fc_in_features=a.view(-1).shape[0],
                                  final_size=final_size_encoder)
        
    def _reshape_decoder(self, trained_decoder, 
                        decoder_initial_shape: Tuple,
                        decoder_final_shape: Tuple):
        
        self.decoder = trained_decoder
        self.decoder._set_initial_shape(decoder_initial_shape)
        self.decoder._set_finale_shape(decoder_final_shape)
    
    def reshape(self, trained_autoencoder,
                d_obs: torch.Tensor,
                decoder_initial_shape: Tuple,
                decoder_final_shape: Tuple
                ):
        self._reshape_encoder(
            trained_autoencoder.encoder,
            d_obs=d_obs)
    
        self._reshape_decoder(
            trained_autoencoder.decoder,
            decoder_initial_shape=decoder_initial_shape,
            decoder_final_shape=decoder_final_shape
            )

    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    

class Physics_deepwave(nn.Module):
    def __init__(self, dh, dt, src,
                 src_loc, rec_loc, rp_properties=None):
        super(Physics_deepwave, self).__init__()
        self.dh = dh
        self.dt = dt
        self.src = src
        self.src_loc = src_loc
        self.rec_loc = rec_loc
        rp_properties = rp_properties
    
    def forward(self, vp):
        
        out = deepwave.scalar(vp, self.dh, self.dt,
                      source_amplitudes=self.src,
                      source_locations=self.src_loc,
                      receiver_locations=self.rec_loc)
        taux = out[-1]
        return taux.permute(0, 2, 1).unsqueeze(0)
    

class Physics_cufwi(nn.Module):
    def __init__(self, dh, dt, src,
                 src_loc, rec_loc, rp_properties=None):
        super(Physics_cufwi, self).__init__()
        
        src_loc = src_loc.squeeze()
        temp = src_loc[:, 0].clone()
        src_loc[:, 0] = src_loc[:, 1]
        src_loc[:, 1] = temp
        src_loc = src_loc.squeeze()

        rec_loc = rec_loc[0, ...]
        temp = rec_loc[:, 0].clone()
        rec_loc[:, 0] = rec_loc[:, 1]
        rec_loc[:, 1] = temp

        self.dh = dh
        self.dt = dt
        # src = src.to(device="cpu")
        self.src = src[0, 0, :].to(device="cpu")
        self.src_loc = src_loc.to(device="cpu")
        self.rec_loc = rec_loc.to(device="cpu")
        rp_properties = rp_properties
    
    def forward(self, vp):
        print(self.src_loc)
        wave = AcousticWave(grid_spacing=self.dh, 
                            dt=self.dt, accuracy=4,
                            source_amplitude=self.src, 
                            src_loc=self.src_loc,
                            rec_loc=self.rec_loc,
                            pml_width=20, chpr=99, 
                            save_wave=False)
        out = wave(vp.to(device="cpu"))

        taux = out[-1]
        return taux.permute(2, 0, 1).unsqueeze(0)
           
           
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
        