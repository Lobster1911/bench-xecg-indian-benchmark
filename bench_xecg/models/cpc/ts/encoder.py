

from dataclasses import dataclass, field
from typing import List

import torch
import numpy as np

from .basic_conv1d_modules.basic_conv1d import _conv1d

@dataclass
class RNNEncoderConfig:
    _target_:str = "cpc.ts.encoder.RNNEncoder"
    timesteps_per_token: int = 1 # timesteps per token a la vision transformer
    input_format_seq_last: bool = True # seq axis last e.g. bs, ch, seq instead of second e.g. bs, seq, ch

    #local pool after first conv
    multi_prediction:bool = False #local_pool named like this for consistency with MLP heads etc
    local_pool_max:bool = False
    local_pool_kernel_size: int = 0
    local_pool_stride: int = 0 #kernel_size if 0
    
    strides:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder strides (space-separated)")
    kss:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder kernel sizes (space-separated)")
    features:List[int]=field(default_factory=lambda: [512,512,512,512]) #help="encoder features (space-separated)")
    dilations:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder dilations (space-separated)")
    normalization:bool=True #help="disable encoder batch/layer normalization")
    layer_norm:bool=False#", action="store_true", help="encoder layer normalization")


class RNNEncoder(torch.nn.Module):
    def __init__(self, hparams_encoder: RNNEncoderConfig, hparams_input_shape: dict, static_stats_train=None):
        '''RNN Encoder is actually just a conv encoder'''
        super().__init__()
        assert(len(hparams_encoder.strides)==len(hparams_encoder.kss) and len(hparams_encoder.strides)==len(hparams_encoder.features) and len(hparams_encoder.strides)==len(hparams_encoder.dilations))

        self.hparams = hparams_encoder
        self.input_shape = hparams_input_shape

        lst = []
        for i,(s,k,f,d) in enumerate(zip(hparams_encoder.strides,hparams_encoder.kss,hparams_encoder.features,hparams_encoder.dilations)):
            lst.append(_conv1d((hparams_input_shape["channels"]*hparams_input_shape["channels2"] if hparams_input_shape["channels2"]>0 else hparams_input_shape["channels"]) if i==0 else hparams_encoder.features[i-1],f,kernel_size=k,stride=s,dilation=d,bn=hparams_encoder.normalization,layer_norm=hparams_encoder.layer_norm))
            if(hparams_encoder.multi_prediction and i==0):#local pool after first conv
                if(hparams_encoder.local_pool_max):
                    lst.append(torch.nn.MaxPool1d(kernel_size=hparams_encoder.local_pool_kernel_size,stride=hparams_encoder.local_pool_stride if hparams_encoder.local_pool_stride!=0 else hparams_encoder.local_pool_kernel_size,padding=(hparams_encoder.local_pool_kernel_size-1)//2))
                else:
                    lst.append(torch.nn.AvgPool1d(kernel_size=hparams_encoder.local_pool_kernel_size,stride=hparams_encoder.local_pool_stride if hparams_encoder.local_pool_stride!=0 else hparams_encoder.local_pool_kernel_size,padding=(hparams_encoder.local_pool_kernel_size-1)//2))        
        
        self.layers = torch.nn.Sequential(*lst)
        self.downsampling_factor = (hparams_encoder.local_pool_stride if hparams_encoder.multi_prediction else 1)*np.prod(hparams_encoder.strides)
        
        self.timesteps_per_token = hparams_encoder.timesteps_per_token
        self.sequence_last = hparams_input_shape["sequence_last"]
        self.output_dim = hparams_encoder.features[-1]

        self.output_shape = {k: v for k, v in hparams_input_shape.items()}
        self.output_shape["channels"] = self.output_dim
        self.output_shape["channels2"] = 0
        self.output_shape["length"] = int(hparams_input_shape["length"]//self.downsampling_factor+ (1 if hparams_input_shape["length"]%self.downsampling_factor>0 else 0))
        self.output_shape["sequence_last"] = False


    def forward(self, x):
        if(not self.sequence_last):
            x = torch.movedim(x,1,-1)
        if(len(x.size())==4):#spectrogram input
            x = x.view(x.size(0),-1,x.size(-1))#flatten
        if(self.timesteps_per_token > 1):#patches a la vision transformer
            assert(x.size(2)%self.timesteps_per_token==0)
            size = x.size()
            x = x.transpose(1,2).reshape(size[0],size[2]//self.timesteps_per_token,-1).transpose(1,2) # output: bs, output_dim, seq//downsampling_factor
        return self.layers(x).transpose(1,2) # bs,seq,feat
    
    def get_output_shape(self):
        return self.output_shape



