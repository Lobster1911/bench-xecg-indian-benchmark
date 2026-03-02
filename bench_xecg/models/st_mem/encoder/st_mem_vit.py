# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# Modified work for BenchECG analysis. Copyright (c) 2025 Dlaska Lab - Digital Cardiology. <https://github.com/dlaskalab/bench-xecg>

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from ..encoder.vit import TransformerBlock
from ...base_model import BaseModel


__all__ = ['ST_MEM_ViT', 'st_mem_vit_small', 'st_mem_vit_base']


class ST_MEM_ViT(BaseModel):
    def __init__(
            self,
            seq_len: int,
            patch_size: int,
            num_leads: int,
            num_classes: Optional[int] = None,
            width: int = 768,
            depth: int = 12,
            mlp_dim: int = 3072,
            heads: int = 12,
            dim_head: int = 64,
            qkv_bias: bool = True,
            drop_out_rate: float = 0.,
            attn_drop_out_rate: float = 0.,
            drop_path_rate: float = 0.,
            linear_probing: bool = False,
            feature_classification: bool = False,
            r_peaks_detection: bool = False,
            sleep_apnea: bool = False,
            window_size: int = 60,
            context_size: int = 0,
            ):
        
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        self._repr_dict = {
            'seq_len': seq_len,
            'patch_size': patch_size,
            'num_leads': num_leads,
            'num_classes': num_classes if num_classes is not None else 'None',
            'width': width,
            'depth': depth,
            'mlp_dim': mlp_dim,
            'heads': heads,
            'dim_head': dim_head,
            'qkv_bias': qkv_bias,
            'drop_out_rate': drop_out_rate,
            'attn_drop_out_rate': attn_drop_out_rate,
            'drop_path_rate': drop_path_rate,
            'linear_probing': linear_probing,
            'feature_classification': feature_classification,
            'r_peaks_detection': r_peaks_detection,
            'sleep_apnea': sleep_apnea,
            'window_size': window_size,
            'context_size': context_size
        }

        self.width = width
        self.depth = depth
        self.linear_probing = linear_probing
        self.patch_size = patch_size
        self.feature_classification = feature_classification
        self.r_peaks_detection = r_peaks_detection
        self.sleep_apnea = sleep_apnea
        self.sampling_freq = 250
        self.context_size = context_size
        self.window_size = window_size


        # embedding layers
        num_patches = seq_len // patch_size
        patch_dim = patch_size
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (n p) -> b c n p', p=patch_size),
                                                nn.LayerNorm(patch_dim),
                                                nn.Linear(patch_dim, width),
                                                nn.LayerNorm(width))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width))
                                                for _ in range(num_leads))

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width,
                                     output_dim=width,
                                     hidden_dim=mlp_dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rate_list[i])
            self.add_module(f'block{i}', block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        # classifier head
        self.head = nn.Identity() if num_classes is None else nn.Sequential(
            nn.Linear(width, num_classes)
        )

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, series):
        series = series.transpose(1, 2)
        num_leads = series.shape[1]
        if num_leads > len(self.lead_embeddings):
            raise ValueError(f'Number of leads ({num_leads}) exceeds the number of lead embeddings')
        
        if series.shape[2] % self.patch_size != 0:
            series = torch.nn.functional.pad(series, (0, self.patch_size - (series.shape[2] % self.patch_size)))

        x = self.to_patch_embedding(series)
        b, _, n, _ = x.shape
        # print(f'Features shape after patch embedding: {x.shape}')
        
        # cut the signal if needed
        if n >= self.pos_embedding.shape[1]:
            x = x[:, :, :self.pos_embedding.shape[1] -1, :]

        n = x.shape[2]
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.lead_embeddings]).unsqueeze(0)
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
        x = x + lead_embeddings
        x = rearrange(x, 'b c n p -> b (c n) p')

        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        # remove SEP embeddings
        x = rearrange(x, 'b (c n) p -> b c n p', c=num_leads)
        x = x[:, :, 1:-1, :]

        if self.feature_classification:
            return x
        
        x = torch.mean(x, dim=(1, 2))
        return self.norm(x)

    def forward(self, series):
        series = series.float()
        if self.linear_probing:
            with torch.no_grad():
                x = self.forward_encoding(series)
        else:
            x = self.forward_encoding(series)

        # print(f'Features shape before head: {x.shape}')

        if self.feature_classification:
            if self.r_peaks_detection or self.sleep_apnea:
                # print('before getting the channel: ', x.shape)
                # get only the second lead for r-peaks detection
                x = x[:, 1, :, :]
                # print('after getting the channel: ', x.shape)
            else:
                x = x.mean(dim=1) 

            if self.sleep_apnea and self.context_size > 0:
                # remove the context patches from the features   
                context_patches = (self.context_size * self.sampling_freq) // self.patch_size
                window_patches = (self.window_size * self.sampling_freq) // self.patch_size
                start = context_patches
                end = context_patches + window_patches
                x = x[:, start:end, :]

            out = self.head(x)
            # print(f"out shape: {out.shape}")  # Debugging output

            return out
        return self.head(x)
    
    def aggregate_per_minute(self, features):
        # print(f'Input features shape: {features.shape}')
        patches_per_segment = (60 * self.sampling_freq) // self.patch_size
        # print(f'Aggregating features per minute with {patches_per_segment} patches per minute')
        # print(f'Input features shape: {features.shape}')
        n_minutes = int(features.size(1)) // patches_per_segment  

        features = features.reshape(
            features.shape[0],
            n_minutes,
            patches_per_segment,
            features.shape[-1]
        )
        features = features.mean(dim=2)
        return features

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str
    
    def get_layers(self):
        return [self.__getattr__(f'block{i}') for i in range(self.depth)]
    
    def additional_params(self, lr, last_layer_lr, wd):
        params = []
        params.append({'params': self.to_patch_embedding.parameters(), 'lr': last_layer_lr, 'name': 'to_patch_embedding'})
        params.append({'params': self.pos_embedding, 'lr': last_layer_lr, 'name': 'pos_embedding'})
        params.append({'params': self.sep_embedding, 'lr': last_layer_lr, 'name': 'sep_embedding'})
        params.append({'params': self.lead_embeddings.parameters(), 'lr': last_layer_lr, 'name': 'lead_embeddings'})
        params.append({'params': self.norm.parameters(), 'lr': lr, 'name': 'ln'})
        return params



def st_mem_vit_small(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=384,
                      depth=12,
                      heads=6,
                      mlp_dim=1536,
                      **kwargs)
    return ST_MEM_ViT(**model_args)


def st_mem_vit_base(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=768,
                      depth=12,
                      heads=12,
                      mlp_dim=3072,
                      **kwargs)
    return ST_MEM_ViT(**model_args)