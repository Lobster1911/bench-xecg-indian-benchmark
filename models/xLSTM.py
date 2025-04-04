import torch
from torch import nn

from models.utils import get_activation_fn, get_xlstm, get_large_xlstm, get_patch_embedding, get_reconstruction_head
from models.modules import HeadModule
from models.SeriesDecomposition import SeriesDecomposition 
from augmentations import RandomDropLeads, FTSurrogate, Jitter
import numpy as np

class myxLSTM(nn.Module):

    def __init__(
            self, 
            num_channels,
            config
        ): 
        super(myxLSTM, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.patch_size = config.patch_size
        self.weight_tying = config.weight_tying
        self.bidirectional = config.bidirectional

        self.activation = get_activation_fn(config.activation_fn)

        self.patch_embedding = get_patch_embedding(config.patch_embedding, config.patch_size, config.embedding_size, num_channels)

        xlstm_emb_size = config.embedding_size
        if config.xlstm_type == 'large':
            self.xlstm = get_large_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads)
        else:
            self.xlstm = get_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads)

        if self.bidirectional:
            if config.xlstm_type == 'large':
                self.xlstm_bi = get_large_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads)
            else:
                self.xlstm_bi = get_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads)            

        self.random_drop_leads = RandomDropLeads(config.random_drop_leads)
        self.random_surrogate = FTSurrogate(0.05, prob=config.random_surrogate_prob)
        self.random_jitter = Jitter(sigma=0.1, prob=config.random_jitter_prob)

        emb_size = config.embedding_size if not self.bidirectional else config.embedding_size * 2

        self.reconstruction = get_reconstruction_head(config.reconstruct_embedding, config.patch_size, emb_size, num_channels)

        if self.weight_tying and config.patch_embedding == 'linear' and config.reconstruct_embedding == 'linear': 
            self.reconstruction.deconv.weight = self.patch_embedding.conv.weight


    def embed_data(self, x, augment=True):
        if augment:
            x = self.random_drop_leads(x)
            x = self.random_surrogate(x)
            x = self.random_jitter(x)

        x = x.permute(0, 2, 1) # put the channels in the middle
        x = self.patch_embedding(x)
        
        return x

    def forward(self, x):
        x = self.embed_data(x, augment=False)

        out = self.xlstm(x)

        if self.bidirectional:
            out = self.get_bidirectional_emb(x, out)

        out = self.reconstruction(out)
        return out
    
    def get_bidirectional_emb(self, x, out):
        out_bi = self.xlstm_bi(x.flip(1))

        out_bi = torch.cat([out_bi.flip(1)[:, 2:, :], torch.zeros(out_bi.shape[0], 2, out_bi.shape[2]).to(out_bi.device)], dim=1)
        out = torch.cat([out, out_bi], dim=-1)
        return out
    
    def generate(self, x, length=10):

        # i do not need to drop the leads here
        x = self.embed_data(x, augment=False)

        state = None
        for i in range(x.shape[1]):
            new_x, state = self.xlstm.step(x[:, i].unsqueeze(1), state=state)

        if self.bidirectional:
            new_x = torch.cat([new_x, torch.zeros_like(new_x)], dim=-1)

        r = self.reconstruction(new_x)

        reconstructed = [r]

        for i in range(length - 1):
            x = self.embed_data(r, augment=False)
            x, state = self.xlstm.step(x, state=state)

            if self.bidirectional:
                x = torch.cat([x, torch.zeros_like(x)], dim=-1)

            r = self.reconstruction(x)

            reconstructed.append(r)
        
        return torch.cat(reconstructed, dim=1)


    def trainable_parameters(self):
        return self.parameters()
    
    def head_parameters(self):
        return self.fc.parameters() 
    

class xLSTMClassificationMIT_BIH(myxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels
        ): 

        super(xLSTMClassificationMIT_BIH, self).__init__(num_channels, config)

        self.fc = HeadModule(
            inp_size=config.embedding_size if not self.bidirectional else config.embedding_size * 2,
            hidden_size=config.embedding_size // 2,
            out_size=num_classes,
            dropout=config.dropout
        )

        self.r_peak_pos_fc = HeadModule(
            inp_size=config.embedding_size if not self.bidirectional else config.embedding_size * 2,
            hidden_size=config.embedding_size // 2,
            out_size=self.patch_size,
            dropout=config.dropout
        )

    def forward(self, x):
        x = self.embed_data(x)

        out = self.xlstm(x) # [batch_size, embedding_dim]

        if self.bidirectional:
            out_bi = self.xlstm_bi(x.flip(1))
            cls = torch.cat([cls, out_bi[:, -1, :]], dim=-1)

        cls = self.fc(out)
        r_peak_pos = self.r_peak_pos_fc(out)
        return cls, r_peak_pos

