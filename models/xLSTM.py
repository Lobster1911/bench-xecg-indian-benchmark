import torch
from torch import nn

from models.utils import get_activation_fn, get_xlstm, get_large_xlstm, get_patch_embedding, get_reconstruction_head
from models.modules import HeadModule
from models.SeriesDecomposition import SeriesDecomposition 
from augmentations import RandomDropLeads, FTSurrogate, Jitter
import numpy as np
import torch.nn.functional as F

class pretrainedxLSTM(nn.Module):
    def __init__(
            self, 
            num_channels,
            config
        ): 
        super(pretrainedxLSTM, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.patch_size = config.patch_size
        self.weight_tying = config.weight_tying
        self.bidirectional = config.bidirectional
        self.training_strategy = config.strategy
        self.use_teacher_student = config.use_teacher_student
        self.ema_0 = config.ema_0
        self.ema_1 = config.ema_1

        self.activation = get_activation_fn(config.activation_fn)

        self.patch_embedding = get_patch_embedding(config.patch_embedding, config.patch_size, config.embedding_size, num_channels)

        xlstm_emb_size = config.embedding_size

        if config.xlstm_type == 'large':
            self.xlstm = get_large_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads, bidirectional=config.bidirectional)
        else:
            self.xlstm = get_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads, bidirectional=config.bidirectional)

        if self.use_teacher_student:    
            if config.xlstm_type == 'large':
                self.xlstm_teacher = get_large_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads, bidirectional=config.bidirectional)
            else:
                self.xlstm_teacher = get_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads, bidirectional=config.bidirectional)
            # do not require gradients for the teacher
            for param in self.xlstm_teacher.parameters():
                param.requires_grad = False
            self.predictor = HeadModule(
                inp_size=config.embedding_size,
                hidden_size=config.embedding_size // 2,
                out_size=config.embedding_size,
                dropout=config.dropout
            )
                 
        self.random_drop_leads = RandomDropLeads(config.random_drop_leads)
        self.random_surrogate = FTSurrogate(0.05, prob=config.random_surrogate_prob)
        self.random_jitter = Jitter(sigma=0.1, prob=config.random_jitter_prob)

        self.reconstruction = get_reconstruction_head(config.reconstruct_embedding, config.patch_size, config.embedding_size, num_channels)

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

        if self.training_strategy == 'masked_token_prediction':
            out = self.xlstm(x, need_expansion=False) # [batch_size, embedding_dim]
        elif self.training_strategy == 'next_token_prediction':
            out = self.xlstm(x) # [batch_size, embedding_dim]

        out, last_emb = self.reconstruction(out)

        if self.use_teacher_student:
            with torch.no_grad():
                out_teacher = self.xlstm_teacher(x)
            return out, out_teacher, last_emb
        
        return out, None, None
    
    def generate(self, x, length=10):
        if self.training_strategy != 'next_token_prediction':
            raise ValueError('Only next token prediction is supported for generation')
         
        # i do not need to drop the leads here
        x = self.embed_data(x, augment=False)

        if self.bidirectional:
            reconstructed = []
            for i in range(length - 1):
                out = self.xlstm(x, need_expansion=False)
                r, _ = self.reconstruction(out[:, -1, :].unsqueeze(1))
                reconstructed.append(r)
                toadd = self.embed_data(r, augment=False)
                x = torch.cat([x, toadd], dim=1)
    
            return torch.cat(reconstructed, dim=1)

        state = None
        for i in range(x.shape[1]):
            new_x, state = self.xlstm.step(x[:, i].unsqueeze(1), state=state)

        r, _ = self.reconstruction(new_x)

        reconstructed = [r]

        for i in range(length - 1):
            x = self.embed_data(r, augment=False)
            x, state = self.xlstm.step(x, state=state)

            r, _ = self.reconstruction(x)

            reconstructed.append(r)
        
        return torch.cat(reconstructed, dim=1)

    def trainable_parameters(self):
        if self.use_teacher_student:
            return [param for name, param in self.named_parameters() if "xlstm_teacher" not in name]
        return self.parameters()
    

class xLSTMClassificationMIT_BIH(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels
        ): 

        super(xLSTMClassificationMIT_BIH, self).__init__(num_channels, config)

        self.use_start_token = config.use_start_token
        if self.use_start_token:
            self.start_token = nn.Parameter(torch.zeros(1, 1, config.embedding_size))

        self.fc = HeadModule(
            inp_size=config.embedding_size,
            hidden_size=config.embedding_size // 2,
            out_size=num_classes,
            dropout=config.dropout
        )

        self.r_peak_pos_fc = HeadModule(
            inp_size=config.embedding_size,
            hidden_size=config.embedding_size // 2,
            out_size=self.patch_size,
            dropout=config.dropout
        )

    def forward(self, x):
        x = self.embed_data(x)

        if self.use_start_token:
            start_token = self.start_token.expand(x.shape[0], -1, -1)
            x = torch.cat([start_token, x], dim=1)

        out = self.xlstm(x, need_expansion=False) # [batch_size, embedding_dim]
        if self.use_start_token: out = out[:, self.start_token.shape[1]:, :] # remove the start tokens

        cls = self.fc(out)
        r_peak_pos = self.r_peak_pos_fc(out)
        return cls, r_peak_pos

    def finetuning_params(self):
        params = []
        params.extend(self.xlstm.parameters())
        params.extend(self.patch_embedding.parameters())
        return params

    def training_params(self):
        params = []
        if self.use_start_token:
            params.append(self.start_token)
        params.extend(self.fc.parameters())
        params.extend(self.r_peak_pos_fc.parameters())
        return params

class xLSTMClassification(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels
        ): 

        super(xLSTMClassification, self).__init__(num_channels, config)

        self.use_cls_token = config.use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_size))

        self.fc = HeadModule(
            inp_size=config.embedding_size,
            hidden_size=config.embedding_size // 2,
            out_size=num_classes,
            dropout=config.dropout
        )

    def forward(self, x):
        x = self.embed_data(x)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        out = self.xlstm(x, need_expansion=False)[:, -1, :]
        cls = self.fc(out)
        return cls
    
    def finetuning_params(self):
        params = []
        params.extend(self.xlstm.parameters())
        params.extend(self.patch_embedding.parameters())
        return params

    def training_params(self):
        params = []
        params.extend(self.fc.parameters())
        return params
