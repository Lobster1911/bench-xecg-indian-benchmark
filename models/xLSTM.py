import torch
from torch import nn

from models.utils import get_activation_fn, get_xlstm, get_large_xlstm, get_patch_embedding, get_reconstruction_head
from models.modules import HeadModule
from models.SeriesDecomposition import SeriesDecomposition 
from augmentations import RandomDropLeads, FTSurrogate, Jitter, RandomResample
import numpy as np
import torch.nn.functional as F
import copy
from models.normalizations import DINOCentering
import torch.distributed as dist

class pretrainedxLSTM(nn.Module):
    def __init__(
            self, 
            num_channels,
            config,
            reconstruction=True
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
            self._patch_embedding_teacher = copy.deepcopy(self.patch_embedding)
            
            if self.training_strategy == 'masked_token_prediction':
                self._xlstm_teacher = copy.deepcopy(self.xlstm)
                # discard the last two blocks of the xlstm, in this way the student has to more layers and it is different than the parent
                # do not require gradients for the teacher and copy from the student
                for param_t in self._xlstm_teacher.parameters():
                    param_t.requires_grad = False

                self._xlstm_teacher.eval()


            for param_t in self._patch_embedding_teacher.parameters():
                param_t.requires_grad = False

            self._patch_embedding_teacher.eval()

            self.vocab = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
            self._vocab_teacher = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
            self._vocab_teacher.weight = self.vocab.weight
            self._vocab_teacher.weight.requires_grad = False

            # self._center_module = nn.BatchNorm1d(config.vocab_size, affine=False, momentum=0.9)
                 
        if reconstruction:
            self.reconstruction = get_reconstruction_head(config.patch_size, config.embedding_size, num_channels)

            if self.weight_tying and config.patch_embedding == 'linear': 
                self.reconstruction.deconv.weight = self.patch_embedding.conv.weight


    def forward(self, x):
        x_emb = self.patch_embedding(x)

        need_expansion = self.training_strategy == 'next_token_prediction' and self.bidirectional
        out = self.xlstm(x_emb, need_expansion = need_expansion) # [batch_size, embedding_dim]

        rec, _ = self.reconstruction(out)

        if self.use_teacher_student:
            out = self.vocab(out) 
            with torch.no_grad():
                x_emb_teacher = self._patch_embedding_teacher(x)

                if self.training_strategy == 'masked_token_prediction':
                    x_emb_teacher = self._xlstm_teacher(x_emb_teacher, need_expansion=False) # [batch_size, embedding_dim]

                out_teacher = self._vocab_teacher(x_emb_teacher)
                # out_teacher = self._center_module(out_teacher.permute(0, 2, 1)).permute(0, 2, 1) # [batch_size, embedding_dim]
                # centering
                return rec, out_teacher, out
            
        return rec, None, None
    
    def generate(self, x, length=10):
        if self.training_strategy != 'next_token_prediction':
            raise ValueError('Only next token prediction is supported for generation')
         
        # i do not need to drop the leads here
        x = self.patch_embedding(x, augment=False)

        if self.bidirectional:
            reconstructed = []
            for i in range(length - 1):
                out = self.xlstm(x, need_expansion=False)
                new_patch = out[:, -1, :].unsqueeze(1)
                r, _ = self.reconstruction(new_patch)
                reconstructed.append(r)
                # toadd = self.patch_embedding(r, augment=False)
                x = torch.cat([x, new_patch], dim=1)
    
            return torch.cat(reconstructed, dim=1)
        else:
            state = None
            for i in range(x.shape[1]):
                new_x, state = self.xlstm.step(x[:, i].unsqueeze(1), state=state)

            r, _ = self.reconstruction(new_x)

            reconstructed = [r]

            for i in range(length - 1):
                new_x, state = self.xlstm.step(new_x, state=state)
                r, _ = self.reconstruction(new_x)
                reconstructed.append(r)
            
            return torch.cat(reconstructed, dim=1)


    def trainable_parameters(self):
        if self.use_teacher_student:
            return [param for name, param in self.named_parameters() if "_xlstm_teacher" not in name and 'reconstruction' not in name and 'patch_embedding' not in name]
        
        return self.parameters()
    

class xLSTMClassificationMIT_BIH(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels
        ): 

        dropout = config.dropout
        config.dropout = 0.0  
        super(xLSTMClassificationMIT_BIH, self).__init__(num_channels, config, reconstruction=False)

        self.use_start_token = config.use_start_token
        if self.use_start_token:
            self.start_token = nn.Parameter(torch.zeros(1, 1, config.embedding_size))

        self.fc = HeadModule(
            inp_size=config.embedding_size,
            hidden_size=config.embedding_size // 2,
            out_size=num_classes,
            dropout=dropout
        )

        self.r_peak_pos_fc = HeadModule(
            inp_size=config.embedding_size,
            hidden_size=config.embedding_size // 2,
            out_size=self.patch_size,
            dropout=dropout
        )

    def forward(self, x):
        x = self.patch_embedding(x)

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
        dropout = config.dropout
        # config.dropout = 0.0  
        super(xLSTMClassification, self).__init__(num_channels, config, reconstruction=False)

        self.use_cls_token = config.use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_size))

        self.fc = HeadModule(
            inp_size=config.embedding_size,
            hidden_size=config.embedding_size // 2,
            out_size=num_classes,
            dropout=dropout
        )

    def forward(self, x):
        x = self.patch_embedding(x)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([x, cls_token], dim=1)

        out = self.xlstm(x, need_expansion=False)# [:, -1, :]
        out = out.max(dim=1)[0]
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
        if self.use_cls_token:
            params.append(self.cls_token)
        return params
