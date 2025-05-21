import torch
from torch import nn

from models.utils import get_xlstm, get_large_xlstm, get_patch_embedding, get_reconstruction_head
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
        self.mask_ratio = config.mask_ratio
        self.embedding_size = config.embedding_size
        self.use_sim_dino = config.use_sim_dino
        self.cls_type = config.cls_type
        self.use_final_layer_norm = config.use_final_layer_norm
        self.masking_type = config.masking_type
        self.xlstm_type = config.xlstm_type


        self.patch_embedding = get_patch_embedding(config.patch_embedding, config.patch_size, config.embedding_size, num_channels)
        xlstm_emb_size = config.embedding_size

        if config.xlstm_type == 'large':
            self.xlstm = get_large_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads, bidirectional=config.bidirectional,  drop_path=config.drop_path_prob)
        else:
            self.xlstm = get_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads, bidirectional=config.bidirectional, drop_path=config.drop_path_prob)

        if self.training_strategy == 'masked_token_prediction':
            self.mask_token = nn.Parameter(torch.zeros(config.embedding_size))
        
        if self.cls_type == 'token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_size))
            nn.init.xavier_uniform_(self.cls_token, gain=1.0)
            
        self.num_reg_tokens = config.num_reg_token
        if config.num_reg_token > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, config.num_reg_token, config.embedding_size))
            nn.init.xavier_uniform_(self.reg_token, gain=1.0)
            

        if self.use_teacher_student:   
            if not self.use_sim_dino:
                self.dino_head = HeadModule(
                    inp_size=config.embedding_size,
                    hidden_size=config.embedding_size // 2,
                    out_size=config.n_prototypes,
                    dropout=config.dropout
                )

                self.ibot_head = HeadModule(
                    inp_size=config.embedding_size,
                    hidden_size=config.embedding_size // 2,
                    out_size=config.n_prototypes,
                    dropout=config.dropout
                )


        self.layer_norm = nn.LayerNorm(config.embedding_size)
                          
        if reconstruction:
            self.reconstruction = get_reconstruction_head(config.patch_size, config.embedding_size, num_channels)

    def init_teacher(self):
        self._teacher = self.create_teacher_module()

    def create_teacher_module(self):
        param = copy.deepcopy(self)
        # remove reconstruction params
        for name in list(param._modules.keys()):
            if 'reconstruction' in name or 'teacher' in name:
                print(f'removing {name} from teacher network')
                del param._modules[name]

        for param_t in param.parameters():
            param_t.requires_grad = False

        param.eval()
        return param
    
    def create_teacher_param(self, original):
        param = copy.deepcopy(original)
        param.requires_grad = False
        return param
    
    def forward_xlstm(self, x):
        # add the [cls] and [reg] tokens
        if self.cls_type == 'token':
            x = self.add_cls_token(x)
        if self.num_reg_tokens > 0:
            x = self.add_reg_tokens(x)

        # pass to xlstm
        need_expansion = self.training_strategy == 'next_token_prediction' and self.bidirectional
        out = self.xlstm(x, need_expansion = need_expansion) # [batch_size, embedding_dim]

        if self.num_reg_tokens > 0:
            out = self.remove_reg_tokens(out)
           
        if self.use_final_layer_norm:
            out = self.layer_norm(out)

        if self.cls_type == 'max':
            cls = out.max(dim=1)[0]

        elif self.cls_type == 'mean' or self.cls_type == 'avg':
            cls = out.mean(dim=1)
        else:
            cls = out[:, -1, :]
            out = out[:, :-1, :]

        return cls, out
    
    def forward(self, x, masking=True, reconstruct=True):
        if masking:   # masking
            mask = self.get_random_mask(x) # 1 is masked and 0 is non masked
            x = x.masked_fill(mask, 0) # apply the mask

        # patching
        x_emb = self.patch_embedding(x)

        # adding mask tokens
        if masking:
            batch_size, tokens_num, _ = x.shape
            patched_mask = mask.view(batch_size, tokens_num // self.patch_size, self.patch_size)[:, :, 0]
            x_emb[patched_mask] = self.mask_token

        
        cls, out = self.forward_xlstm(x_emb)

        # reconstruct signal
        if reconstruct:
            rec, _ = self.reconstruction(out.clone().detach())

        tortn = {
            'patches': out,
            'cls': cls,
        }
        
        if masking: tortn['mask'] = mask
        if reconstruct: tortn['reconstruction'] = rec
        
        if not self.use_sim_dino:
            cls_after_head = self.dino_head(cls)
            patches_after_head = self.ibot_head(out)
            tortn['cls_after_head'] = cls_after_head
            tortn['patches_after_head'] = patches_after_head
        
        return tortn

    @torch.no_grad()
    def teacher_fwd(self, x):
        return self._teacher(x, masking=False, reconstruct=False)
         
    def add_reg_tokens(self, x):
        reg_tokens = self.reg_token.expand(x.shape[0], -1, -1)
        half = self.num_reg_tokens // 2
        return torch.cat([reg_tokens[:, :half, :], x, reg_tokens[:, half:, :]], dim=1)
    
    def remove_reg_tokens(self, x):
        half = self.num_reg_tokens // 2
        return x[:, half:-(self.num_reg_tokens - half), :]
    
    def add_cls_token(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls_token, x], dim=1)
    

    def get_random_mask(self, x):
        """
        Retutn a mask of the same shape as x, masked values are set to TRUE
        """
        # check when the x was all 0 and set the mask to 0
        padding_mask = (x.abs().sum(dim=-1) == 0).unsqueeze(-1)
        num_patches = x.shape[1] // self.patch_size

        if self.masking_type == 'random':
            # masking the signal
            rand = torch.rand(x.shape[0], num_patches, device=x.device)
            mask = (rand < self.mask_ratio) # this is true for masked
            # repeat the mask to num_patches * patch_size
            mask = mask.repeat_interleave(self.patch_size, dim=1).unsqueeze(-1)
        elif self.masking_type == 'block':
            rand = torch.rand(x.shape[0], num_patches, device=x.device)
            mask = (rand < self.mask_ratio / 4) # this is true for masked
            # after a masked patch, the next 3 patches are masked
            for i in range(1, 4):
                mask = mask | mask.roll(-1, dims=1)
            # repeat the mask to num_patches * patch_size
            mask = mask.repeat_interleave(self.patch_size, dim=1).unsqueeze(-1)

        return mask & ~padding_mask
    
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
            return [param for name, param in self.named_parameters() if "teacher" not in name and 'reconstruction' not in name]
        
        return self.parameters()
