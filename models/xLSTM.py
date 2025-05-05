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
        self.mask_ratio = config.mask_ratio
        self.embedding_size = config.embedding_size
        self.use_sim_dino = config.use_sim_dino

        self.patch_embedding = get_patch_embedding(config.patch_embedding, config.patch_size, config.embedding_size, num_channels)

        xlstm_emb_size = config.embedding_size

        if config.xlstm_type == 'large':
            self.xlstm = get_large_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads, bidirectional=config.bidirectional)
        else:
            self.xlstm = get_xlstm(xlstm_emb_size, dropout=config.dropout, blocks=config.xlstm_config, num_heads=config.num_heads, bidirectional=config.bidirectional, drop_path=config.drop_path_prob)

        if self.training_strategy == 'masked_token_prediction':
            self.mask_token = nn.Parameter(torch.zeros(config.embedding_size))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_size))
            self.reg_token = nn.Parameter(torch.zeros(1, config.num_reg_token, config.embedding_size))
            nn.init.xavier_uniform_(self.cls_token, gain=1.0)

        if self.use_teacher_student:   
            if not self.use_sim_dino:
                self.dino_head = HeadModule(
                    inp_size=config.embedding_size,
                    hidden_size=config.embedding_size // 2,
                    out_size=config.n_prototypes,
                    dropout=0.
                )

                self.ibot_head = HeadModule(
                    inp_size=config.embedding_size,
                    hidden_size=config.embedding_size // 2,
                    out_size=config.n_prototypes,
                    dropout=0.
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
    
    def forward(self, x, masking=True, reconstruct=True):
        num_reg_tokens = self.reg_token.shape[1]

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

        # add the [cls] and [reg] tokens
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        reg_tokens = self.reg_token.expand(x.shape[0], -1, -1)
        x_emb = torch.cat([x_emb, cls_token, reg_tokens], dim=1)

        # pass to xlstm
        need_expansion = self.training_strategy == 'next_token_prediction' and self.bidirectional
        out = self.xlstm(x_emb, need_expansion = need_expansion) # [batch_size, embedding_dim]

        out = out[:, :-num_reg_tokens, :]

        # reconstruct signal
        if reconstruct:
            rec, _ = self.reconstruction(out[:, :-1, :].clone().detach())

        out = self.layer_norm(out)

        cls = out[:, -1, :]
        patches = out[:, :-1, :]
       
        tortn = {
            'patches': patches,
            'cls': cls,
        }
        
        if masking: tortn['mask'] = mask
        if reconstruct: tortn['reconstruction'] = rec
        
        if not self.use_sim_dino:
            cls_after_head = self.dino_head(cls)
            patches_after_head = self.ibot_head(patches)
            tortn['cls_after_head'] = cls_after_head
            tortn['patches_after_head'] = patches_after_head
        
        return tortn

    @torch.no_grad()
    def teacher_fwd(self, x):
        return self._teacher(x, masking=False, reconstruct=False)
        
    def _forward(self, x):
        if self.training_strategy == 'masked_token_prediction':
            mask = self.get_random_mask(x) # 1 is masked and 0 is non masked
            # mask a rnadom number of patches
            masked_x = x.masked_fill(mask, 0) # apply the mask
            x_emb = self.patch_embedding(masked_x)

            # set the elements to 0 with the mask token
            batch_size, tokens_num, _ = x.shape
            patched_mask = mask.view(batch_size, tokens_num // self.patch_size, self.patch_size)[:, :, 0]
            x_emb[patched_mask] = self.mask_token

            # add the cls_token
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            reg_tokens = self.reg_token.expand(x.shape[0], -1, -1)
            x_emb = torch.cat([x_emb, cls_token, reg_tokens], dim=1)
        else:
            mask = None
            x_emb = self.patch_embedding(x)

        num_reg_tokens = self.reg_token.shape[1]

        need_expansion = self.training_strategy == 'next_token_prediction' and self.bidirectional
        out = self.xlstm(x_emb, need_expansion = need_expansion) # [batch_size, embedding_dim]

        rec, _ = self.reconstruction(out[:, :-(1 + num_reg_tokens), :].clone().detach())

        if self.use_teacher_student:
            # out = self.layer_norm(out)
            student_cls = out[:, -(1 + num_reg_tokens), :]
            student_patches = out[:, :-(1 + num_reg_tokens), :]

            # student_cls_after_head = self.dino_head(student_cls)
            # student_patches = self.ibot_head(student_patches)

            with torch.no_grad():
                if self.training_strategy == 'masked_token_prediction':
                    x_emb_teacher = self._patch_embedding_teacher(masked_x)
                    cls_token = self._cls_token_teacher.expand(x.shape[0], -1, -1)
                    reg_tokens = self._reg_tokens_teacher.expand(x.shape[0], -1, -1)
                    x_emb_teacher = torch.cat([x_emb_teacher, cls_token, reg_tokens], dim=1)
                else: 
                    x_emb_teacher = self._patch_embedding_teacher(x)
                
                out_teacher = self._xlstm_teacher(x_emb_teacher, need_expansion=need_expansion)
                # out_teacher = self._layer_norm_teacher(out_teacher)

                teacher_cls = out_teacher[:, -(1 + num_reg_tokens), :]
                teacher_patches = out_teacher[:, :-(1 + num_reg_tokens), :]

                # teacher_cls_after_head = self._dino_head_teacher(teacher_cls)
                # teacher_patches = self._ibot_head_teacher(teacher_patches)

            return {
                'reconstruction': rec, 
                'teacher_patches': teacher_patches,
                'student_patches': student_patches,
                'teacher_cls': teacher_cls,
                'student_cls': student_cls,
                # 'student_cls_after_head': student_cls_after_head,
                # 'teacher_cls_after_head': teacher_cls_after_head,
                'mask': mask,
            }
            
        return {
            'reconstruction': rec, 
            'mask': mask,
        }
    
    def get_random_mask(self, x):
        """
        Retutn a mask of the same shape as x, masked values are set to TRUE
        """
        # masking the signal
        num_patches = x.shape[1] // self.patch_size
        rand = torch.rand(x.shape[0], num_patches, device=x.device)
        mask = (rand < self.mask_ratio) # this is true for masked
        # repeat the mask to num_patches * patch_size
        mask = mask.repeat_interleave(self.patch_size, dim=1).unsqueeze(-1)
        # check when the x was all 0 and set the mask to 0
        padding_mask = (x.abs().sum(dim=-1) == 0).unsqueeze(-1)
        #print('padding mask', padding_mask.shape)
        #print('mask', mask.shape)
        return mask | padding_mask
    
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
            nn.init.xavier_uniform_(self.cls_token, gain=1.0)


        self.norm = nn.LayerNorm(normalized_shape=config.embedding_size)

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

        if self.use_cls_token: out = out[:, -1, :] # remove the cls token
        else: out = out.max(dim=1)[0]

        out = self.norm(out)
        cls = self.fc(out)
        return cls
    
    def finetuning_params(self):
        params = []
        params.extend(self.xlstm.parameters())
        params.extend(self.patch_embedding.parameters())       
        if self.use_cls_token:
            params.append(self.cls_token)
        return params

    def training_params(self):
        params = []
        params.extend(self.fc.parameters())
 
        return params


class Centering(nn.Module):
    def __init__(self, dim, momentum=0.9):
        """
        Args:
            dim (int): Dimension of the embeddings to center.
            momentum (float): EMA momentum for updating the center (e.g., 0.9 or 0.99).
        """
        super().__init__()
        self.momentum = momentum
        self.register_buffer('center', torch.zeros(1, dim))

    @torch.no_grad()
    def update_center(self, batch_output):
        """
        Update the running center with the current batch output.

        Args:
            batch_output (Tensor): Batch of teacher outputs [batch_size, dim].
        """
        batch_center = batch_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.momentum + batch_center * (1 - self.momentum)

    def forward(self, teacher_output):
        """
        Center the teacher output by subtracting the running center.

        Args:
            teacher_output (Tensor): Teacher outputs [batch_size, dim].

        Returns:
            centered_output (Tensor): Centered teacher outputs.
        """
        return teacher_output - self.center