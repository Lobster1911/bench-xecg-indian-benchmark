import torch
from torch import nn

from models.utils import get_normalization_layer, get_xlstm, get_large_xlstm, get_patch_embedding, get_reconstruction_head,  get_transformer
import copy
from models.pooling import AttentionPooling, LinearAttentionPooling
from models.base_model import BaseModel

class pretrainedxLSTM(BaseModel):
    def __init__(
            self, 
            num_channels,
            config,
            reconstruction=True
        ): 
        super(pretrainedxLSTM, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.patch_size = config.patch_size
        self.bidirectional = config.bidirectional
        self.use_teacher_student = config.strategy == 'lejepa'
        self.mask_ratio = config.mask_ratio
        self.embedding_size = config.embedding_size
        self.cls_type = config.cls_type
        self.masking_type = config.masking_type
        self.encoder_type = config.encoder_type
        self.sampling_freq = config.sampling_freq
        self.linear_probing = config.linear_probing

        self.patch_embedding = get_patch_embedding(config.patch_embedding, config.patch_size, config.embedding_size, num_channels)

        if config.encoder_type == 'large':
            self.core = get_large_xlstm(config)
        elif config.encoder_type == 'transformer':
            self.core = get_transformer(config)
        else:
            self.core = get_xlstm(config)

        self.mask_token = nn.Parameter(torch.zeros(config.embedding_size))
        
        if self.cls_type == 'token' or self.cls_type == 'token_2':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_size))
            nn.init.xavier_uniform_(self.cls_token, gain=1.0)
        elif self.cls_type == 'attn_pool':
            self.attn_pool = AttentionPooling(config.embedding_size, config.num_heads)
        elif self.cls_type == 'lin_attn_pool':
            self.attn_pool = LinearAttentionPooling(config.embedding_size)
            
        self.num_reg_tokens = config.num_reg_token
        if config.num_reg_token > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, config.num_reg_token, config.embedding_size))
            nn.init.xavier_uniform_(self.reg_token, gain=1.0)
          
        if reconstruction:
            self.reconstruction = get_reconstruction_head(config.patch_size, config.embedding_size, num_channels)

        self.normalization_layer = get_normalization_layer(config, config.embedding_size)

        if self.linear_probing:
            # freezing model
            for name, param in self.named_parameters():
                if 'head' not in name:  # no freezing last layer
                    param.requires_grad = False

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
    
    def pooling(self, out, padding_mask=None, pooling_type='avg'):
        cls = None
        if pooling_type == 'max':
            if padding_mask is None:
                cls = out.max(dim=1)[0]
            else:
                cls = out.masked_fill(padding_mask, -torch.inf).max(dim=1)[0]
        elif pooling_type == 'mean' or pooling_type == 'avg':
            if padding_mask is None:
                cls = out.mean(dim=1)
            else:
                cls = out.masked_fill(padding_mask, 0).sum(dim=1) / (out.shape[1] - padding_mask.sum(dim=1)).clamp(min=1)

        elif pooling_type == 'mix':
            if padding_mask is None:
                max_p = out.max(dim=1)[0]
                mean_p = out.mean(dim=1)
            else:
                max_p = out.masked_fill(padding_mask, -torch.inf).max(dim=1)[0]
                mean_p = out.masked_fill(padding_mask, 0).sum(dim=1) / (out.shape[1] - padding_mask.sum(dim=1)).clamp(min=1)
            cls = torch.cat([max_p, mean_p], dim=-1)
        elif pooling_type == 'token':
            cls = out[:, -1, :]
            out = out[:, :-1, :]
        elif pooling_type == 'token_2':
            cls_1 = out[:, 0, :]
            cls_2 = out[:, -1, :]
            out = out[:, 1:-1, :]
            cls = cls_1 + cls_2
        elif pooling_type == 'attn_pool' or pooling_type == 'lin_attn_pool':
            if padding_mask is None:
                cls = self.attn_pool(out).squeeze()
            else:
                cls = self.attn_pool(out.masked_fill(padding_mask, 0)).squeeze()  
        else:
            return cls, out

        cls = self.normalization_layer(cls)    
        return cls, out
    
    def forward_core(self, x, padding_mask=None):
        # add the [cls] and [reg] tokens
        if self.cls_type == 'token':
            x = self.add_cls_token(x)
        elif self.cls_type == 'token_2':
            x = self.add_cls_token_2(x)

        if self.num_reg_tokens > 0:
            x = self.add_reg_tokens(x)

        # pass to xlstm
        need_expansion = False
        out = self.core(x, need_expansion = need_expansion) # [batch_size, embedding_dim]

        if self.num_reg_tokens > 0:
            out = self.remove_reg_tokens(out)

        cls, out = self.pooling(out, padding_mask, pooling_type=self.cls_type)
        return cls, out

    def mask_signal_if_needed(self, x, masking):
        padding_mask = self.get_padding_mask(x)

        if masking:   # masking
            mask = self.get_random_mask(x) # 1 is masked and 0 is non masked
            x = x.masked_fill(mask, 0) # apply the mask

        # patching
        x_emb = self.patch_embedding(x)

        # adding mask tokens
        if masking:
            batch_size, seq_len, _ = x.shape
            # patching the mask, dimension [batch_size, seq_len]
            patched_mask = mask.view(batch_size, seq_len // self.patch_size, self.patch_size)[:, :, 0]
            x_emb = torch.where(
                patched_mask.unsqueeze(-1), 
                self.mask_token.expand_as(x_emb),
                x_emb
            )
            # x_emb[patched_mask] = self.mask_token

        return x_emb, padding_mask, mask if masking else None
    
    def forward(self, x, masking=True, reconstruct=True):
        x_emb, padding_mask, mask = self.mask_signal_if_needed(x, masking)
        
        cls, out = self.forward_core(x_emb, padding_mask=padding_mask)

        # reconstruct signal
        if reconstruct:
            rec, _ = self.reconstruction(out.clone().detach())

        tortn = {
            'patches': out,
            'cls': cls,
        }
        
        if masking: tortn['mask'] = mask
        if reconstruct: tortn['reconstruction'] = rec
        
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
    
    def add_cls_token_2(self, x):
        cls_token_1 = self.cls_token.expand(x.shape[0], -1, -1)
        cls_token_2 = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls_token_1, x, cls_token_2], dim=1)
    
    def get_padding_mask(self, x):
        """
        Return a padding mask of shape [batch_size, num_patches, embedding_size]
        where masked values are set to TRUE
        """
        padding_mask = (x.abs().sum(dim=-1) == 0).unsqueeze(-1)
        num_patches = x.shape[-2] // self.patch_size
        padding_mask_patched = padding_mask.view(-1, num_patches, self.patch_size)[:, :, 0].unsqueeze(-1).expand(-1, -1, self.embedding_size)
        return padding_mask_patched

    def get_random_mask(self, x):
        """
        Retutn a mask of the same shape as x, masked values are set to TRUE
        """
        # check when the x was all 0 and set the mask to 0
        padding_mask = (x.abs().sum(dim=-1) == 0).unsqueeze(-1)
        num_patches = x.shape[-2] // self.patch_size

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
    
    def trainable_parameters(self):
        if self.use_teacher_student:
            return [param for name, param in self.named_parameters() if "teacher" not in name and 'reconstruction' not in name]
        
        return self.parameters()

    def get_features(self, x, feature_classification=False):
        """
        This function should be the complete forward pass apart from the classification head.
        """
        x_emb, mask, _ = self.mask_signal_if_needed(x, False)
        
        cls, out = self.forward_core(x_emb, padding_mask=mask)

        if feature_classification:
            return {'feat': out}
        
        tortn = {}

        if self.cls_type != 'avg' and self.cls_type != 'mean':
            avg, _ = self.pooling(out, padding_mask=mask, pooling_type='avg')
            tortn['avg'] = avg
        else:
            tortn['avg'] = cls
        
        if self.cls_type != 'max':
            max, _ = self.pooling(out, padding_mask=mask, pooling_type='max')
            tortn['max'] = max
        else:
            tortn['max'] = cls

        if self.cls_type == 'attn_pool':
            tortn['attn_pool'] = cls

        if self.cls_type == 'token' or self.cls_type == 'token_2':
            tortn['token'] = cls
        
        return tortn

    
    def get_layers(self):
        """
        This function should return the layers of the model where to apply the layerwise decay
        """
        return self.core.model.blocks
    
    def additional_params(self, lr, last_layer_lr, wd):
        """
        This fucntion should return additional parameters used by a model (like classification token and so on...)
        """
        params = []
        params.append({"params": self.patch_embedding.parameters(), "lr": last_layer_lr, "name": "patch_embedding"})

        if self.encoder_type =='large':
            params.append({'params': self.core.model.out_norm.parameters(), 'lr': lr, 'weight_decay': wd, 'name': 'ln2'})
        else:
            params.append({'params': self.core.model.post_blocks_norm.parameters(), 'lr': lr, 'weight_decay': wd, 'name': 'ln2'})

        if self.cls_type == 'token' or self.cls_type == 'token_2':
            params.append({'params': self.cls_token, 'lr': lr, 'weight_decay': wd, 'name': 'cls'})
        elif self.cls_type == 'attn_pool' or self.cls_type == 'lin_attn_pool':
            params.append({'params': self.attn_pool.parameters(), 'lr': last_layer_lr, 'weight_decay': wd, 'name': 'cls'})

        if self.num_reg_tokens > 0:
            params.append({'params': self.reg_token, 'lr': last_layer_lr, 'weight_decay': wd, 'name': 'reg_tokens'})

        if hasattr(self.core, 'post_blocks_norm'):
            params.append({'params': self.core.post_blocks_norm, 'lr': lr, 'name': 'post_block_norm'})

        return params
        
