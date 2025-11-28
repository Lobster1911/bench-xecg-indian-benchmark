from torch import nn
import torch 
import numpy as np
from torch.nn import functional as F
from xlstm.xlstm_large.model import mLSTMStateType


class LinearPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.conv = nn.Conv1d(num_channels, num_hiddens, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x, permute=True):
        if permute: x = x.permute(0, 2, 1) # put the channels in the middle
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x

class NonLinearPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, num_hiddens, kernel_size=patch_size, stride=patch_size)
        self.conv2 = nn.Conv1d(num_channels, num_hiddens, kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(num_hiddens, num_hiddens)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1) # put the channels in the middle
        x1 = self.act(self.conv1(x))
        x2 = self.conv2(x)
        x = (x1 + x2).flatten(2).transpose(1, 2)
        return x
    

class ConvPatchEmbedding(nn.Module):
    def __init__(self, patch_size=25, num_hiddens=256, num_channels=12):
        super().__init__()
        
        # LOGIC TO DETERMINE STRIDES AUTOMATICALLY
        # We want Total Stride == patch_size.
        # We try to split it into 2 layers to allow for feature extraction.
        
        if patch_size % 4 == 0:
            # Case for 100, 64, 128, etc.
            self.stride_1 = 4
            self.stride_2 = patch_size // 4
            kernel_1 = 15 # Good default for ~150ms coverage
            pad_1 = 7     # Keeps size consistent
        elif patch_size % 5 == 0:
            # Case for 25, 50, 75
            self.stride_1 = 5
            self.stride_2 = patch_size // 5
            kernel_1 = 11 # Slightly smaller kernel for smaller patches
            pad_1 = 5
        elif patch_size % 2 == 0:
            # Case for 2, 6, 10, etc.
            self.stride_1 = 2
            self.stride_2 = patch_size // 2
            kernel_1 = 7
            pad_1 = 3
        else:
            # Prime numbers or odd sizes (e.g., 23) -> Fallback to single layer
            self.stride_1 = patch_size
            self.stride_2 = 1 
            kernel_1 = patch_size
            pad_1 = 0

        # Dimension checks
        mid_channels = num_hiddens // 2

        layers = []
        
        # --- LAYER 1: Feature Extraction ---
        # If stride_1 is patch_size (fallback), this does all the work.
        layers.append(nn.Conv1d(num_channels, mid_channels, kernel_size=kernel_1, stride=self.stride_1, padding=pad_1, bias=False))
        layers.append(nn.BatchNorm1d(mid_channels))
        layers.append(nn.GELU())

        # --- LAYER 2: Aggregation (Only if we split the stride) ---
        if self.stride_2 > 1:
            # We set kernel_size equal to stride_2 to fully consume the window
            layers.append(nn.Conv1d(mid_channels, num_hiddens, kernel_size=self.stride_2, stride=self.stride_2, bias=False))
            layers.append(nn.BatchNorm1d(num_hiddens))
            layers.append(nn.GELU())
        else:
            # If we didn't split (fallback case), we just project channel dims
            layers.append(nn.Conv1d(mid_channels, num_hiddens, kernel_size=1, stride=1, bias=False))
            layers.append(nn.BatchNorm1d(num_hiddens))
            layers.append(nn.GELU())

        self.stem = nn.Sequential(*layers)

        # Print setup for verification
        print(f" initialized with Total Stride: {self.stride_1 * self.stride_2} (S1:{self.stride_1} x S2:{self.stride_2})")

    @torch._dynamo.disable
    def forward(self, x, permute=True):
        if permute: x = x.permute(0, 2, 1) # [B, C, L]
        x = self.stem(x)
        x = x.transpose(1, 2) # [B, N_Patches, Embed_Dim]
        return x
    
    
      
class EmbedPatching(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12, use_pre_head=False):
        super().__init__()
        self.use_pre_head = use_pre_head  
        if use_pre_head: self.pre_head = HeadModule(num_hiddens, num_hiddens // 2, num_hiddens)
        self.deconv = nn.ConvTranspose1d(num_hiddens, num_channels, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        if self.use_pre_head: x = self.pre_head(x)
        out = x.transpose(1, 2)
        out = self.deconv(out).transpose(1, 2)
        # print('x shape after deconv', x.shape) [1, 3584, 12]
        return out, x
     
class HeadModule(nn.Module):
    
    def __init__(self, inp_size, hidden_size, out_size, dropout=0.1, activation='relu'):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_size),
        )
        
    def forward(self, x):
        return self.head(x)
    
class vanillaxLSTMWrapper(nn.Module):
    def __init__(self, xlstm, dropout=0.2, bidirectional=False, drop_path=0.):
        super(vanillaxLSTMWrapper, self).__init__() 
        self.model = xlstm
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.drop_path = DropPath()
        self.dropout_rates = [x.item() for x in torch.linspace(0, drop_path, len(self.model.blocks))]

    def step(self, x, state=None):
        return self.model.step(x, state=state)

    def forward(self, x: torch.Tensor, need_expansion=True):
        expanded = False

        for i, block in enumerate(self.model.blocks):
            if self.bidirectional: 
                if not expanded and i > 0 and need_expansion:
                    bs, seq_len, _ = x.shape
                    x = x.unsqueeze(1).repeat(1, seq_len, 1, 1)
                    tril_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=x.dtype, device=x.device)).unsqueeze(0).unsqueeze(-1)
                    x = x * tril_mask
                    x = x.reshape(bs * seq_len, seq_len, -1)
                    expanded = True
                    # print('x shape after expand', x.shape)
                # flip the sequence
                if i > 0:
                    x = x.flip(1)
            
            if self.dropout_rates[i] == 0. or not self.training:
                x = block(x)
            else:
                x = self.drop_path(x, block, self.dropout_rates[i])
            # x = block(x)

        if self.bidirectional and expanded:
            x = x.reshape(bs, seq_len, seq_len, -1)
            # print('x shape after reshape', x.shape)
            # keep only the diagonal
            x = torch.diagonal(x, dim1=1, dim2=2).transpose(1,2)
            # print('x shape after diagonal', x.shape)

        x = self.model.post_blocks_norm(x)
        return x
     
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in the main path of residual blocks)."""
    def __init__(self, is_large_mlstm=False):
        super(DropPath, self).__init__()
        self.is_large_mlstm = is_large_mlstm

    def forward(self, x, block, drop_path_prob, state: mLSTMStateType | None = None):
        if drop_path_prob == 0. or not self.training:
            if self.is_large_mlstm:
                return block(x, state)
            else:
                return block(x)
        
        # indexes of the batch
        idxs = torch.randperm(x.shape[0])
        num_to_keep = int(np.ceil((1.0 - drop_path_prob) * x.shape[0]))
        idxs_to_keep = idxs[:num_to_keep]  # First N elements are kept

        if self.is_large_mlstm:
            out, _ = block(x[idxs_to_keep], None)
            x[idxs_to_keep] = out
            # dont need to have a state in training
            return x, None
        else:
            x[idxs_to_keep] = block(x[idxs_to_keep])
            return x

class mLSTMWrapper(nn.Module):
    def __init__(self, xlstm, dropout=0.2, bidirectional=False, drop_path=0.):
        super(mLSTMWrapper, self).__init__() 
        self.model = xlstm
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.drop_path = DropPath(is_large_mlstm=True)
        self.dropout_rates = [x.item() for x in torch.linspace(0, drop_path, len(self.model.blocks))]

    def forward(self, x, need_expansion=True):
        len_seq = x.shape[1]
        # print('len_seq', len_seq)
        pad_len = (64 - len_seq % 64) % 64
        x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2]).to(x.device)], dim=1)
        x, _ = self.model_forward_wrap(x, need_expansion=need_expansion)
        if pad_len > 0:
           x = x[:, :-pad_len, :]
        
        return x
    
    def step(self, x, state):
        len_seq = x.shape[1]
        pad_len = max(16 - len_seq, 2**int(np.ceil(np.log2(len_seq))) - len_seq)
        x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2]).to(x.device)], dim=1)
        x, state = self.model_forward_wrap(x, state)
        return x[:, pad_len:, :], state
    
    def init_layer_weights(self, layer):
        # initialize the weights of the layer
        for name, param in layer.named_parameters():
            if 'weight' in name:
                if len(param.shape) == 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_uniform_(param[0])
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def model_forward_wrap(self, x, state = None, need_expansion=True):
        # print('x shape before model', x.shape)

        if state is None:
            state = {i: None for i in range(len(self.model.blocks))}

        expanded = False

        for i, block in enumerate(self.model.blocks):
            if self.bidirectional: 
                if not expanded and i > 0 and need_expansion:
                    bs, seq_len, _ = x.shape
                    x = x.unsqueeze(1).repeat(1, seq_len, 1, 1)
                    tril_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=x.dtype, device=x.device)).unsqueeze(0).unsqueeze(-1)
                    x = x * tril_mask
                    x = x.reshape(bs * seq_len, seq_len, -1)
                    expanded = True
                    # print('x shape after expand', x.shape)
                # flip the sequence
                if i > 0:
                    x = x.flip(1)

            block_state = state[i]
            x = self.dropout(x)
            
            # print(x.dtype, x.device, x.shape)
            with torch.amp.autocast('cuda', enabled=False):
               x, block_state_new = self.drop_path(x, block, self.dropout_rates[i], state=block_state)
            #x, block_state_new = block(x, block_state)

            if block_state is None:
                state[i] = block_state_new
            else:
                # layer state is a tuple of three tensors: c, n, m
                # we update the state in place in order to avoid creating new tensors
                for state_idx in range(len(block_state)):
                    state[i][state_idx].copy_(block_state_new[state_idx])

        if self.bidirectional and expanded:
            x = x.reshape(bs, seq_len, seq_len, -1)
            # print('x shape after reshape', x.shape)
            # keep only the diagonal
            x = torch.diagonal(x, dim1=1, dim2=2).transpose(1,2)
            # print('x shape after diagonal', x.shape)

        x = self.model.out_norm(x)

        return x, state