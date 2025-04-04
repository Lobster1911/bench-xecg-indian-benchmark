from torch import nn
import torch 
import numpy as np
from fastonn import SelfONN1d
from torch.nn import functional as F

class LinearPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.conv = nn.Conv1d(num_channels, num_hiddens, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x
    
class EnrichedLinearPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12, enrich_dim=64, kernel_size=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.erich_conv1 = nn.Conv1d(num_channels, enrich_dim, kernel_size=kernel_size, padding=0)
        self.erich_conv2 = nn.Conv1d(enrich_dim, enrich_dim, kernel_size=kernel_size, padding=0)
        self.conv = nn.Conv1d(num_channels + enrich_dim * 2, num_hiddens, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        # x [bs, num_channels, num_samples]
        # apply padding of kernel_size - 1 on the left side
        x_padded = F.pad(x, (self.kernel_size - 1, 0))
        enriched_x1 = self.erich_conv1(x_padded) # [bs, enrich_dim, num_samples]
        enriched_x1_padded = F.pad(enriched_x1, (self.kernel_size - 1, 0))
        enriched_x2 = self.erich_conv2(enriched_x1_padded)

        x = torch.cat([x, enriched_x1, enriched_x2], dim=1) # [bs, num_channels + enrich_dim, num_samples]
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x
      
class EmbedPatching(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12, use_pre_head=False):
        super().__init__()
        self.use_pre_head = use_pre_head  
        if use_pre_head: self.pre_head = HeadModule(num_hiddens, num_hiddens // 2, num_hiddens)
        self.deconv = nn.ConvTranspose1d(num_hiddens, num_channels, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        if self.use_pre_head: x = self.pre_head(x)
        x = x.transpose(1, 2)
        x = self.deconv(x).transpose(1, 2)
        # print('x shape after deconv', x.shape) [1, 3584, 12]
        return x
    
class ConvPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv1d(num_channels, num_hiddens // 4, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm1d(num_hiddens // 4)
        self.conv2 = nn.Conv1d(num_hiddens // 4, num_hiddens // 2, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(num_hiddens // 2)
        self.conv3 = nn.Conv1d(num_hiddens // 2, num_hiddens, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(num_hiddens)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.activation = nn.ReLU()

        # Calculate the output size after the convolutions and pooling
        out_size = ((patch_size - 7 + 1) // 2 - 5 + 1) // 2 - 3 + 1
        out_size = (out_size // 2) * num_hiddens

        self.linear = nn.Linear(out_size, num_hiddens)

    def forward(self, x):
        # transform [bs, n_channels, n_samples] -> [bs, n_channels, n_patches, patch_size]
        x = x.unfold(2, self.patch_size, self.patch_size).transpose(1, 2)
        batch_size, n_patches, n_channels, _ = x.shape
        x = x.reshape(-1, n_channels, self.patch_size) # [bs * n_patches, n_channels, patch_size]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.flatten(1)
        # print('x shape after conv', x.shape)
        x = self.linear(x)

        x = x.reshape(batch_size, n_patches, -1)

        # print('x shape after unfold', x.shape)
        return x
    
    
class ONNConvPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.patch_size = patch_size
        self.conv1 = SelfONN1d(num_channels, num_hiddens // 4, kernel_size=3, stride=1, q=3)
        self.bn1 = nn.BatchNorm1d(num_hiddens // 4)
        self.conv2 = SelfONN1d(num_hiddens // 4, num_hiddens, kernel_size=3, stride=1, q=3)
        self.bn2 = nn.BatchNorm1d(num_hiddens)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.activation = nn.Tanh()

        # Calculate the output size after the convolutions and pooling
        out_size = (patch_size - 3 + 1) // 2 - 3 + 1
        out_size = (out_size // 2) * num_hiddens

        self.linear = nn.Linear(out_size, num_hiddens)

    def forward(self, x):
        # print('x shape', x.shape)
        # transform [bs, n_channels, n_samples] -> [bs, n_channels, n_patches, patch_size]
        x = x.unfold(2, self.patch_size, self.patch_size).transpose(1, 2)
        batch_size, n_patches, n_channels, _ = x.shape
        x = x.view(-1, n_channels, self.patch_size) # [bs * n_patches, n_channels, patch_size]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.flatten(1)
        # print('x shape after conv', x.shape)
        x = self.linear(x)

        x = x.view(batch_size, n_patches, -1)

        # print('x shape after unfold', x.shape)
        return x



class HeadModule(nn.Module):
    
    def __init__(self, inp_size, hidden_size, out_size, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_size),
        )
        
    def forward(self, x):
        return self.head(x)
    
class mLSTMWrapper(nn.Module):
    def __init__(self, xlstm, dropout=0.2):
        super(mLSTMWrapper, self).__init__() 
        self.model = xlstm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        len_seq = x.shape[1]
        pad_len = max(16 - len_seq, 2**int(np.ceil(np.log2(len_seq))) - len_seq)
        x = torch.cat([torch.zeros(x.shape[0], pad_len, x.shape[2]).to(x.device), x], dim=1)
        x, _ = self.model_forward_wrap(x)
        return x[:, pad_len:, :]
    
    def step(self, x, state):
        len_seq = x.shape[1]
        pad_len = max(16 - len_seq, 2**int(np.ceil(np.log2(len_seq))) - len_seq)
        x = torch.cat([torch.zeros(x.shape[0], pad_len, x.shape[2]).to(x.device), x], dim=1)
        x, state = self.model_forward_wrap(x, state)
        return x[:, pad_len:, :], state
    
    def model_forward_wrap(self, x, state = None):
        if state is None:
            state = {i: None for i in range(len(self.model.blocks))}

        for i, block in enumerate(self.model.blocks):
            block_state = state[i]
            x = self.dropout(x)
            x, block_state_new = block(x, block_state)

            if block_state is None:
                state[i] = block_state_new
            else:
                # layer state is a tuple of three tensors: c, n, m
                # we update the state in place in order to avoid creating new tensors
                for state_idx in range(len(block_state)):
                    state[i][state_idx].copy_(block_state_new[state_idx])

        x = self.model.out_norm(x)

        return x, state