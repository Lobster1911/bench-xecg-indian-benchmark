from torch import nn
import torch 
import numpy as np
from torch.nn import functional as F
from xlstm.xlstm_large.model import mLSTMStateType


class LinearPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.conv = nn.Conv1d(num_channels, num_hiddens, kernel_size=patch_size, stride=patch_size, bias=False)

    # @torch.compiler.disable
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
    


class ChannelAttentivePatchEmbedding(nn.Module):
    def __init__(self, patch_size=25, num_hiddens=256, num_channels=1, num_heads=4):
        super().__init__()
        self.num_channels = num_channels
        self.num_hiddens = num_hiddens

        # ------------------------------------------------------------------
        # 1. INDEPENDENT TEMPORAL PROCESSING (Grouped Convolutions)
        # ------------------------------------------------------------------
        # "groups=num_channels" tells PyTorch: 
        # "Don't mix the leads yet. Apply the same filter to each lead independently."
        # This is mathematically identical to running a loop over channels, but 100x faster/lighter.
        
        # Determine strides (Same logic as before)
        if patch_size % 4 == 0:
            s1, s2 = 4, patch_size // 4
            k1, p1 = 15, 7
        elif patch_size % 5 == 0:
            s1, s2 = 5, patch_size // 5
            k1, p1 = 11, 5
        elif patch_size % 2 == 0:
            s1, s2 = 2, patch_size // 2
            k1, p1 = 7, 3
        else:
            s1, s2 = patch_size, 1
            k1, p1 = patch_size, 0

        mid_channels = num_hiddens // 2
        
        # We output (num_channels * mid_channels) filters.
        # Because groups=num_channels, input channel i only goes to output channels [i*mid : (i+1)*mid]
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels, 
                out_channels=num_channels * mid_channels, 
                kernel_size=k1, stride=s1, padding=p1, 
                groups=num_channels, bias=False # <--- KEY OPTIMIZATION
            ),
            nn.BatchNorm1d(num_channels * mid_channels),
            nn.GELU(),
            
            nn.Conv1d(
                in_channels=num_channels * mid_channels, 
                out_channels=num_channels * num_hiddens, 
                kernel_size=s2 if s2 > 1 else 1, stride=s2, 
                groups=num_channels, bias=False # <--- KEY OPTIMIZATION
            ),
            nn.BatchNorm1d(num_channels * num_hiddens),
            nn.GELU()
        )

        self.channel_pos_embed = nn.Parameter(torch.randn(1, num_channels, num_hiddens) * 0.02)


        # ------------------------------------------------------------------
        # 2. SPATIAL CHANNEL ATTENTION (The "Very Small Transformer")
        # ------------------------------------------------------------------
        # We use a single Transformer Layer to let channels "talk" to each other
        self.channel_mixer = nn.TransformerEncoderLayer(
            d_model=num_hiddens,
            nhead=num_heads,
            dim_feedforward=num_hiddens * 2, # Keep it small/efficient
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-Norm is generally more stable
        )
        
        # ------------------------------------------------------------------
        # 3. AGGREGATION
        # ------------------------------------------------------------------
        # Learnable weight to combine channels, or we can just use MeanPool
        # Using a small linear layer to smooth the transition after MeanPooling
        self.out_proj = nn.Sequential(
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, num_hiddens)
        )

    @torch._dynamo.disable
    def forward(self, x, permute=True):
        # x: [Batch, Length, Channels]
        # print('x shape at ChannelAttentivePatchEmbedding input', x.shape)
        if permute: 
            x = x.permute(0, 2, 1) 
        
        B, _, _ = x.shape
        # print('x shape after permute', x.shape)
        
        # -------------------------------------------------
        # Step 1: Independent Channel Processing
        # -------------------------------------------------
        # Reshape to treat every channel as an independent sample
        # [B, C, L] -> [B*C, 1, L]
        # x_flat = x.reshape(B * C, 1, L)
        # print('x shape after flatten', x_flat.shape)
        
        # Apply Conv Stem
        # Output: [B*C, num_hiddens, Num_Patches]
        feat = self.stem(x)
        # print('feat shape after stem', feat.shape)
        
        _, _, N_Patches = feat.shape
        
        # -------------------------------------------------
        # Step 2: Reshape for Channel Attention
        # -------------------------------------------------
        # We need [Batch, Num_Patches, Channels, Hidden]
        # First: [B*C, H, N] -> [B, C, H, N]
        feat = feat.view(B, self.num_channels, self.num_hiddens, N_Patches)
        # print('feat shape after view', feat.shape)
        
        # Permute to: [B, N_Patches, C, H]
        # This aligns the "Sequence" for the transformer to be the Channels (C)
        feat = feat.permute(0, 3, 2, 1) 
        # print('feat shape after permute', feat.shape)

        # Merge Batch and Time to treat each (TimeStep) as an independent attention problem
        # [B*N, C, H]
        feat_for_att = feat.reshape(B * N_Patches, self.num_channels, self.num_hiddens)
        # print('feat shape after reshape for attention', feat_for_att.shape)

        # add position embedding
        feat_for_att = feat_for_att + self.channel_pos_embed

        # -------------------------------------------------
        # Step 3: Apply Attention
        # -------------------------------------------------
        # The Transformer attends across C (12 leads).
        # It learns relationships like "Lead V1 is noisy, trust Lead II more"
        feat_attended = self.channel_mixer(feat_for_att) # [B*N, C, H]
        # print('feat shape after channel attention', feat_attended.shape)
        
        # -------------------------------------------------
        # Step 4: Aggregation (Fusion)
        # -------------------------------------------------
        # We need to collapse the Channel dimension to get 1 vector per TimeStep
        # Global Average Pooling across channels is robust here
        feat_fused = feat_attended.mean(dim=1) # [B*N, H]
        # print('feat shape after channel fusion', feat_fused.shape)
        
        # Unpack back to [Batch, Num_Patches, H]
        feat_out = feat_fused.view(B, N_Patches, self.num_hiddens)
        # print('feat shape after final reshape', feat_out.shape)
        
        # Final projection/Norm
        feat_out = self.out_proj(feat_out)
        # print('feat shape after out_proj', feat_out.shape)
        
        return feat_out
    
    
      
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
                    x = x.flip(1).contiguous()
            
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
        
        # Create mask instead of indexing
        batch_size = x.shape[0]
        keep_prob = 1.0 - drop_path_prob
        mask = torch.rand(batch_size, device=x.device) < keep_prob
        
        if self.is_large_mlstm:
            out, _ = block(x, None)
            # Use where instead of in-place assignment
            x = torch.where(mask.view(-1, 1, 1), out, x)
            return x, None
        else:
            out = block(x)
            # Use where instead of in-place assignment
            x = torch.where(mask.view(-1, 1, 1), out, x)
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