from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
import torch
import torch.nn as nn
from collections import OrderedDict


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert (
            self.embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True
        )

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        return x
    

class LinearAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, 1),
        )
        nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        attention_weights = self.softmax(self.attention_layer(x))
        return attention_weights.transpose(1, 2) @ x
