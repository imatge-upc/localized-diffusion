import torch
import torch.nn as nn

from collections import OrderedDict

from .activations import QuickGELU

class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from:
    [Visual Reasoning with a General Conditioning Layer](https://arxiv.org/pdf/1709.07871.pdf)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(in_channels))
        self.beta = nn.Parameter(torch.empty(in_channels))
    
    def forward(self, x, z):
        gamma = self.gamma(z)
        beta = self.beta(z)
        output = gamma[None, :, None, None] * x + beta[None, :, None, None]
        return output

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        original_type = x.dtype
        result = super().forward(x.type(torch.float32))
        return result.type(original_type)

class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block for the Fuser Model"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ResnetBlock(nn.Module):
    """Common Resnet Block"""
    def __init__(self, in_c, out_c, down, ksize=3):
        super().__init__()
        padding = ksize // 2
        self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, padding) if in_c != out_c else nn.Identity()
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, padding)
        self.down = nn.AvgPool2d(2, stride=2) if down else nn.Identity()

    def forward(self, x):
        x = self.down(x)
        x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        return h + x 