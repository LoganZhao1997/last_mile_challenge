import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers import ScaledDotProductAttention, AttentionHead, MultiHeadAttention,LayerNorm


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, d_feature=128, d_ff=1024, n_heads=8, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attn_head = MultiHeadAttention(d_model, d_feature,n_heads,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_head = self.attn_head(x,x,x,mask=mask)
        x = x + self.dropout(self.layer_norm1(attn_head))

        pos = self.ff(x)
        x = x + self.dropout(self.layer_norm2(pos))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, d_feature=128, d_ff=1024, n_heads=8, dropout=0.1, n_blocks=6):
        super(TransformerEncoder, self).__init__()
        self.encoders = nn.ModuleList(
            [EncoderBlock(d_model=d_model, d_feature=d_model // n_heads, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_blocks)]
        )

    def forward(self, x, mask=None):
        for encoder in self.encoders:
            x = encoder(x, mask=mask)
        return x