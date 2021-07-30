import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers import ScaledDotProductAttention, AttentionHead, MultiHeadAttention,LayerNorm,ContextEmbed

class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, d_feature=128, d_ff=1024, n_heads=8, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.context= ContextEmbed(d_feature)
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads,dropout=dropout)
        self.ff = nn. Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.layer_norm1 = LayerNorm(d_model)
        # self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context,x, enc_out, src_mask=None, tgt_mask=None):
        masked_attn = self.masked_attn_head(context, x, x,mask=tgt_mask)
        x = x + self.dropout(self.layer_norm1(masked_attn))

        pos = self.ff(x)
        x = x + self.dropout(self.layer_norm3(pos))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, d_feature=128, d_ff=1024, n_heads=8, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.decoders = nn.ModuleList(
            [DecoderBlock(d_model=d_model, d_feature=d_model // n_heads,n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_blocks)]
        )

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return x