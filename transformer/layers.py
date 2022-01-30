import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)
        assert q.size(-1) == d_k

        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(d_k)

        if mask is not None:
            attn = attn.masked_fill(mask == False, -1e9)
        p_attn = F.softmax(attn, dim=-1)
        p_attn = self.dropout(p_attn)
        output = torch.bmm(p_attn, v)
        return output


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_feature, dropout=0.1):
        super(AttentionHead, self).__init__()
        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.q_tfm = nn.Linear(d_model, d_feature)
        self.k_tfm = nn.Linear(d_model, d_feature)
        self.v_tfm = nn.Linear(d_model, d_feature)

    def forward(self, queries, keys, values, mask=None):
        Q = self.q_tfm(queries)
        K = self.k_tfm(keys)
        V = self.v_tfm(values)
        output = self.attn(Q, K, V, mask=mask)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        assert d_model == d_feature * n_heads

        self.attn_heads = nn.ModuleList([AttentionHead(d_model, d_feature, dropout=dropout) for _ in range(n_heads)])
        self.projection = nn.Linear(d_feature*n_heads, d_model)

    def forward(self, queries, keys, values, mask=None):
        x = [attn(queries, keys, values, mask=mask) for attn in self.attn_heads]
        x = torch.cat(x, dim=-1)
        x = self.projection(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return output

class BatchNorm(object):
    """docstring for BatchNorm"""
    def __init__(self, arg):
        super(BatchNorm, self).__init__()
        self.arg = arg

class ContextEmbed(nn.Module):
    """docstring for ContextEmbed"""
    def __init__(self,d_feature = 128):
        super(ContextEmbed, self).__init__()
        self.placeholder = torch.randn_like([1,d_feature])
        self.project = nn.Linear(3*d_feature,d_feature)
    def forward(self,x,t):
        # output context_embed vector
        graph_embed = x.mean(-1) 
        if t ==1:
            hidden_context = graph_embed+placeholder+placeholder 
        else:
            hidden_context = graph_embed+previous_node+first_node
        out = project(hidden_context)
        return out
        
