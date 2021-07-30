import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import TransformerEncoder
from decoder import TransformerDecoder


def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('bool')
    output = torch.from_numpy(mask) == 0
    return output

class Transformer(nn.Module):
    def __init__(self, vocab, N=6, d_model=512, d_ff=2048, d_z=20, h=8, dropout=0.1, pos_max_len=512):
        super(Transformer_VAE, self).__init__()
        self.d_z = d_z
        self.d_model = d_model
        self.encoder = TransformerEncoder(n_blocks=N, d_model=d_model, n_heads=h, d_ff=d_ff, dropout=dropout)
        self.decoder = TransformerDecoder(n_blocks=N, d_model=d_model, n_heads=h, d_ff=d_ff, dropout=dropout)


    def device(self):
        return self.embedding.word_embedding.weight.device

    def forward_encoder(self, x, train=True):
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=PAD).to(self.device())
        src_mask = (x != PAD).unsqueeze(-2).detach()
        x = self.embedding(x)
        x = self.encoder(x, src_mask)
        return x

    def forward_decoder(self, x):
        #concat

        y = self.decoder(inputs, None, None, tgt_mask)

        loss = F.cross_entropy()
        return loss

    def forward(self, x, train=True):
        x = self.forward_encoder(x, train=train)
        loss = self.forward_decoder(z, x)
        return x, loss