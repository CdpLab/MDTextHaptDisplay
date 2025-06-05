import torch
import torch.nn as nn
import math


class TruncateModule(nn.Module):
    def __init__(self, target_length):
        super(TruncateModule, self).__init__()
        self.target_length = target_length

    def forward(self, x, truncate_length):
        return x[:, :, :truncate_length]


def PositionalEncoding(q_len, d_model, normalize):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


class PositionalEmbedding(nn.Module):
    def __init__(self, q_len=5000, d_model=128, pos_embed_type='sincos', learnable=False, r_layers=1, c_in=21, scale=1):
        super(PositionalEmbedding, self).__init__()
        self.pos_embed_type = pos_embed_type
        self.learnable = learnable
        self.scale = scale
        if pos_embed_type is None:
            W_pos = torch.empty((q_len, d_model))
        elif pos_embed_type == 'zero':
            W_pos = torch.empty((q_len, 1))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pos_embed_type == 'zeros':
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pos_embed_type == 'normal' or pos_embed_type == 'gauss':
            W_pos = torch.zeros((q_len, 1))
            torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
        elif pos_embed_type == 'uniform':
            W_pos = torch.zeros((q_len, 1))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)
        elif pos_embed_type == 'random':
            W_pos = torch.rand(c_in, q_len, d_model)
        elif pos_embed_type == 'sincos':
            W_pos = PositionalEncoding(q_len, d_model, normalize=True)
        elif pos_embed_type == 'rnn':
            W_pos = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=r_layers)
        else:
            raise ValueError(f"{pos_embed_type} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
        if 'rnn' in pos_embed_type:
            self.pos = W_pos
        else:
            W_pos = W_pos.unsqueeze(0)
            if learnable:
                self.pos = nn.Parameter(W_pos, requires_grad=learnable)
            else:
                self.register_buffer('pos', W_pos)
                self.pos = W_pos

    def forward(self, x):
        if 'rnn' in self.pos_embed_type:
            output, _ = self.pos(x)
            return output
        if self.pos.dim() > 3:
            batch_size = x.size(0) // self.pos.size(1)
            self.pos = self.pos.repeat(batch_size, 1, 1, 1)
            self.pos = torch.reshape(self.pos, (-1, self.pos.shape[2], self.pos.shape[3]))
            return self.pos
        else:
            return self.pos[:, self.scale - 1:x.size(1) * self.scale:self.scale]


#
class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, patch_len, stride, dropout,
                 process_layer,
                 pos_embed_type, learnable, r_layers=1,
                 ch_ind=1):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.ch_ind = ch_ind
        self.process_layer = process_layer
        self.pos_embed_type = pos_embed_type

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model)

        # Positional embedding
        if self.pos_embed_type is not None:
            self.position_embedding = PositionalEmbedding(seq_len, d_model, pos_embed_type, learnable, r_layers,
                                                          patch_len)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        # x: [B, M, L], x_mark: [B, 4, L]
        n_vars = x.shape[1]
        x = self.process_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        if x_mark is not None and not self.ch_ind:
            x_mark = x_mark.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            x = torch.cat([x, x_mark], dim=1)  # [B, (M+4), N, P]

        # Input value embedding
        x = self.value_embedding(x)  # [B, M, N, D]

        x = torch.reshape(x, (-1, x.shape[2], x.shape[3]))  # [B*M, N, P]

        if self.pos_embed_type is not None:
            x = x + self.position_embedding(x)

        return self.dropout(x), n_vars
