import torch
import torch.nn as nn

from ap.entity.config_entity import ModelTrainerConfig


class SelfAttention(nn.Module) :
    def __init__(self, config : ModelTrainerConfig) :
        super(SelfAttention, self).__init__()
        self.embed_size = config.embed_size
        self.heads = config.heads
        self.head_dim = self.embed_size // self.heads
        self.device = config.device

        # check if head_dim is an integer
        assert self.head_dim * self.heads == self.embed_size, "Embed size needs to be divisible by heads"

        self.values  = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device=self.device)
        self.keys    = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device=self.device)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device=self.device)

        self.fc_out  = nn.Linear(self.heads*self.head_dim, self.embed_size).to(device=self.device)

    def forward(self, X, mask) :
        N = X.shape[0] # N = Batch Size
        value_len, key_len, query_len = X.shape[1], X.shape[1], X.shape[1]

        # Split embedding into self.heads pieces
        values = X.reshape(N, value_len, self.heads, self.head_dim)
        keys = X.reshape(N, key_len, self.heads, self.head_dim)
        queries = X.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape : (N, query_len, heads, head_dim)
        # keys shape : (N, key_len, heads, head_dim)
        # energy shape : (N, heads, query_len, key_len)

        if mask is not None :
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size**(1/2)), dim=3)

        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape : (N, heads, query_len, key_len)
        # values shape : (N, key_len, heads, head_dim)
        # after einsum (N, query_len, heads, head_dim) the flatten last two dimensions

        out = self.fc_out(out)
        return out
