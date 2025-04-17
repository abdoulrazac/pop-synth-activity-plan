import torch
import torch.nn as nn

from ap_gpt.entity.config_entity import ModelTrainerConfig



class SelfAttention(nn.Module) :
    def __init__(self, config : ModelTrainerConfig) :
        super(SelfAttention, self).__init__()
        self.embed_size = config.embed_size
        self.heads = config.heads
        self.head_dim = self.embed_size // self.heads

        # check if head_dim is an integer
        assert self.head_dim * self.heads == self.embed_size, "Embed size needs to be divisible by heads"

        self.values  = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys    = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out  = nn.Linear(self.heads*self.head_dim, self.embed_size)

    def forward(self, values, keys, queries, mask) :
        N = queries.shape[0] # N = Batch Size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

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
