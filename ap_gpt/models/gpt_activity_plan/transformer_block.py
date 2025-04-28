import torch
import torch.nn as nn

from ap_gpt.entity.config_entity import  ModelTrainerConfig

from .self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelTrainerConfig):
        super(TransformerBlock, self).__init__()

        embed_size, dropout, forward_expansion = config.embed_size, config.dropout, config.forward_expansion

        self.attention = SelfAttention(config)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask):
        attention = self.attention(X, mask)
        attention = self.dropout(attention)
        x = self.norm1(attention + X)

        forward = self.dropout(self.feed_forward(x))
        out = self.norm2(forward + x)
        return out
