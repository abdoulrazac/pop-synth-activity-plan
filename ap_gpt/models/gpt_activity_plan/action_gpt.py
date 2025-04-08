import torch
import torch.nn as nn
import numpy as np

from ap_gpt.entity.config_entity import ModelConfig

from .transformer_block import TransformerBlock

class ActionGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super(ActionGPT, self).__init__()
        embed_size, num_layers, vocab_size, dropout = (
        config.embed_size, config.num_layers, config.vocab_size, config.dropout)

        self.pad_token_idx = config.pad_token_idx
        self.max_len = config.max_len
        self.device = config.device

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(self.max_len, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        ## FC for each output (action, duration, distance)
        self.fc_action = nn.Linear(self.max_len * embed_size, config.name_vocab_size["action"])
        self.fc_duration = nn.Linear(self.max_len * embed_size, config.name_vocab_size["duration"])
        self.fc_distance = nn.Linear(self.max_len * embed_size, config.name_vocab_size["distance"])

    def make_mask(self, x):
        mask = torch.tensor(~np.isin(x.cpu().numpy(), list(self.pad_token_idx))).unsqueeze(1).unsqueeze(2)
        return mask.to(self.device)

    def forward(self, x, training=False):
        N, seq_length = x.shape

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        mask = self.make_mask(x)

        x = self.word_embedding(x) + self.position_embedding(positions)

        if training:
            x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, x, x, mask, training)

        x = x.reshape(N, -1)

        return self.fc_action(x), self.fc_duration(x), self.fc_distance(x)
