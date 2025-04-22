import torch
import torch.nn as nn
import numpy as np

from ap_gpt.entity.config_entity import ModelTrainerConfig

from .transformer_block import TransformerBlock


class ActionGPT(nn.Module):
    def __init__(self, config: ModelTrainerConfig):
        super(ActionGPT, self).__init__()

        embed_size, num_layers, vocab_size, dropout = (
            config.embed_size, config.num_layers, config.vocab_size, config.dropout)

        self.pad_token_idx = config.pad_token_idx
        self.max_sequence_length = config.max_sequence_length
        self.device = config.device

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(self.max_sequence_length, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        ## FC for each output (action, duration, distance)
        self.fc_action_out = nn.Linear(self.max_sequence_length * embed_size, config.name_vocab_size["action"], bias=False)
        self.fc_duration_out = nn.Linear(self.max_sequence_length * embed_size, config.name_vocab_size["duration"], bias=False)
        self.fc_distance_out = nn.Linear(self.max_sequence_length * embed_size, config.name_vocab_size["distance"], bias=False)

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

        # FC for each output
        action_out = self.fc_action_out(x)
        duration_out = self.fc_duration_out(x)
        distance_out = self.fc_distance_out(x)

        return action_out, duration_out, distance_out
