import torch
import torch.nn as nn

from ap_gpt.entity.config_entity import ModelTrainerConfig
from .transformer_block import TransformerBlock
from ..base_model import BaseModel


class ActionGPT(BaseModel, nn.Module):
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        super(ActionGPT, self).__init__()

        embed_size, num_layers, vocab_size, dropout = (
            model_trainer_config.embed_size, model_trainer_config.num_layers,
            model_trainer_config.vocab_size, model_trainer_config.dropout
        )

        self.pad_token_idx = model_trainer_config.pad_token_idx
        self.max_sequence_length = model_trainer_config.max_sequence_length
        self.device = model_trainer_config.device

        self.word_embedding = nn.Embedding(vocab_size, embed_size).to(device=self.device)
        self.position_embedding = nn.Embedding(self.max_sequence_length, embed_size).to(device=self.device)
        self.layers = nn.ModuleList([TransformerBlock(model_trainer_config) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout).to(device=self.device)
        self.norm = nn.LayerNorm(embed_size).to(device=self.device)

        ## FC for each output (action, duration, distance)
        self.fc_action_out = nn.Linear(self.max_sequence_length * embed_size, model_trainer_config.name_vocab_size["action"]).to(device=self.device)
        self.fc_duration_out = nn.Linear(self.max_sequence_length * embed_size, model_trainer_config.name_vocab_size["duration"]).to(device=self.device)
        self.fc_distance_out = nn.Linear(self.max_sequence_length * embed_size, model_trainer_config.name_vocab_size["distance"]).to(device=self.device)

    def make_mask(self, x):
        pad_token_idx = torch.tensor(self.pad_token_idx, device=x.device)
        pad_token_mask = torch.isin(x, pad_token_idx)
        mask = ~pad_token_mask.unsqueeze(1).unsqueeze(2)
        return mask

    def forward(self, x):
        N, seq_length = x.shape

        # Déplacer positions vers le même dispositif que x
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        # Assurer que pad_token_idx est sur le bon dispositif
        pad_token_idx = torch.tensor(self.pad_token_idx, device=x.device)
        mask = self.make_mask(x)

        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        x = x.reshape(N, -1)

        # FC pour chaque sortie
        action_out = torch.softmax(self.fc_action_out(x), dim=1)
        duration_out = torch.softmax(self.fc_duration_out(x), dim=1)
        distance_out = torch.softmax(self.fc_distance_out(x), dim=1)

        return action_out, duration_out, distance_out
