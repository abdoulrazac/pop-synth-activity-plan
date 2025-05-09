import torch
import torch.nn as nn

from ap.entity.config_entity import ModelTrainerConfig
from .transformer_block import TransformerBlock


class ActionGPT(nn.Module):
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        super(ActionGPT, self).__init__()

        embed_size, num_layers, vocab_size, dropout, hidden_dim = (
            model_trainer_config.embed_size, model_trainer_config.num_layers,
            model_trainer_config.vocab_size, model_trainer_config.dropout,
            model_trainer_config.hidden_dim
        )

        self.pad_token_idx = model_trainer_config.pad_token_idx
        self.max_sequence_length = model_trainer_config.max_sequence_length
        self.device = model_trainer_config.device

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(self.max_sequence_length, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(model_trainer_config) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)

        ## FC for each output (action, duration, distance)
        outer_dim = self.max_sequence_length * embed_size
        self.fc_action_out = nn.Sequential(
            nn.Linear(outer_dim, outer_dim),
            nn.ReLU(),
            nn.Linear(outer_dim, model_trainer_config.name_vocab_size["action"])
        )
        self.fc_duration_out = nn.Sequential(
            nn.Linear(outer_dim, outer_dim),
            nn.ReLU(),
            nn.Linear(outer_dim, model_trainer_config.name_vocab_size["duration"])
        )
        self.fc_distance_out = nn.Sequential(
            nn.Linear(outer_dim, outer_dim),
            nn.ReLU(),
            nn.Linear(outer_dim, model_trainer_config.name_vocab_size["distance"])
        )

    def make_mask(self, x):
        pad_token_idx = torch.tensor(self.pad_token_idx, device=x.device)
        mask = (~torch.isin(x, pad_token_idx)).unsqueeze(1).unsqueeze(2)
        return mask

    def forward(self, x):
        N, seq_length = x.shape

        # Déplacer positions vers le même dispositif que x
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        # Assurer que pad_token_idx est sur le bon dispositif
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
