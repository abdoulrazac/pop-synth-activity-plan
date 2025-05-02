import torch.nn as nn

from ap.entity.config_entity import ModelTrainerConfig


class ActionLSTM(nn.Module):
    def __init__(self, model_trainer_config: ModelTrainerConfig) :
        super(ActionLSTM, self).__init__()

        vocab_size, embed_size, dropout, hidden_dim, num_layers = (
            model_trainer_config.vocab_size, model_trainer_config.embed_size, model_trainer_config.dropout,
            model_trainer_config.hidden_dim, model_trainer_config.num_layers
        )

        self.name_vocab_size = model_trainer_config.name_vocab_size
        self.device = model_trainer_config.device

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.mlp_embedding = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.LayerNorm(embed_size),
            nn.ELU(),
            nn.Linear(hidden_dim, embed_size)
        )

        self.lstm = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Output heads
        self.fc_action_out = nn.Linear(hidden_dim, self.name_vocab_size["action"])
        self.fc_duration_out = nn.Linear(hidden_dim, self.name_vocab_size["duration"])
        self.fc_distance_out = nn.Linear(hidden_dim, self.name_vocab_size["distance"])

    def forward(self, x):
        # x: [batch_size, seq_len] (token indices)
        x = self.embedding(x)  # -> [batch_size, seq_len, embed_dim]
        x = self.mlp_embedding(x)
        x, _ = self.lstm(x)

        last_time_step = x[:, -1, :]  # Use last LSTM output

        y1 = self.fc_action_out(last_time_step)
        y2 = self.fc_duration_out(last_time_step)
        y3 = self.fc_distance_out(last_time_step)
        return y1, y2, y3