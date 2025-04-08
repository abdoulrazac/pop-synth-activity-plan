import torch.nn as nn

# BaseClass for all models;
class APBaseModel(nn.Module):
    def __init__(self, config):
        super(APBaseModel, self).__init__()
        self.config = config