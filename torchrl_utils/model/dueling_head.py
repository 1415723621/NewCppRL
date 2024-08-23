import torch
from torch import nn


class DuelingHead(nn.Module):
    def __init__(self, embed_dim: int, num_actions: int):
        super(DuelingHead, self).__init__()
        self.value_fc = nn.Linear(embed_dim, 1)
        self.advantage_fc = nn.Linear(embed_dim, num_actions)

    def forward(self, x):
        value: torch.Tensor = self.value_fc(x)
        advantage: torch.Tensor = self.advantage_fc(x)
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_value
