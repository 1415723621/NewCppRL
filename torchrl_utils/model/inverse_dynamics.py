import torch
import torch.nn.functional as F
from torch import nn


class InverseDynamic(nn.Module):
    def __init__(self, embed_dim=2, hidden_dim=256, action_num=15):
        super(InverseDynamic, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, action_num),
        )

    def forward(self, s_t, s_tp1):
        input = torch.cat([s_t, s_tp1], dim=-1)
        y = self.encoder(input)
        y = F.softmax(y, dim=-1)
        return y
