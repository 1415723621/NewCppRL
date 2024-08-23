from typing import Sequence

from torch import nn

from torchrl_utils.model.conv_encoder import ConvEncoder
from torchrl_utils.model.dueling_head import DuelingHead


class DeepQNet(nn.Module):
    def __init__(self,
                 raster_shape: Sequence[int],
                 cnn_channels: Sequence[int] = (16, 32, 64),
                 kernel_sizes: Sequence[int] = (8, 4, 3),
                 strides: Sequence[int] = (1, 1, 1),
                 obs_dim=14,
                 hidden_dim=256,
                 action_num=15):
        super(DeepQNet, self).__init__()
        self.encoder = ConvEncoder(
            raster_shape=raster_shape,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            vec_dim=obs_dim,
            vec_out=hidden_dim,
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            DuelingHead(hidden_dim, action_num)
        )

    def forward(self, observation, vector=None):
        embed = self.encoder(observation, vector)
        q_values = self.q_head(embed)
        return q_values
