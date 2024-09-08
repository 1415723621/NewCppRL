from typing import Sequence, Optional

from torch import nn

from torchrl_utils.model.conv_encoder import ConvEncoder
from torchrl_utils.model.dueling_head import DuelingHead


class DeepQNet(nn.Module):
    def __init__(self,
                 raster_shape: Sequence[int],
                 cnn_channels: Sequence[int] = (16, 32, 64),
                 kernel_sizes: Sequence[int] = (8, 4, 3),
                 strides: Sequence[int] = (1, 1, 1),
                 vec_dim=14,
                 hidden_dim=256,
                 output_num=15,
                 cnn_activation_class: Optional[nn.Module] = nn.ELU,
                 mlp_activation_class: Optional[nn.Module] = nn.ReLU,
                 dueling_head: bool = False,
                 action_head: Optional[nn.Module] = None):
        super(DeepQNet, self).__init__()
        self.encoder = ConvEncoder(
            raster_shape=raster_shape,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            vec_dim=vec_dim,
            vec_out=hidden_dim,
            cnn_activation_class=cnn_activation_class,
            mlp_activation_class=mlp_activation_class,
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        if mlp_activation_class:
            self.q_head.append(mlp_activation_class(inplace=False))
        if dueling_head:
            self.q_head.append(DuelingHead(hidden_dim, output_num))
        else:
            self.q_head.append(nn.Linear(hidden_dim, output_num))
        # self.action_head = action_head
        self.action_head = action_head

    def forward(self, observation, vector=None, action=None):
        embed = self.encoder(observation, vector, action)
        q_values = self.q_head(embed)
        if self.action_head:
            q_values = self.action_head(q_values)
        return q_values
