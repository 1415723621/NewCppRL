from typing import Sequence

import torch
from torch import nn
from torchrl.modules.models.utils import SquashDims

from torchrl_utils.model.impala_net import _ConvNetBlock


class ConvEncoder(nn.Module):
    def __init__(self,
                 raster_shape: Sequence[int],
                 cnn_channels: Sequence[int] = (16, 32, 64),
                 kernel_sizes: Sequence[int] = (8, 4, 3),
                 strides: Sequence[int] = (1, 1, 1),
                 vec_dim=14,
                 vec_out=256,
                 ):
        super(ConvEncoder, self).__init__()
        in_ch = raster_shape[0]
        layers = []
        for i in range(len(cnn_channels)):
            layers += [_ConvNetBlock(
                in_ch, cnn_channels[i], kernel_size=kernel_sizes[i], stride=strides[i]
            )]
            in_ch = cnn_channels[i]
        layers += [torch.nn.ReLU(inplace=True), SquashDims()]
        self.cnn_encoder = torch.nn.Sequential(*layers)
        cnn_output = self.cnn_encoder(torch.ones(raster_shape))
        self.post_encoder = nn.Sequential(
            nn.Linear(vec_dim + cnn_output.size(0), vec_out),
            nn.ReLU(),
            nn.Linear(vec_out, vec_out),
            nn.ReLU(),
        )

    def forward(self, observation, vector=None):
        embed = self.cnn_encoder(observation)
        if vector is not None:
            embed = torch.concatenate([
                embed,
                vector,
            ], dim=-1)
        embed = self.post_encoder(embed)
        return embed
