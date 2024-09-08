from typing import Sequence, Optional

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
                 cnn_activation_class: Optional[torch.nn.Module] = torch.nn.ELU,
                 mlp_activation_class: Optional[torch.nn.Module] = torch.nn.ReLU,
                 ):
        super(ConvEncoder, self).__init__()
        in_ch = raster_shape[0]
        layers = []
        for i in range(len(cnn_channels)):
            layers.extend([_ConvNetBlock(
                in_ch, cnn_channels[i], kernel_size=kernel_sizes[i], stride=strides[i],
                activation_function=cnn_activation_class,
            )])
            in_ch = cnn_channels[i]
        if mlp_activation_class:
            layers.append(mlp_activation_class(inplace=False))
        layers.append(SquashDims())
        self.cnn_encoder = torch.nn.Sequential(*layers)
        dummy_inputs = torch.ones(raster_shape)
        if dummy_inputs.ndim < 4:
            dummy_inputs = dummy_inputs.unsqueeze(0)
        cnn_output = self.cnn_encoder(dummy_inputs)
        self.post_encoder = nn.Sequential(
            nn.Linear(vec_dim + cnn_output.size(1), vec_out),
        )
        if mlp_activation_class:
            self.post_encoder.append(mlp_activation_class(inplace=False))

    def forward(self, observation: torch.Tensor, vector=None, action=None):
        embed = self.cnn_encoder(observation)
        to_be_concat = [embed]
        if vector is not None:
            to_be_concat.append(vector)
        if action is not None:
            to_be_concat.append(action)
        embed = torch.concatenate(to_be_concat, dim=-1)
        embed = self.post_encoder(embed)
        return embed
