import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Sequence

from tensordict.nn import AddStateIndependentNormalScale
from torchrl.modules import MLP
from torchrl.modules.models.utils import SquashDims
from torchvision.transforms.v2.functional import center_crop
from .impala_net import _ConvNetBlock


class SingleNet(torch.nn.Module):
    def __init__(
            self,
            cat_dim: int = 0,
            out_dim: int = 2,
            hid_dim: int = 256,
            channels: List[int] = [16, 16, 16, 16],
            init_gain: float = None,
            actor_head=None,
    ) -> None:
        super().__init__()
        last_in = 3
        last_size = 96
        layers = []
        for num_ch in channels:
            last_size //= 2
            layers.append(_ConvNetBlock(num_in=last_in, num_ch=num_ch))
            last_in = num_ch
        layers += [nn.ReLU(inplace=False), SquashDims()]
        self.convs = nn.Sequential(*layers)

        post_fc_dim = last_size * last_size * last_in
        self.post_fc_1 = torch.nn.Linear(in_features=post_fc_dim + cat_dim,
                                         out_features=hid_dim)
        self.post_fc_2 = torch.nn.Linear(in_features=hid_dim,
                                         out_features=out_dim)
        if init_gain is not None:
            torch.nn.init.orthogonal_(self.post_fc_1.weight, init_gain)
            self.post_fc_1.bias.data.zero_()
            torch.nn.init.orthogonal_(self.post_fc_2.weight, init_gain)
            self.post_fc_2.bias.data.zero_()
        self.actor_head = actor_head
        self.relu = torch.nn.ReLU()

    def forward(self,
                observation: torch.Tensor,
                pose: torch.Tensor = None,
                action: torch.Tensor = None) -> torch.Tensor:
        observation = observation.float() / 255.
        dims = [-1, -3, -2]
        observation = observation.permute(
            *list(range(observation.ndimension() - len(dims))), *dims
        )
        obs_embed = self.convs(observation)
        # If there's action, concat it
        if pose is not None:
            to_be_cat = [obs_embed, pose]
            obs_embed = torch.cat(to_be_cat, dim=-1)
        if action is not None:
            to_be_cat = [obs_embed, action]
            obs_embed = torch.cat(to_be_cat, dim=-1)
        # Pass to the last fc
        obs_embed = self.post_fc_1(obs_embed)
        obs_embed = self.relu(obs_embed)
        output = self.post_fc_2(obs_embed)
        # If any actor head for chunking output
        if self.actor_head is not None:
            output = self.actor_head(output)
        return output
