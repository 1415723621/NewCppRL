import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Sequence

from tensordict.nn import AddStateIndependentNormalScale
from torchrl.modules import MLP, ConvNet
from torchrl.modules.models.utils import SquashDims
from torchvision.transforms.v2.functional import center_crop
from .impala_net import _ConvNetBlock


class SgcnnNet(torch.nn.Module):
    def __init__(
            self,
            cat_dim: int = 0,
            out_dim: int = 2,
            hid_dim: int = 256,
            channels: List[int] = [32, 64, 64],
            kernel_sizes: List[int] = [8, 4, 3],
            strides: List[int] = [4, 2, 1],
            downscale_factor: int = 4,
            init_gain: float = None,
            actor_head=None,
    ) -> None:
        super().__init__()

        self.downscale_factor = downscale_factor
        convs_list = []
        for _ in range(downscale_factor):
            # last_in = 2
            # layers = []
            # for num_ch in channels:
            #     layers.append(_ConvNetBlock(num_in=last_in, num_ch=num_ch))
            #     last_in = num_ch
            # layers += [nn.ReLU(inplace=False), SquashDims()]
            # convs = nn.Sequential(*layers)
            convs = ConvNet(
                activation_class=torch.nn.ReLU,
                num_cells=channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
            )
            convs_list.append(convs)
        self.sgcnn = nn.ModuleList(convs_list)

        self.post_fc = MLP(
            activation_class=torch.nn.ReLU,
            out_features=out_dim,
            activate_last_layer=True,
            num_cells=[hid_dim],
        )
        if init_gain is not None:
            torch.nn.init.orthogonal_(self.post_fc_1.weight, init_gain)
            self.post_fc_1.bias.data.zero_()
            torch.nn.init.orthogonal_(self.post_fc_2.weight, init_gain)
            self.post_fc_2.bias.data.zero_()
        self.actor_head = actor_head
        self.relu = torch.nn.ReLU()

        self.downscale = nn.MaxPool2d(2, 2)

    def forward(self,
                observations: torch.Tensor,
                pose: torch.Tensor = None,
                action: torch.Tensor = None) -> torch.Tensor:
        downscale_time = self.downscale_factor - 1
        h, w = observations.shape[-2], observations.shape[-1]
        origin_ndim = observations.ndim
        if origin_ndim == 3:
            observations = observations.unsqueeze(0)
        scale_grouped_embed = []
        for convs in self.sgcnn:
            # Center crop the img
            obs_crop = torch.nn.functional.interpolate(
                center_crop(
                    observations,
                    [h, w]
                ),
                size=[h, w],
                mode='nearest'
            )
            # Downscale into target size
            for _ in range(downscale_time):
                obs_crop = self.downscale(obs_crop)
            # Pass to conv layer
            # Add into embed list
            scale_grouped_embed.append(convs(obs_crop))
            # Update args for next round
            downscale_time -= 1
            h, w = h // 2, w // 2
        # Cat embeds in last dim
        obs_embed = torch.cat(scale_grouped_embed, dim=-1)
        # If there's action, concat it
        if pose is not None:
            to_be_cat = [obs_embed, pose]
            obs_embed = torch.cat(to_be_cat, dim=-1)
        if action is not None:
            to_be_cat = [obs_embed, action]
            obs_embed = torch.cat(to_be_cat, dim=-1)
        # Pass to the last fc
        obs_embed = self.post_fc(obs_embed)
        # If any actor head for chunking output
        if self.actor_head is not None:
            obs_embed = self.actor_head(obs_embed)
        if origin_ndim == 3:
            obs_embed = obs_embed.squeeze(0)
        return obs_embed
