from pathlib import Path

import torch
from torchrl.data import CompositeSpec, OneHotDiscreteTensorSpec
from torchrl.modules import MLP, QValueActor
from torchrl.modules.models.utils import SquashDims
import gymnasium as gym
import envs  # noqa
from model.impala_net import _ConvNetBlock
from torchrl_utils.model.dueling_head import DuelingHead


def get_dqn(state_dict_path: str):
    env = gym.make(
        'Pasture',
        render_mode=None,
    )
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    layers = [_ConvNetBlock(num_in, num_ch) for num_in, num_ch in ((input_shape[0], 64), (64, 128))]
    layers += [torch.nn.ReLU(inplace=True), SquashDims()]
    cnn = torch.nn.Sequential(*layers)
    # cnn = ConvNet(
    #     activation_class=torch.nn.ReLU,
    #     num_cells=[32, 64, 64],
    #     kernel_sizes=[8, 4, 3],
    #     strides=[1, 1, 1],
    # )
    cnn_output = cnn(torch.ones(input_shape))
    mlp = MLP(
        in_features=cnn_output.shape[-1],
        activation_class=torch.nn.ReLU,
        out_features=512,
        activate_last_layer=True,
        # num_cells=[512],
    )
    dueling_head = DuelingHead(embed_dim=512, num_actions=num_actions)
    dqn = torch.nn.Sequential(cnn, mlp, dueling_head)
    qvalue_module = QValueActor(
        module=dqn,
        spec=CompositeSpec(action=OneHotDiscreteTensorSpec(n=num_actions)),
        in_keys=["observation"],
    )
    qvalue_module.load_state_dict(torch.load(Path(__file__).parent / state_dict_path))
    return qvalue_module
