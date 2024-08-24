import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.envs import (
    ExplorationType,
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.modules import QValueActor

from torchrl_utils.model.conv_encoder import ConvEncoder
from torchrl_utils.model.deep_q_net import DeepQNet
from torchrl_utils.utils_env import make_env


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_modules(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # Define distribution class and kwargs
    distribution_class = OneHotCategorical
    distribution_kwargs = {}

    # Define input keys
    in_keys = ["observation", "vector"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    encoder_out_dim = 512
    common_encoder = ConvEncoder(
        raster_shape=input_shape,
        cnn_channels=(32, 64, 64),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        vec_dim=1,
        vec_out=encoder_out_dim,
    )
    # common_cnn = ConvNet(
    #     activation_class=torch.nn.ReLU,
    #     num_cells=[32, 64, 64],
    #     kernel_sizes=[8, 4, 3],
    #     strides=[4, 2, 1],
    # )
    # common_cnn_output = common_cnn(torch.ones(input_shape))
    # common_mlp = MLP(
    #     in_features=common_cnn_output.shape[-1],
    #     activation_class=torch.nn.ReLU,
    #     activate_last_layer=True,
    #     out_features=512,
    #     num_cells=[],
    # )
    # common_mlp_output = common_mlp(common_cnn_output)

    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=common_encoder,
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define on head for the policy
    policy_net = MLP(
        in_features=encoder_out_dim,
        out_features=num_outputs,
        activation_class=torch.nn.ReLU,
        num_cells=[256],
    )
    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=CompositeSpec(action=action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.ReLU,
        in_features=encoder_out_dim,
        out_features=1,
        num_cells=[256],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module


def make_ppo_models():
    proof_environment = make_env(device="cpu")
    common_module, policy_module, value_module = make_ppo_modules(
        proof_environment
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        td = actor_critic(td)
        del td

    del proof_environment

    return actor_critic

# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules(proof_environment):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # layers = [_ConvNetBlock(num_in, num_ch) for num_in, num_ch in ((input_shape[0], 64), (64, 128))]
    # layers += [torch.nn.ReLU(inplace=True), SquashDims()]
    # cnn = torch.nn.Sequential(*layers)
    # cnn_output = cnn(torch.ones(input_shape))
    # mlp = MLP(
    #     in_features=cnn_output.shape[-1],
    #     activation_class=torch.nn.ReLU,
    #     out_features=512,
    #     activate_last_layer=True,
    #     # num_cells=[512],
    # )
    # dueling_head = DuelingHead(embed_dim=512, num_actions=num_outputs)
    # dqn = torch.nn.Sequential(cnn, mlp, dueling_head)
    dqn = DeepQNet(
        raster_shape=input_shape,
        cnn_channels=(32, 64, 64),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        obs_dim=1,
        hidden_dim=512,
        action_num=num_outputs,
    )
    # dqn = MLP(
    #     in_features=input_shape[-1],
    #     activation_class=torch.nn.ReLU,
    #     out_features=num_outputs,
    #     num_cells=[128, 256],
    # )

    qvalue_module = QValueActor(
        module=dqn,
        spec=CompositeSpec(action=action_spec),
        in_keys=["observation", "vector"],
    )
    return qvalue_module


def make_dqn_model():
    proof_environment = make_env(device="cpu")
    qvalue_module = make_dqn_modules(proof_environment)
    del proof_environment
    return qvalue_module
