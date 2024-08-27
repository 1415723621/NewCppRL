import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.envs import (
    ExplorationType,
)
from torchrl.modules import (
    ActorValueOperator,
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    ValueOperator,
)

from torchrl_utils.model.conv_encoder import ConvEncoder
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
        cnn_activation_class=None,
        mlp_activation_class=torch.nn.Tanh,
    )
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
        activation_class=torch.nn.Tanh,
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
        activation_class=torch.nn.Tanh,
        in_features=encoder_out_dim,
        out_features=1,
        num_cells=[256],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module


def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.orthogonal_(m.weight, gain=2**0.5)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        # nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        torch.nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


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
    actor_critic.apply(init_weights)

    del proof_environment

    return actor_critic
