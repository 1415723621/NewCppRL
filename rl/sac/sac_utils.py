import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.data import CompositeSpec
from torchrl.modules import (
    OneHotCategorical,
    ProbabilisticActor,
    SafeModule, ReparamGradientStrategy,
)

from torchrl_utils.model.deep_q_net import DeepQNet
from torchrl_utils.utils_env import make_env


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_sac_modules(proof_environment):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # Define distribution class and kwargs
    distribution_class = OneHotCategorical
    distribution_kwargs = {
        "grad_method": ReparamGradientStrategy.RelaxedOneHot
    }

    # Define input keys
    in_keys = ["observation", "vector"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    encoder_out_dim = 512
    policy_net = DeepQNet(
        raster_shape=input_shape,
        cnn_channels=(32, 64, 64),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        vec_dim=1,
        hidden_dim=encoder_out_dim,
        output_num=num_outputs,
        cnn_activation_class=None,
        mlp_activation_class=torch.nn.SiLU,
        dueling_head=False,
    )
    policy_module = SafeModule(
        module=policy_net,
        in_keys=in_keys,
        out_keys=["logits"],
    )
    policy_module = ProbabilisticActor(
        spec=CompositeSpec(action=action_spec),
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )
    qvalue_net = DeepQNet(
        raster_shape=input_shape,
        cnn_channels=(32, 64, 64),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        vec_dim=1,
        hidden_dim=encoder_out_dim,
        output_num=num_outputs,
        cnn_activation_class=None,
        mlp_activation_class=torch.nn.SiLU,
        dueling_head=True,
    )
    qvalue_module = TensorDictModule(
        in_keys=in_keys,
        out_keys=["action_value"],
        module=qvalue_net,
    )
    return policy_module, qvalue_module


def make_sac_models():
    proof_environment = make_env(device="cpu")
    policy_module, qvalue_module = make_sac_modules(
        proof_environment
    )
    actor_critic = torch.nn.ModuleList([policy_module, qvalue_module])

    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        for net in actor_critic:
            net(td)
        del td

    del proof_environment

    return actor_critic
