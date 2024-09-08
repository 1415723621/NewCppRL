import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule, InteractionType, NormalParamExtractor
from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

from torchrl_utils.model.deep_q_net import DeepQNet
from torchrl_utils.utils_env import make_env


def make_sac_modules(proof_environment):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    action_spec = proof_environment.action_spec
    if proof_environment.batch_size:
        action_spec = action_spec[(0,) * len(proof_environment.batch_size)]

    # Define distribution class and kwargs
    distribution_class = TanhNormal
    # distribution_class = TruncatedNormal
    distribution_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": False,
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
        output_num=2 * action_spec.shape[-1],
        cnn_activation_class=torch.nn.ReLU,
        mlp_activation_class=torch.nn.ReLU,
        action_head=NormalParamExtractor(
            scale_mapping=f"biased_softplus_{1.0}",
            scale_lb=0.1,
        ),
    )
    # policy_net = MLP(
    #     in_features=1,
    #     out_features=2 * action_spec.shape[-1],
    #     num_cells=[],
    #     activation_class=None,
    # )
    # policy_net = torch.nn.Sequential(
    #     policy_net,
    #     NormalParamExtractor(
    #         scale_mapping=f"biased_softplus_{1.0}",
    #         scale_lb=0.1,
    #     ),
    # )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=in_keys,
        out_keys=["loc", "scale"],
    )
    policy_module = ProbabilisticActor(
        spec=action_spec,
        module=policy_module,
        in_keys=["loc", "scale"],
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
        vec_dim=1 + 2,
        hidden_dim=encoder_out_dim,
        output_num=1,
        cnn_activation_class=torch.nn.ReLU,
        mlp_activation_class=torch.nn.ReLU,
    )
    # qvalue_net = MLP(
    #     in_features=1 + 2,
    #     out_features=1,
    #     num_cells=[],
    # )
    qvalue_module = ValueOperator(
        in_keys=in_keys + ["action"],
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
            td_ = net(td)
            pass
        del td

    del proof_environment

    return actor_critic
