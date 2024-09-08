import torch.nn
from torchrl.data import CompositeSpec
from torchrl.modules import QValueActor

from torchrl_utils.model.deep_q_net import DeepQNet
from torchrl_utils.utils_env import make_env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules(proof_environment):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs # 用于提取环境的各种属性和空间规格
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n #表示多重嵌套顺次索引
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
        cnn_channels=(32, 64, 128),
        kernel_sizes=(3, 3, 3),
        strides=(1, 1, 1),
        vec_dim=1,
        hidden_dim=512,
        output_num=num_outputs,
        cnn_activation_class=torch.nn.ReLU,
        mlp_activation_class=torch.nn.ReLU,
    )
    # dqn = MLP(
    #     in_features=input_shape[-1],
    #     activation_class=torch.nn.ReLU,
    #     out_features=num_outputs,
    #     num_cells=[128, 256],
    # )

    qvalue_module = QValueActor(
        module=dqn,
        spec=CompositeSpec(action=action_spec), # 包括动作空间的规格说明
        in_keys=["observation", "vector"],
    )
    return qvalue_module


def make_dqn_model():
    """
    生成网络预测模型
    """
    proof_environment = make_env(device="cpu")
    qvalue_module = make_dqn_modules(proof_environment)
    del proof_environment
    return qvalue_module
