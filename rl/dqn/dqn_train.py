import os
import time
import math
import tqdm
import yaml
import tempfile

import numpy as np
import torch.nn
import torch.optim

from pathlib import Path
from omegaconf import DictConfig
from tensordict.nn import TensorDictSequential
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyMemmapStorage
from torchrl.envs import MultiStepTransform
from torchrl.modules import EGreedyModule
from torchrl.objectives import HardUpdate, DQNLoss
from torchrl.record.loggers import get_logger

from rl.dqn.dqn_utils import make_dqn_model
from torchrl_utils import (
    CustomDQNLoss, # 经过放缩log化使值归一在合适的范围传递误差，之前势场其实这么做是合理的
    value_rescale_inv,
    make_env
)

base_dir = Path(__file__).parent.parent.parent
algo_name = 'dqn'


def main(cfg: "DictConfig"):  # noqa: F821
    # 1. 训练准备：创建ckpt地址、配置设备和随机种子
    ckpt_dir = time.strftime('%Y%m%d_%H%M%S', time.localtime())[2:]
    if cfg.ckpt_name:
        ckpt_dir += f'_{cfg.ckpt_name}'
    if not Path(f'{base_dir}/ckpt').exists():
        os.mkdir(f'{base_dir}/ckpt')
    if not Path(f'{base_dir}/ckpt/{algo_name}').exists():
        os.mkdir(f'{base_dir}/ckpt/{algo_name}')
    if not Path(f'{base_dir}/ckpt/{algo_name}/{ckpt_dir}').exists():
        os.mkdir(f'{base_dir}/ckpt/{algo_name}/{ckpt_dir}')

    device = cfg.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 2. 实例化训练模块：预训练模型、策略采样模块、collector、replay buffer、loss module、optimizer、logger
    if cfg.pretrained_model:
        model = torch.load(f'{base_dir}/{cfg.pretrained_model}').to(device)
    else:
        model = make_dqn_model()

    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
        spec=model.spec, # 动作空间
    )
    model_explore = TensorDictSequential( # 顺序构建神经网络模型
        model, # 通过model计算Q值，输出action_n*1的的qQ值估计输出
        greedy_module, # 然后传送到greedy_model进行动作采样
    ).to(device)

    # Create the collector
    collector = MultiaSyncDataCollector( # 环境是同步运行的
        create_env_fn=[lambda: make_env(
            num_envs=1,
            device='cpu',
        )] * cfg.collector.num_envs,
        policy=model_explore,
        frames_per_batch=cfg.collector.frames_per_batch, # 影响数据采集的效率，更小计算负载更多，但是新数据回传更快，帮助见到更多数据
        total_frames=cfg.collector.total_frames,
        device='cpu',
        storing_device='cpu',
        max_frames_per_traj=-1,
        init_random_frames=cfg.collector.init_random_frames, # 收集多少数据后才开始训练
    )

    # Create the replay buffer
    tempdir = tempfile.TemporaryDirectory()
    scratch_dir = tempdir.name
    replay_buffer = TensorDictPrioritizedReplayBuffer(
        alpha=0.7, # TD_Weight**alpha的重要性
        beta=0.5, # 控制重要性采样， 0表示全都一样，1表示完全修正alpha多采样带来的贡献（权重被多采样的平均）
        pin_memory=False, # 设置为True数据保存再内存中，可以加速训练过程，放在内存中可能放不下所以先存下来
        prefetch=10, # 采样前预取多少批次数据，有助于减少数据读取延迟
        storage=LazyMemmapStorage( # 通过内存映射机制讲数据存储在磁盘中
            max_size=cfg.buffer.buffer_size,
            scratch_dir=scratch_dir,
        ),
        batch_size=cfg.buffer.batch_size, # 模型训练一次的累计梯度和
        transform=MultiStepTransform(n_steps=cfg.loss.nstep, gamma=cfg.loss.gamma) if cfg.loss.nstep > 1 else None,
    )

    # Create the loss module
    if cfg.loss.use_value_rescale:
        loss_module = CustomDQNLoss(
            value_network=model,
            loss_function=cfg.loss.loss_type,
            delay_value=True, # 延迟更新
            double_dqn=True, # double dqn 要求延迟更新
        )
    else:
        loss_module = DQNLoss(
            value_network=model,
            loss_function=cfg.loss.loss_type,
            delay_value=True,
            double_dqn=True,
        )
    loss_module.make_value_estimator(
        gamma=cfg.loss.gamma,
    ) # 设定自定义的值放缩的值估计器
    loss_module = loss_module.to(device)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.loss.hard_update_freq
    ) # TODO: 搞清这个参数怎么调好，以及其具体的作用

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    logger = None
    if cfg.logger.backend:
        if cfg.logger.backend == 'wandb':
            logger = get_logger(
                cfg.logger.backend,
                logger_name=f'{base_dir}/ckpt', # 存放wandb位置
                experiment_name=ckpt_dir,
                wandb_kwargs={
                    "config": dict(cfg), # TODO: 添加奖励数据
                },
            )
        else:
            logger = get_logger(
                cfg.logger.backend,
                logger_name=f'{base_dir}/ckpt/{algo_name}',
                experiment_name=ckpt_dir,
            )

    # Main loop
    collected_frames = 0
    start_time = time.time()
    batch_size = cfg.buffer.batch_size
    test_interval = cfg.logger.test_interval # 每隔多久测试一次
    frames_per_batch = cfg.collector.frames_per_batch
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    init_random_frames = cfg.collector.init_random_frames
    num_updates = math.ceil(frames_per_batch / batch_size * cfg.loss.utd_ratio) # 每次采集对模型的更新次数
    q_losses = torch.zeros(num_updates, device=device)
    sampling_start = time.time()

    for i, data in enumerate(collector): # 每获得一批（frames_per_batch）数量的新数据，就记录和训练一轮次
        log_info = {}
        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())
        data = data.reshape(-1) # 所有数据全部展平成一维
        current_frames = data.numel()
        collected_frames += current_frames
        greedy_module.step(current_frames) # 更新greedy_module的epsilon值

        # 总结新获得数据的episode奖励和长度，Get and log training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]] # 取出有效的episode结束累计奖励
        if len(episode_rewards) > 0:
            episode_reward_mean = episode_rewards.mean().item()
            episode_length = data["next", "step_count"][data["next", "done"]]
            episode_length_mean = episode_length.sum().item() / len(episode_length)
            episode_weed_ratio = data["next", "weed_ratio"][data["next", "done"]]
            episode_weed_ratio_mean = episode_weed_ratio.sum().item() / len(episode_length)
            log_info.update(
                {
                    "train/episode_reward": episode_reward_mean,
                    "train/episode_length": episode_length_mean,
                    "train/episode_weed_ratio": episode_weed_ratio_mean,
                }
            )
        data.pop('weed_ratio') # 去除额外信息
        data.pop(('next', 'weed_ratio')) # 去除额外信息
        replay_buffer.extend(data)

        if collected_frames < init_random_frames: # 如果还没到初始frames，则log后继续收集, 开始不存在训练，速度就快
            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, step=collected_frames)
            continue

        # optimization steps TODO: 还可以完全搞清楚这里训练的运行逻辑
        training_start = time.time()
        for j in range(num_updates):
            sampled_tensordict = replay_buffer.sample(batch_size).to(device)
            loss_td = loss_module(sampled_tensordict)
            q_loss = loss_td["loss"]
            optimizer.zero_grad()
            q_loss.backward()
            if cfg.optim.max_grad_norm: # 目前没做梯度截断
                torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=cfg.optim.max_grad_norm
                )
            optimizer.step()
            target_net_updater.step()
            q_losses[j].copy_(q_loss.detach()) # 记录拆离计算图的q_loss值
            replay_buffer.update_tensordict_priority(sampled_tensordict)
        training_time = time.time() - training_start

        # Get and log q-values, loss, epsilon, sampling time and training time
        log_info.update(
            {
                "train/q_loss": q_losses.mean().item(),
                "train/epsilon": greedy_module.eps,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )
        if cfg.loss.use_value_rescale:
            log_info.update(
                {
                    "train/q_values": value_rescale_inv((data["action_value"] * data["action"]).sum()).item()
                                      / frames_per_batch,
                    "train/q_logits": (data["action_value"] * data["action"]).sum().item()
                                      / frames_per_batch,
                }
            )
        else:
            log_info.update(
                {
                    "train/q_values": (data["action_value"] * data["action"]).sum().item()
                                      / frames_per_batch,
                }
            )
        # Get and log evaluation rewards and eval time
        prev_test_frame = ((i - 1) * frames_per_batch) // test_interval
        cur_test_frame = (i * frames_per_batch) // test_interval
        final = collected_frames >= collector.total_frames
        if (i > 0 and (prev_test_frame < cur_test_frame)) or final:
            model_name = str(collected_frames // 1000).rjust(5, '0')
            torch.save(
                model,
                f'{base_dir}/ckpt/{algo_name}/{ckpt_dir}/t[{model_name}].pt'
            )

        # Log all the information
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=collected_frames)

        # update weights of the inference policy
        collector.update_policy_weights_()
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")
    time.sleep(5)


if __name__ == "__main__":
    cfg = yaml.load(open(f'{base_dir}/configs/train_{algo_name}_config.yaml'), Loader=yaml.FullLoader)
    cfg = DictConfig(cfg)
    main(cfg)
