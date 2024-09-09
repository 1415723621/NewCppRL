import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch.nn
import torch.optim
import tqdm
import yaml
from omegaconf import DictConfig
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
from torchrl.objectives import SoftUpdate, SACLoss
from torchrl.record.loggers import get_logger

from rl.sac_cont.sac_cont_utils import make_sac_models
from torchrl_utils import (
    make_env
)

base_dir = Path(__file__).parent.parent.parent
algo_name = 'sac_cont'


def main(cfg: "DictConfig"):  # noqa: F821
    ckpt_dir = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
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

    # Make the components
    if cfg.pretrained_model:
        actor_critic = torch.load(f'{base_dir}/{cfg.pretrained_model}')
        actor_critic = actor_critic.to(device)
    else:
        actor_critic = make_sac_models()
        actor_critic = actor_critic.to(device)
    # torch.save(
    #     actor_critic,
    #     f'{base_dir}/ckpt/{algo_name}/{ckpt_dir}/t[00000].pt'
    # )
    actor = actor_critic[0]
    q_critic = actor_critic[1]

    # Create the collector
    collector = MultiaSyncDataCollector(
        create_env_fn=[lambda: make_env(
            num_envs=1,
            device='cpu',
        )] * cfg.collector.num_envs,
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device='cpu',
        storing_device='cpu',
        max_frames_per_traj=-1,
    )

    # Create the replay buffer
    tempdir = tempfile.TemporaryDirectory()
    scratch_dir = tempdir.name
    replay_buffer = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.5,
        pin_memory=False,
        prefetch=10,
        storage=LazyMemmapStorage(
            max_size=cfg.buffer.buffer_size,
            scratch_dir=scratch_dir,
        ),
        batch_size=cfg.buffer.batch_size,
    )
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=q_critic,
        num_qvalue_nets=2,
        loss_function=cfg.loss.loss_function,
        delay_actor=False,
        delay_qvalue=True,
    )
    loss_module.make_value_estimator(gamma=cfg.loss.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.loss.target_update_polyak)

    # Create the optimizer
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    optimizer_actor = torch.optim.AdamW(
        actor_params,
        lr=cfg.optim.lr_actor,
        weight_decay=cfg.optim.weight_decay_actor,
    )
    optimizer_critic = torch.optim.AdamW(
        critic_params,
        lr=cfg.optim.lr_critic,
        weight_decay=cfg.optim.weight_decay_critic,
    )
    optimizer_alpha = torch.optim.AdamW(
        [loss_module.log_alpha],
        lr=cfg.optim.lr_alpha,
        weight_decay=cfg.optim.weight_decay_alpha,
    )

    # Create the logger
    logger = None
    if cfg.logger.backend:
        if cfg.logger.backend == 'wandb':
            logger = get_logger(
                cfg.logger.backend,
                logger_name=f'{base_dir}/ckpt',
                experiment_name=ckpt_dir,
                wandb_kwargs={
                    "config": dict(cfg),
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

    # cfg.loss.clip_epsilon = cfg_loss_clip_epsilon
    init_random_frames = cfg.collector.init_random_frames
    batch_size = cfg.buffer.batch_size
    frames_per_batch = cfg.collector.frames_per_batch
    num_updates = math.ceil(frames_per_batch / batch_size * cfg.loss.utd_ratio)
    test_interval = cfg.logger.test_interval
    actor_losses = torch.zeros(num_updates, device=device)
    q_losses = torch.zeros(num_updates, device=device)
    alpha_losses = torch.zeros(num_updates, device=device)
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    sampling_start = time.time()

    for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel()
        collected_frames += current_frames

        # Get training rewards and episode lengths
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
        data.pop('weed_ratio')  # 去除额外信息
        data.pop(('next', 'weed_ratio'))  # 去除额外信息
        replay_buffer.extend(data)

        if collected_frames < init_random_frames:
            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, step=collected_frames)
            continue

        training_start = time.time()
        for j in range(num_updates):
            # Sample from replay buffer
            sampled_tensordict = replay_buffer.sample()
            if sampled_tensordict.device != device:
                sampled_tensordict = sampled_tensordict.to(
                    device, non_blocking=True
                )
            else:
                sampled_tensordict = sampled_tensordict.clone()

            # Compute loss
            loss_out = loss_module(sampled_tensordict)

            actor_loss, q_loss, alpha_loss = (
                loss_out["loss_actor"],
                loss_out["loss_qvalue"],
                loss_out["loss_alpha"],
            )

            # Update actor
            optimizer_actor.zero_grad()
            actor_loss.backward()
            if cfg.optim.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=cfg.optim.max_grad_norm
                )
            optimizer_actor.step()

            # Update critic
            optimizer_critic.zero_grad()
            q_loss.backward()
            optimizer_critic.step()
            q_losses[j].copy_(q_loss.detach())

            actor_losses[j].copy_(actor_loss.detach())

            # Update alpha
            optimizer_alpha.zero_grad()
            alpha_loss.backward()
            optimizer_alpha.step()

            alpha_losses[j].copy_(alpha_loss.detach())

            # Update target params
            target_net_updater.step()

            # Update priority
            # if prb:
            replay_buffer.update_tensordict_priority(sampled_tensordict)

        # Get training losses and times
        training_time = time.time() - training_start
        log_info.update(
            {
                "train/q_loss": q_losses.mean().item(),
                "train/a_loss": actor_losses.mean().item(),
                "train/alpha_loss": alpha_losses.mean().item(),
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )

        # Get and log evaluation rewards and eval time
        prev_test_frame = ((i - 1) * frames_per_batch) // test_interval
        cur_test_frame = (i * frames_per_batch) // test_interval
        final = collected_frames >= collector.total_frames
        if (i > 0 and (prev_test_frame < cur_test_frame)) or final:
            model_name = str(collected_frames // 1000).rjust(5, '0')
            torch.save(
                actor_critic,
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
