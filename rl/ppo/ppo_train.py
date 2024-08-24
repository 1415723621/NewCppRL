import math
import os
import tempfile
import time
from pathlib import Path

import torch.nn
import torch.optim
import tqdm
import yaml
from omegaconf import DictConfig
from tensordict.nn import TensorDictSequential
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyMemmapStorage
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record.loggers import get_logger

import envs  # noqa
from envs.cpp_env_v2 import CppEnvironment
from rl.ppo.ppo_utils import make_ppo_models
from torchrl_utils import (
    CustomVideoRecorder,
    CustomDQNLoss,
    value_rescale_inv,
    make_env,
    eval_model
)
import time

import torch.optim
import tqdm

from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE

base_dir = Path(__file__).parent.parent.parent
nvec = CppEnvironment.nvec
algo_name = 'ppo'

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

    # Make the components
    if cfg.pretrained_model:
        actor_critic = torch.load(f'{base_dir}/{algo_name}/{cfg.pretrained_model}')
        actor = actor_critic.get_policy_operator().to(device)
        critic = actor_critic.get_value_operator().to(device)
    else:
        actor_critic = make_ppo_models()
        actor = actor_critic.get_policy_operator().to(device)
        critic = actor_critic.get_value_operator().to(device)

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
        reset_at_each_iter=True,
    )

    # Create the replay buffer
    tempdir = tempfile.TemporaryDirectory()
    scratch_dir = tempdir.name
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(
            max_size=cfg.collector.frames_per_batch,
            scratch_dir=scratch_dir,
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.loss.mini_batch_size,
    )
    # replay_buffer = TensorDictPrioritizedReplayBuffer(
    #     alpha=0.7,
    #     beta=0.5,
    #     pin_memory=False,
    #     prefetch=10,
    #     storage=LazyMemmapStorage(
    #         max_size=cfg.buffer.buffer_size,
    #         scratch_dir=scratch_dir,
    #     ),
    #     batch_size=cfg.loss.mini_batch_size,
    # )

    # Create the loss module
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
    )

    # Create the optimizer
    optimizer = torch.optim.AdamW(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    logger = None
    if cfg.logger.backend:
        if cfg.logger.backend == 'wandb':
            logger = get_logger(
                cfg.logger.backend,
                logger_name=f'{base_dir}/ckpt/{algo_name}',
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

    # Create the test environment
    test_env = make_env(
        num_envs=1,
        # device=device,
        device='cpu',
        from_pixels=cfg.logger.video,
    )
    if cfg.logger.video:
        skip_frames = 20
        test_env.insert_transform(
            0,
            CustomVideoRecorder(
                logger,
                tag=f"eval/video",
                in_keys=["pixels"],
                skip=skip_frames,
                make_grid=False,
                nrow=2,
                max_len=cfg.logger.test_steps // skip_frames,
            ),
        )
    test_env.eval()

    # Main loop
    collected_frames = 0
    start_time = time.time()
    num_mini_batches = math.ceil(cfg.collector.frames_per_batch / cfg.loss.mini_batch_size)
    num_network_updates = 0
    total_network_updates = (
        cfg.collector.total_frames * cfg.loss.ppo_epochs * num_mini_batches
    )

    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_optim_max_grad_norm = cfg.optim.max_grad_norm
    # cfg.loss.clip_epsilon = cfg_loss_clip_epsilon
    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, cfg.loss.mini_batch_size])

    test_interval = cfg.logger.test_interval
    frames_per_batch = cfg.collector.frames_per_batch
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    sampling_start = time.time()

    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Get training rewards and episode lengths
        if data["next", "done"].any():
            episode_rewards = data["next", "episode_reward"][data["next", "done"]].mean()
            episode_length = data["next", "step_count"][data["next", "done"]]
        else:
            episode_rewards = data["next", "episode_reward"][-1].mean()
            episode_length = data["next", "step_count"][-1]
        # episode_rewards = data["next", "episode_reward"][data["next", "terminated"]]
        # if len(episode_rewards) > 0:
        #     episode_length = data["next", "step_count"][data["next", "terminated"]]
        log_info.update(
            {
                "train/reward": episode_rewards.mean().item(),
                "train/episode_length": episode_length.sum().item()
                / len(episode_length),
            }
        )

        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):

            # Re-Compute GAE
            with torch.no_grad():
                data = adv_module(data.to(device, non_blocking=True))
            data_reshape = data.reshape(-1)
            # Update the data buffer
            replay_buffer.extend(data_reshape)

            for k, batch in enumerate(replay_buffer):

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in optimizer.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                num_network_updates += 1
                if cfg_loss_anneal_clip_eps:
                    loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                # Get a data batch
                batch = batch.to(device, non_blocking=True)

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )
                # Backward pass
                loss_sum.backward()
                if cfg_optim_max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(loss_module.parameters()), max_norm=cfg_optim_max_grad_norm
                    )

                # Update the networks
                optimizer.step()
                optimizer.zero_grad()

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": alpha * cfg_optim_lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": alpha * cfg_loss_clip_epsilon,
            }
        )

        # Get and log evaluation rewards and eval time
        prev_test_frame = ((i - 1) * frames_per_batch) // test_interval
        cur_test_frame = (i * frames_per_batch) // test_interval
        final = collected_frames >= collector.total_frames
        if (i > 0 and (prev_test_frame < cur_test_frame)) or final:
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                actor.eval()
                eval_start = time.time()
                td_test = eval_model(actor, test_env, cfg.logger.test_steps)
                if td_test["next", "done"].any():
                    test_rewards = td_test["next", "episode_reward"][td_test["next", "done"]].mean()
                else:
                    test_rewards = td_test["next", "episode_reward"][-1].mean()
                eval_time = time.time() - eval_start
                actor.train()
                log_info.update(
                    {
                        "eval/reward": test_rewards,
                        "eval/eval_time": eval_time,
                    }
                )
                model_name = str(collected_frames // 1000).rjust(5, '0')
                torch.save(
                    actor_critic,
                    f'{base_dir}/ckpt/{algo_name}/{ckpt_dir}/t[{model_name}]_r[{test_rewards:.3f}].pt'
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
    cfg = yaml.load(open(f'{base_dir}/configs/{algo_name}_train_config.yaml'), Loader=yaml.FullLoader)
    cfg = DictConfig(cfg)
    main(cfg)
