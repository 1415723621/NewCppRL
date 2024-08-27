import collections
import os
from pathlib import Path
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import get_logger, Logger
import tqdm

import envs  # noqa
from torchrl_utils.local_video_recorder import LocalVideoRecorder

algo_name = 'dqn'
base_dir = Path(__file__).parent.parent.parent
env_cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
episodes = 5
max_step = 800
video = True

device = 'cpu'
ckpt_path = f'../../ckpt/dqn/2024-08-27_18-56-37_LessRewards'
start_idx = 0
ckpt_dir = ckpt_path.split('/')[-1]


def eval_actor(env: gym.Env,
               actor: torch.nn.Module,
               logger: Logger,
               collected_frames: int,
               recorder: Optional[LocalVideoRecorder]):
    rewards = []
    eval_start = time.time()
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        for i in tqdm.trange(episodes):
            obs, info = env.reset()
            done = False
            t = 0
            ret = 0.
            # Render
            if video:
                pixels = env.render()
                recorder.apply(pixels)
            while not done:
                if isinstance(obs, dict):
                    observation = obs['observation']
                    vector = obs['vector']
                observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
                vector = torch.tensor([vector]).float().to(device).unsqueeze(0)
                action = actor(observation=observation, vector=vector)[0].argmax().item()
                obs, reward, done, _, info = env.step(action)
                ret += reward
                t += 1
                if t >= max_step:
                    done = True
                # Render
                if video:
                    pixels = env.render()
                    recorder.apply(pixels)
            rewards.append(ret)
            # print(f'\tEpisode ({i + 1} / {episodes}) | Cost {time.time() - ep_start:.2f} seconds.')
    eval_time = time.time() - eval_start
    rewards_mean = np.mean(rewards)
    rewards_std = np.std(rewards)
    log_info = {
        "eval/reward": rewards_mean,
        "eval/reward_std": rewards_std,
        "eval/eval_time": eval_time,
    }
    for key, value in log_info.items():
        logger.log_scalar(key, value, step=collected_frames)
    if video:
        video_tensor = recorder.dump()
        logger.log_video('eval/video', video_tensor, step=collected_frames)
    print(f'\tEvaluation finished, cost {eval_time:.2f} seconds. \n\tReward = {rewards_mean:.2f} ± {rewards_std:.2f}')
    return rewards_mean, rewards_std


if __name__ == '__main__':
    train_cfg = yaml.load(open(f'{base_dir}/configs/train_{algo_name}_config.yaml'), Loader=yaml.FullLoader)
    train_cfg = DictConfig(train_cfg)
    logger = None
    if train_cfg.logger.backend == 'wandb':
        logger = get_logger(
            train_cfg.logger.backend,
            logger_name=f'{base_dir}/ckpt',
            experiment_name=ckpt_dir,
            wandb_kwargs={
                "config": dict(train_cfg),
            },
        )
    else:
        logger = get_logger(
            train_cfg.logger.backend,
            logger_name=f'{base_dir}/ckpt/{algo_name}',
            experiment_name=ckpt_dir,
        )
    env = gym.make(
        render_mode='rgb_array',  # if render else None,
        **env_cfg.env.params,
    )
    # if render:
    #     env = HumanRendering(env)
    recorder = None
    if video:
        skip_frames = 20
        recorder = LocalVideoRecorder(
            max_len=(max_step * episodes) // skip_frames + 2,
            skip=skip_frames,
            use_memmap=True,
            fps=6,
        )

    model_list = sorted(os.listdir(ckpt_path))
    last_num = len(model_list)
    for model_id in model_list:
        if model_id.split('.')[-1] == 'pt':
            last_num -= 1
    last_num += start_idx
    model_pool = collections.deque()
    print(model_list)
    print('Start watching.')
    while True:
        model_list = sorted(os.listdir(ckpt_path))
        cur_num = len(model_list)
        # print(model_list)
        if cur_num > last_num:
            print(f'{cur_num - last_num} new models detected.')
            model_pool += model_list[last_num:(cur_num + 1)]
            while len(model_pool) > 0:
                print(f'{len(model_pool)} left in Model Pool.')
                model_id = model_pool.popleft()
                if len(model_id) > 11:
                    continue
                print(f'Processing model {model_id}.')
                collected_frames = int(model_id[2:7]) * 1000
                pt_path = f'{ckpt_path}/{model_id}'
                print(f'Collected {collected_frames}, evaluating...')
                model = torch.load(pt_path).to(device)
                actor = model[0]
                rewards_mean, rewards_std = eval_actor(env, actor, logger, collected_frames, recorder)
                model_name = str(collected_frames // 1000).rjust(5, '0')
                os.rename(pt_path, f'{ckpt_path}/t[{model_name}]_r[{rewards_mean:.2f}±{rewards_std:.2f}].pt')
                print('Continue watching.')
        last_num = cur_num
        time.sleep(10)
