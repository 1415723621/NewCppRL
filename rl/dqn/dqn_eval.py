import collections
import os
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import get_logger, Logger

import envs  # noqa
from torchrl_utils.local_video_recorder import LocalVideoRecorder

algo_name = 'dqn'
base_dir = Path(__file__).parent.parent.parent
env_cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
episodes = 4
max_frames = 4
max_step = 1000
skip_frames = 20
video = True

device = 'cpu'
ckpt_root = f'../../ckpt/{algo_name}'
ckpt_name = sorted(os.listdir(ckpt_root))[-1]
ckpt_path = f'../../ckpt/{algo_name}/{ckpt_name}'
# ckpt_path = f'../../ckpt/dqn/2024-08-27_18-56-37_LessRewards'
start_idx = 0
ckpt_dir = ckpt_path.split('/')[-1]


def eval_actor(envs: list[gym.Env],
               actor: torch.nn.Module,
               logger: Logger,
               collected_frames: int,
               recorder: Optional[LocalVideoRecorder]):
    eval_start = time.time()
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        obss = []
        for env in envs:
            obs, _ = env.reset()
            obss.append(obs)
        # done = False
        rets = [0.] * episodes
        dones = [False] * episodes
        # Render
        if video:
            pixels = []
            for idx, env in enumerate(envs):
                if idx >= max_frames:
                    break
                pixels.append(env.render_map())
            recorder.apply(torch.from_numpy(np.stack(pixels, 0)))
        pbar = tqdm.tqdm(total=max_step)
        for t in range(max_step):
            pbar.update(1)
            if (t + 1) % skip_frames == 0:
                pbar.set_postfix({"reward": np.mean(rets),
                                  "reward_min": np.min(rets),
                                  "reward_max": np.max(rets),
                                  "reward_std": (np.max(rets) - np.min(rets)),
                                  'agents_alive': f'{dones.count(False)} / {episodes}'})
            observation = []
            vector = []
            for obs in obss:
                if isinstance(obs, dict):
                    observation.append(obs['observation'])
                    vector.append([obs['vector']])
            observation = torch.from_numpy(np.stack(observation, axis=0)).float().to(device)
            vector = torch.tensor(numpy.array(vector)).float().to(device)
            actions = actor(observation=observation, vector=vector).argmax(dim=-1).tolist()
            act_idx = 0
            obss = []
            for idx, env in enumerate(envs):
                if not dones[idx]:
                    obs, reward, done, _, _ = env.step(actions[act_idx])
                    obss.append(obs)
                    rets[idx] += reward
                    dones[idx] |= done
                    act_idx += 1
            # Render
            if video and (t + 1) % skip_frames == 0:
                for idx, env in enumerate(envs):
                    if idx >= max_frames:
                        break
                    if not dones[idx]:
                        pixels[idx] = env.render_map()
                recorder.apply(torch.from_numpy(np.stack(pixels, 0)))
            if all(dones):
                break
    eval_time = time.time() - eval_start
    rewards_mean = np.mean(rets)
    rewards_min = np.min(rets)
    rewards_max = np.max(rets)
    rewards_std = rewards_max - rewards_min
    log_info = {
        "eval/reward": rewards_mean,
        "eval/reward_min": rewards_min,
        "eval/reward_max": rewards_max,
        "eval/reward_std": rewards_std,
        "eval/eval_time": eval_time,
    }
    for key, value in log_info.items():
        logger.log_scalar(key, value, step=collected_frames)
    if video:
        video_tensor = recorder.dump()
        logger.log_video('eval/video', video_tensor, step=collected_frames)
    print(f'\tEvaluation finished, cost {eval_time:.2f} seconds. \n'
          f'\tReward = {rewards_mean:.2f} ± {rewards_std:.2f}, {rewards_min} ~ {rewards_max}')
    return rewards_mean, rewards_min, rewards_max


if __name__ == '__main__':
    print(f'Dir: {ckpt_path}')
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
    envs = []
    for _ in range(episodes):
        envs.append(gym.make(
            render_mode=None,  # if render else None,
            **env_cfg.env.params,
        ))
    # if render:
    #     env = HumanRendering(env)
    recorder = None
    if video:
        recorder = LocalVideoRecorder(
            max_len=(max_step * episodes) // skip_frames + 2,
            skip=1,
            use_memmap=True,
            make_grid=True,
            nrow=2,
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
                rewards_mean, rewards_min, rewards_max = eval_actor(envs, actor, logger, collected_frames, recorder)
                model_name = str(collected_frames // 1000).rjust(5, '0')
                os.rename(
                    pt_path,
                    f'{ckpt_path}/t[{model_name}]_r[{rewards_mean:.2f}={rewards_min:.2f}~{rewards_max:.2f}].pt'
                )
                print('Continue watching.')
        last_num = cur_num
        time.sleep(10)
