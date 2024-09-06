import collections
import os
import time
from pathlib import Path
from typing import Optional, Any

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


class CustomEvaluator:
    algo_name = 'dqn'
    base_dir = Path(__file__).parent.parent
    env_cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))

    def __init__(self,
                 episodes=16,
                 max_frames=4,
                 max_step=1500,
                 skip_frames=30,
                 video=True,
                 device='cpu',
                 start_idx=0,
                 ckpt_path: Optional[str] = None,
                 ):
        self.episodes = episodes
        self.max_frames = max_frames
        self.max_step = max_step
        self.skip_frames = skip_frames
        self.video = video
        self.device = device
        self.start_idx = start_idx
        if ckpt_path is None:
            ckpt_root = f'{self.base_dir}/ckpt/{self.algo_name}'
            ckpt_name = sorted(os.listdir(ckpt_root))[-1]
            self.ckpt_path = f'{self.base_dir}/ckpt/{self.algo_name}/{ckpt_name}'
        self.ckpt_dir = self.ckpt_path.split('/')[-1]

    def get_actions(self,
                    actor: torch.nn.Module,
                    obss: list[Any]) -> list[int]:
        observation = []
        vector = []
        for obs in obss:
            if isinstance(obs, dict):
                observation.append(obs['observation'])
                vector.append([obs['vector']])
        observation = torch.from_numpy(np.stack(observation, axis=0)).float().to(self.device)
        vector = torch.tensor(numpy.array(vector)).float().to(self.device)
        actions = actor(observation=observation, vector=vector).argmax(dim=-1).tolist()
        return actions

    def get_actor(self,
                  pt_path: str) -> torch.nn.Module:
        model = torch.load(pt_path).to(self.device)
        actor = model[0]
        return actor

    def eval_actor(self,
                   envs: list[gym.Env],
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
            rets = [0.] * self.episodes
            weed_ratios = [0.] * self.episodes
            dones = [False] * self.episodes
            # Render
            if self.video:
                pixels = []
                for idx, env in enumerate(envs):
                    if idx >= self.max_frames:  # 只取前max_frames个场景绘制视频记录
                        break
                    pixels.append(env.render_map())
                recorder.apply(torch.from_numpy(np.stack(pixels, 0)))
            pbar = tqdm.tqdm(total=self.max_step)
            for t in range(self.max_step):
                pbar.update(1)
                if (t + 1) % self.skip_frames == 0:  # 每隔skip_frames帧存一下奖励数据
                    rewards_mean = np.mean(rets)
                    rewards_min = np.min(rets)
                    rewards_max = np.max(rets)
                    rewards_std = rewards_max - rewards_min
                    pbar.set_postfix(
                        {"reward": f'{rewards_mean:.2f} ± {rewards_std:.2f}, {rewards_min} ~ {rewards_max}',
                         "weed_ratio": np.mean(weed_ratios),
                         'agents_alive': f'{dones.count(False)} / {self.episodes}'})  # 这里只是展示在tqdm，不是直接log
                actions = self.get_actions(actor, obss)
                act_idx = 0
                obss = []
                for idx, env in enumerate(envs):
                    if not dones[idx]:
                        obs, reward, done, _, _ = env.step(actions[act_idx])
                        obss.append(obs)
                        rets[idx] += reward
                        dones[idx] |= done
                        weed_ratios[idx] = obs["weed_ratio"]
                        act_idx += 1
                # Render
                if self.video and (t + 1) % self.skip_frames == 0:
                    for idx, env in enumerate(envs):
                        if idx >= self.max_frames:
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
            "eval/weed_ratio": np.mean(weed_ratios),
            "eval/eval_time": eval_time,
        }
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=collected_frames * 2)
            if self.video:
                video_tensor = recorder.dump()
                logger.log_video('eval/video', video_tensor, step=collected_frames * 2)  # TODO: 以后搞清楚为什么乘以2
        print(f'\tEvaluation finished, cost {eval_time:.2f} seconds. \n'
              f'\tReward = {rewards_mean:.2f} ± {rewards_std:.2f}, {rewards_min} ~ {rewards_max}')
        return rewards_mean, rewards_min, rewards_max

    def run(self):
        print(f'Dir: {self.ckpt_path}')
        train_cfg = yaml.load(open(f'{self.base_dir}/configs/train_{self.algo_name}_config.yaml'), Loader=yaml.FullLoader)
        train_cfg = DictConfig(train_cfg)
        logger = None
        if train_cfg.logger.backend:
            if train_cfg.logger.backend == 'wandb':
                logger = get_logger(
                    train_cfg.logger.backend,
                    logger_name=f'{self.base_dir}/ckpt',
                    experiment_name=self.ckpt_dir,
                    wandb_kwargs={
                        "config": dict(train_cfg),
                    },
                )
            else:
                logger = get_logger(
                    train_cfg.logger.backend,
                    logger_name=f'{self.base_dir}/ckpt/{self.algo_name}',
                    experiment_name=self.ckpt_dir,
                )
        envs = []
        for _ in range(self.episodes):
            envs.append(gym.make(
                render_mode=None,  # if render else None,
                **self.env_cfg.env.params,
            ))
        recorder = None
        if self.video:
            recorder = LocalVideoRecorder(
                max_len=(self.max_step * self.episodes) // self.skip_frames + 2,
                skip=1,
                use_memmap=True,
                make_grid=True,
                nrow=2,
                fps=6,
            )
        model_list = sorted(os.listdir(self.ckpt_path))
        last_num = len(model_list)
        for model_id in model_list:
            if model_id.split('.')[-1] == 'pt':
                last_num -= 1
        last_num += self.start_idx
        model_pool = collections.deque()
        print(model_list)
        print('Start watching.')
        collected_frames = 0
        while True:
            model_list = sorted(os.listdir(self.ckpt_path))
            cur_num = len(model_list)
            # print(model_list)
            if cur_num > last_num:  # 每次检测有无新模型进入
                print(f'{cur_num - last_num} new models detected.')
                model_pool += model_list[last_num:(cur_num + 1)]
                while len(model_pool) > 0:
                    print(f'{len(model_pool)} left in Model Pool.')
                    model_id = model_pool.popleft()
                    time.sleep(1)
                    if len(model_id) > 11:
                        continue
                    print(f'Processing model {model_id}.')
                    collected_frames = int(model_id[2:7]) * 1000
                    pt_path = f'{self.ckpt_path}/{model_id}'
                    print(f'Collected {collected_frames}, evaluating...')
                    actor = self.get_actor(pt_path)
                    rewards_mean, rewards_min, rewards_max = self.eval_actor(envs, actor, logger, collected_frames, recorder)
                    model_name = str(collected_frames // 1000).rjust(5, '0')  # TODO: 乘以1000除以1000不是很懂,以后搞懂这个细节
                    os.rename(
                        pt_path,
                        f'{self.ckpt_path}/t[{model_name}]_r[{rewards_mean:.2f}={rewards_min:.2f}~{rewards_max:.2f}].pt'
                    )
                if collected_frames >= train_cfg.collector.total_frames:  # 终止条件是用当前帧数判断一下是否到达配置文件最大帧数，如果到达则说明验证结束
                    print('Finished.')
                    break
                print('Continue watching.')
            last_num = cur_num
            time.sleep(10)


if __name__ == '__main__':
    evaluator = CustomEvaluator(
        episodes=16,
        max_frames=4,
        max_step=1500,
        skip_frames=30,
        video=True,
        device='cpu',
        start_idx=0,
        ckpt_path=None,
        # ckpt_path='../../ckpt/dqn/240901_051858_test',
    )
    evaluator.run()
