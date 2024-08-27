from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.wrappers import HumanRendering
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type

import envs  # noqa

base_dir = Path(__file__).parent.parent.parent
cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
episodes = 10
render = True

device = 'cpu'
# pt_path = f'../ckpt/train/2024-08-24_03-56-49_CnnElu/t[00400]_r[1650.20].pt'
# pt_path = f'../ckpt/train/2024-08-24_03-56-49_CnnElu/t[01650]_r[1215.28].pt'
pt_path = f'../../ckpt/t[01600]_r[-80.25].pt'
model = torch.load(pt_path).to(device)
actor = model[0]

# cfg.env.params.num_obstacles_range = [0, 0]

env = gym.make(
    render_mode='rgb_array' if render else None,
    **cfg.env.params,
)
if render:
    env = HumanRendering(env)
# reset_options = {
#     'obstacle'
# }

costs = []

with set_exploration_type(ExplorationType.MODE), torch.no_grad():
    i = 0
    failed_count = 0
    while i < episodes:
        obs, info = env.reset()
        done = False
        ret = 0.
        max_r = -100.
        t = 0
        while not done:
            if isinstance(obs, dict):
                observation = obs['observation']
                vector = obs['vector']
            observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
            vector = torch.tensor([vector]).float().to(device).unsqueeze(0)
            # print(obs)
            # print(observation[0, 5:8])
            # Get Output
            action = actor(observation=observation, vector=vector)
            # print(action[0])
            action = action[0].argmax()
            action = int(action)
            # print(action)
            obs, reward, done, _, info = env.step(action)
            max_r = max(max_r, reward)
            t += 1
            ret += 0.99 ** t * reward
            print(f'{t:04d} | {reward:.3f}, {ret:.3f}')
            if render:
                env.render()
        print(f'Max r: {max_r}')
env.close()
costs = np.array(costs)
print(f'{costs.mean()} +- {costs.std()}')
print(f'{failed_count} / {i + failed_count} = {failed_count / (i + failed_count)}')
