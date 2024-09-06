from pathlib import Path

import gymnasium as gym
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
pt_path = f'../../ckpt/t[01950]_r[517.41=434.62~622.00].pt'
model = torch.load(pt_path).to(device)
actor = model[0]

# cfg.env.params.num_obstacles_range = [0, 0]

env = gym.make(
    render_mode='rgb_array' if render else None,
    **cfg.env.params,
    # state_pixels=True,
    state_pixels=False,
)
if render:
    env = HumanRendering(env)

with set_exploration_type(ExplorationType.MODE), torch.no_grad():
    for i in range(episodes):
        obs, info = env.reset()
        done = False
        ret = 0.
        t = 0
        while not done:
            if isinstance(obs, dict):
                observation = obs['observation']
                vector = obs['vector']
            observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
            vector = torch.tensor([vector]).float().to(device).unsqueeze(0)
            # Get Output
            logits = actor(observation=observation, vector=vector)
            action = logits[0].argmax().item()
            obs, reward, done, _, info = env.step(action)
            t += 1
            ret += reward
            print(f'{t:04d} | {reward:.3f}, {ret:.3f}')
            if render:
                env.render()
env.close()
