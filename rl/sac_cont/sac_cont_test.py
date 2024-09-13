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
act_randomly = True
# act_randomly = False

device = 'cpu'
pt_path = f'../../ckpt/sac_cont/2024-09-08_22-43-52_wd1e-4_polyak9997/t[03750]_r[2758.92=2736.49~2813.41].pt'
actor_critic = torch.load(pt_path).to(device)
actor = actor_critic[0].to(device)

# cfg.env.params.num_obstacles_range = [0, 0]
env = gym.make(
    render_mode='rgb_array' if render else None,
    **cfg.env.params,
)
if render:
    env = HumanRendering(env)

exploration_type = ExplorationType.RANDOM if act_randomly else ExplorationType.DETERMINISTIC

with set_exploration_type(exploration_type), torch.no_grad():
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
            action = logits[2][0].tolist()
            print(action)
            obs, reward, done, _, info = env.step(action)
            t += 1
            ret += reward
            print(f'{t:04d} | {reward:.3f}, {ret:.3f}')
            if render:
                env.render()
env.close()
