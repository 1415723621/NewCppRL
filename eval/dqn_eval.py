import os

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

import envs  # noqa
import gymnasium as gym
from gymnasium.wrappers import HumanRendering

from eval.get_model import get_dqn

device = 'cpu'
# pt_path = f'./ckpt/2024-04-07_16-54-22/t[03990]_r[5222].pt'

# pt_path = f'./ckpt/2024-06-06_13-39-21/t[01140]_r[-1614].pt'
pt_path = './t[01000]_sd.pt'
model = get_dqn(pt_path)
actor = model[0]

env = gym.make(
    'Pasture',
    render_mode='rgb_array',
    # action_type="discrete",
    # prevent_stiff=True,
    # sgcnn=True,
    # global_obs=True,
    # use_apf=True,
    # diff_traj=True,
    # weed_ratio=0.002,
)
env = HumanRendering(env)
state, _ = env.reset()
step = 0
env.render()

with set_exploration_type(ExplorationType.MODE), torch.no_grad():
# with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
    while True:
        # Render Img
        observation = torch.from_numpy(state).to(device).unsqueeze(0)
        # Get Output
        action = actor(observation=observation)
        action = action[0].argmax()
        action = int(action)
        # print(action)
        # print(output[0][0][action - 2])
        # print(output[0][0][action - 1])
        # print(output[0][0][action])
        # print(output[0][0][action + 1])
        # print(output[0][0][action + 2])
        # action = torch.tanh(action)
        # print(action)
        state, reward, done, truncated, info = env.step(action)
        print(f't {step} / 1000: {reward}')
        step += 1
        env.render()
        if done:
            state, _ = env.reset()
            env.render()
            step = 0
