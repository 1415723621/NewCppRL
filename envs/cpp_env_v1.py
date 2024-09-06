from __future__ import annotations

import os
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import HumanRendering

from envs.cpp_env_base import CppEnvBase
from envs.utils import MowerAgent, total_variation


class CppEnv(CppEnvBase):
    """
    There's no mist in this env.
    """

    def get_maps_and_mask(self) -> tuple[np.ndarray, list[float]]:
        maps = np.stack((
            self.map_frontier,
            self.map_obstacle,
            np.logical_and(self.map_weed, np.logical_not(self.map_frontier)),
            self.map_trajectory,
        ), axis=-1)
        mask = [0., 0., 1., 0.]
        return maps, mask

    def get_reward(self,  # 这个y_tp1不是很懂啥意思
                   steer_tp1: float,
                   x_t: int,
                   y_t: int,
                   x_tp1: int,
                   y_tp1: int) -> float:
        weed_num_tp1 = self.map_weed.sum(dtype=np.int32)
        frontier_area_tp1 = self.map_frontier.sum(dtype=np.int32)
        frontier_tv_tp1 = total_variation(self.map_frontier.astype(np.int32))
        # Const
        reward_const = -0.1
        # Turning
        reward_turn_gap = -0.5 * abs(steer_tp1 - self.steer_t) / self.w_range.max
        reward_turn_direction = -0.30 * (0. if (steer_tp1 * self.steer_t >= 0
                                                or (steer_tp1 == 0 and self.steer_t == 0))
                                         else 1.)
        reward_turn_self = 0.25 * (0.4 - abs(steer_tp1 / self.w_range.max) ** 0.5)
        reward_turn = 0.0 * (reward_turn_gap
                             + reward_turn_direction
                             + reward_turn_self
                             )
        # Frontier
        reward_frontier_coverage = (self.frontier_area_t - frontier_area_tp1) / (
                2 * MowerAgent.width * self.v_range.max)
        reward_frontier_tv = 0.5 * (self.frontier_tv_t - frontier_tv_tp1) / self.v_range.max
        reward_frontier = 0.125 * (reward_frontier_coverage
                                   + reward_frontier_tv
                                   )
        # Weed
        reward_weed = 20.0 * (self.weed_num_t - weed_num_tp1)
        # Summary
        reward = (reward_const
                  + reward_frontier
                  + reward_weed
                  + reward_turn
                  )
        reward = np.where(
            np.abs(reward) < 1e-8,
            0.,
            reward,
        )
        self.weed_num_t = weed_num_tp1
        self.frontier_area_t = frontier_area_tp1
        self.frontier_tv_t = frontier_tv_tp1
        self.steer_t = steer_tp1
        return reward


if __name__ == "__main__":
    if_render = True
    episodes = 3
    env = CppEnv(
        render_mode='rgb_array' if if_render else None,
        # state_pixels=True,
        state_pixels=False,
    )
    env: CppEnv = HumanRendering(env)

    for _ in range(episodes):
        obs, info = env.reset(seed=120, options={
            'weed_dist': 'gaussian',
            # 'map_id': 80,
            "weed_num": 100
        })
        done = False
        while not done:
            action = env.action_space.sample()
            # action = 1 * 21 + 10
            obs, reward, done, _, info = env.step(action)
            # obs, reward, done, _, info = env.step((0, 4))
            print(reward)
            if if_render:
                env.render()

    env.close()
