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

    def __init__(
            self,
            action_type: str = 'discrete',
            render_mode: str = None,
            state_pixels: bool = False,
            state_size: tuple[int, int] = (128, 128),
            state_downsize: tuple[int, int] = (128, 128),
            num_obstacles_range: tuple[int, int] = (5, 8),
            use_sgcnn: bool = True,
            use_global_obs: bool = True,
            map_dir: str = 'envs/maps/1-400',
    ):
        super().__init__()
        # Read maps
        self.map_dir = Path(__file__).parent.parent / map_dir
        self.map_names = sorted(os.listdir(self.map_dir))
        # Environmental parameters
        self.action_type = action_type
        self.dimensions = cv2.imread(str(self.map_dir / self.map_names[0])).shape[:-1]
        self.state_size = state_size
        self.state_downsize = state_downsize
        self.num_obstacles_range = num_obstacles_range
        self.use_sgcnn = use_sgcnn
        self.use_global_obs = use_global_obs
        obs_shape = (4, *self.state_downsize,)
        if use_sgcnn:
            obs_shape = (16 + 4 * use_global_obs, *(ds // 8 for ds in self.state_downsize))

        # RL parameters
        self.observation_space = gym.spaces.Box(
            low=0., high=1., shape=obs_shape, dtype=np.float32
        )
        if self.action_type == 'discrete':
            self.action_space = gym.spaces.Discrete(n=self.nvec[0] * self.nvec[1])
        elif self.action_type == 'continuous':
            self.action_space = gym.spaces.Box(low=np.array([self.v_range.min, self.w_range.min]),
                                               high=np.array([self.v_range.max, self.w_range.max]),
                                               shape=(2,),
                                               dtype=np.float32)
        elif self.action_type == 'multi_discrete':
            self.action_space = gym.spaces.MultiDiscrete(nvec=self.nvec)
        else:
            raise NotImplementedError(
                f'Available action spaces are ["discrete", "continuous", "multi_discrete"], got {action_type}'
            )

        # Agents
        self.agent = MowerAgent()
        self.last_state = None
        self.map_id = None
        self.map_distance = None
        self.map_obstacle = None
        self.map_trajectory = None
        self.map_frontier = None
        self.t = None
        # Misc
        self.render_mode = render_mode
        self.state_pixels = state_pixels
        self.screen = None
        self.clock = None
        self.isopen = True

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

    def observation(self) -> np.ndarray:
        obs = np.stack((
            self.map_frontier,
            self.map_obstacle,
            np.logical_and(self.map_weed, np.logical_not(self.map_frontier)),
            self.map_trajectory,
        ), axis=-1)
        diag_r = self.state_size[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)
        obs = cv2.copyMakeBorder(obs, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                 cv2.BORDER_CONSTANT, value=np.array((0., 1., 0., 0.)), )
        leftmost = round(self.agent.y)
        rightmost = round(self.agent.y + 2 * diag_r_int)
        upmost = round(self.agent.x)
        bottommost = round(self.agent.x + 2 * diag_r_int)
        obs_cropped = obs[leftmost:rightmost, upmost:bottommost, :]

        rotation_mat = cv2.getRotationMatrix2D((diag_r, diag_r), 180 + self.agent.direction, 1.0)
        dst_size = 2 * diag_r_int
        delta_leftmost = int(diag_r_int - self.state_size[0] / 2)
        delta_rightmost = delta_leftmost + self.state_size[0]
        obs_rotated = cv2.warpAffine(obs_cropped.astype(np.float32), rotation_mat, (dst_size, dst_size))
        obs_rotated = obs_rotated[
                      delta_leftmost:delta_rightmost,
                      delta_leftmost:delta_rightmost,
                      :]
        # _range = np.max(obs_rotated[:, :, 0]) - np.min(obs_rotated[:, :, 0])
        # obs_rotated[:, :, 0] = (obs_rotated[:, :, 0] - np.min(obs_rotated[:, :, 0])) / _range
        self.obs_ego_centric = obs_rotated
        obs_rotated_resize = cv2.resize(obs_rotated, self.state_downsize)
        obs = obs_rotated_resize.transpose(2, 0, 1)
        if self.use_sgcnn:
            obs = self.get_sgcnn_obs(obs)
        return obs


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
