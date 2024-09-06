from __future__ import annotations

import cv2
import numpy as np
from cpu_apf import cpu_apf_bool  # noqa
from gymnasium.wrappers import HumanRendering

from envs.cpp_env_base import CppEnvBase
from envs.utils import MowerAgent, total_variation


class CppEnv(CppEnvBase):

    def observation(self) -> dict[str, np.ndarray | float]:
        obs = np.stack((
            np.logical_and(self.map_frontier, self.map_mist),
            np.logical_not(self.map_mist),
            self.map_obstacle,
            self.map_weed,
        ), axis=-1)
        diag_r = self.state_size[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)
        obs = cv2.copyMakeBorder(obs, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                 cv2.BORDER_CONSTANT, value=np.array((0., 0., 1., 0.)), )
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
        obs_rotated_resize = cv2.resize(obs_rotated, self.state_downsize)
        obs = obs_rotated_resize.transpose(2, 0, 1)
        if self.use_sgcnn:
            obs = self.get_sgcnn_obs(obs)
        return {'observation': obs,
                'vector': self.agent.last_steer / self.w_range.max,
                'weed_ratio': 1 - self.weed_num_t / self.weed_num}

    def get_reward(self,
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
        # if reward > 5:
        #     pass
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
        env.action_space.seed(66)
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
