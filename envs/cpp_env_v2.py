from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from cpu_apf import cpu_apf_bool  # noqa
from gymnasium.wrappers import HumanRendering
from sympy.codegen.cnodes import static

from envs.cpp_env_base import CppEnvBase
from envs.utils import get_map_pasture_larger, total_variation_mat


class CppEnv(CppEnvBase):
    """
    Using Apf Of Frontier Edges as observation.
    """
    render_farmland_outsides = False  # 是否渲染农田外的扩散场
    render_weed = False  # 是否渲染操的扩散场
    render_obstacle = False  # 是否渲染障碍物的扩散场

    @staticmethod
    def get_discounted_apf(map_apf: np.ndarray, max_step: int, eps: Optional[float] = None) -> float:
        map_apf, is_empty = cpu_apf_bool(map_apf)
        if not is_empty:
            gamma = (max_step - 1) / max_step
            map_apf = gamma ** map_apf
            if eps is None:
                eps = gamma ** max_step
            map_apf = np.where(map_apf < eps, 0., map_apf)
        return map_apf

    def get_maps_and_mask(self) -> tuple[np.ndarray, list[float]]:
        if self.noise_weed and self.np_random.uniform() < self.noise_weed:
            map_weed_ = self.map_weed_noisy
        else:
            map_weed_ = self.map_weed
        apf_frontier = np.logical_and(total_variation_mat(self.map_frontier), self.map_mist)
        apf_obstacle = np.logical_and(total_variation_mat(self.map_obstacle), self.map_mist)
        apf_weed = np.logical_and(map_weed_, np.logical_not(self.map_frontier))
        apf_trajectory = self.map_trajectory
        if self.use_apf:
            apf_frontier = self.get_discounted_apf(apf_frontier, 30)
            apf_weed = self.get_discounted_apf(apf_weed, 40, 1e-2)
            apf_trajectory = self.get_discounted_apf(apf_trajectory, 4)
            apf_obstacle = np.logical_and(np.pad(apf_obstacle,
                                                 pad_width=[[1, 1], [1, 1]],
                                                 mode='constant',
                                                 constant_values=(1, 1)),
                                          np.pad(self.map_mist,
                                                 pad_width=[[1, 1], [1, 1]],
                                                 mode='constant',
                                                 constant_values=(1, 1)))
            apf_obstacle = self.get_discounted_apf(apf_obstacle, 10)
            apf_obstacle = apf_obstacle[1:-1, 1:-1]
        apf_obstacle = np.maximum(apf_obstacle, np.logical_and(self.map_obstacle, self.map_mist))
        maps_list = [
            apf_frontier,
            np.logical_not(self.map_mist),
            apf_obstacle,
            apf_weed,
        ]
        mask = [0., 0., 1., 0.]
        if self.use_traj:
            maps_list.append(apf_trajectory)
            mask.append(0.)
        maps = np.stack(maps_list, axis=-1)
        self.obs_apf = maps.transpose(2, 0, 1)
        return maps, mask


def get_extra_reward(self,
                     steer_tp1: float,
                     x_t: int,
                     y_t: int,
                     x_tp1: int,
                     y_tp1: int) -> float:
    reward_apf = 0.
    if self.use_apf:
        reward_apf_frontier = 0.0 * (self.obs_apf[0][y_tp1, x_tp1] - self.obs_apf[0][y_t, x_t])
        reward_apf_obstacle = 0.3 * (self.obs_apf[2][y_tp1, x_tp1] - self.obs_apf[2][y_t, x_t])
        reward_apf_obstacle = min(0., reward_apf_obstacle)
        reward_apf_weed = 5.0 * (self.obs_apf[3][y_tp1, x_tp1] - self.obs_apf[3][y_t, x_t])
        reward_apf_traj = 0.
        if self.use_traj:
            reward_apf_traj = 0.0 * (self.obs_apf[4][y_tp1, x_tp1] - self.obs_apf[4][y_t, x_t])
            reward_apf_traj = min(0., reward_apf_traj)
        reward_apf = 1.0 * (reward_apf_frontier
                            + reward_apf_obstacle
                            + reward_apf_weed
                            + reward_apf_traj
                            )
    return reward_apf


def render_map(self) -> np.ndarray:
    rendered_map = np.ones((self.dimensions[1], self.dimensions[0], 3), dtype=np.uint8) * 255.
    if self.render_farmland_outsides:
        frontier_apf_out = np.where(np.logical_not(self.map_frontier_full), self.obs_apf[0],
                                    0.)  # frontier外部有态势场的地方
        rendered_map = np.where(
            np.expand_dims(frontier_apf_out, axis=-1),
            (
                    np.expand_dims(frontier_apf_out, axis=-1) * np.array((255, 38, 255))
                    + np.expand_dims(1 - frontier_apf_out, axis=-1) * rendered_map
            ).astype(np.uint8),
            rendered_map,
        )
    frontier_apf_in = np.where(self.map_frontier_full, self.obs_apf[0], 0.)
    rendered_map = np.where(
        np.expand_dims(frontier_apf_in, axis=-1),
        (
                np.expand_dims(frontier_apf_in, axis=-1) * np.array((255., 215., 0.))
                + np.expand_dims(1 - frontier_apf_in, axis=-1) * rendered_map
        ).astype(np.uint8),
        rendered_map,
    )
    tv_frontier = total_variation_mat(self.map_frontier)
    rendered_map = np.where(
        np.expand_dims(tv_frontier, axis=-1),
        np.array((255, 38, 255)),
        rendered_map,
    )
    cv2.ellipse(img=rendered_map,
                center=self.agent.position_discrete,
                axes=(self.vision_length, self.vision_length),
                angle=self.agent.direction,
                startAngle=-self.vision_angle / 2,
                endAngle=self.vision_angle / 2,
                color=(192, 192, 192),
                thickness=-1,
                )
    weed_undiscovered = get_map_pasture_larger(np.logical_and(self.map_weed, self.map_frontier))
    rendered_map = np.where(
        np.expand_dims(weed_undiscovered, axis=-1),
        np.array((255, 0, 0)),
        rendered_map,
    )
    weed_discovered = get_map_pasture_larger(np.logical_and(self.map_weed, np.logical_not(self.map_frontier)))
    rendered_map = np.where(
        np.expand_dims(weed_discovered, axis=-1),
        np.array((0, 255, 0)),
        rendered_map,
    )
    if self.render_weed:
        rendered_map = np.where(
            np.expand_dims(self.obs_apf[3] != 0, axis=-1),
            (
                    np.expand_dims(self.obs_apf[3], axis=-1) * np.array((0, 255, 0))
                    + np.expand_dims(1 - self.obs_apf[3], axis=-1) * rendered_map
            ).astype(np.uint8),
            rendered_map,
        )
    rendered_map = np.where(
        np.expand_dims(self.map_trajectory, axis=-1),
        np.array((0, 128, 255)),
        rendered_map,
    )
    rendered_map = np.where(
        np.expand_dims(self.map_obstacle, axis=-1),
        np.array((128, 128, 128)),
        rendered_map,
    )
    tv_obstacle = total_variation_mat(self.map_obstacle)
    rendered_map = np.where(
        np.expand_dims(tv_obstacle, axis=-1),
        np.array((48, 48, 48)),
        rendered_map,
    )
    if self.render_obstacle:
        rendered_map = np.where(
            np.expand_dims(np.logical_and(self.obs_apf[2], np.logical_not(self.map_obstacle)), axis=-1),
            (
                    np.expand_dims(self.obs_apf[2], axis=-1) * np.array((48., 48., 48.))
                    + np.expand_dims(1 - self.obs_apf[2], axis=-1) * rendered_map
            ).astype(np.uint8),
            rendered_map,
        )
    # cv2.polylines(rendered_map, self.min_area_rect, True, (0, 255, 0), 1)
    cv2.fillPoly(rendered_map, [self.agent.convex_hull.round().astype(np.int32)], color=(255, 0, 0))
    rendered_map = np.where(
        np.expand_dims(self.map_mist, axis=-1),
        rendered_map,
        (rendered_map * 0.5).astype(np.uint8),
    )
    return rendered_map


if __name__ == "__main__":
    if_render = True
    episodes = 3
    env = CppEnv(
        render_mode='rgb_array' if if_render else None,
        state_pixels=True,
        # state_pixels=False,
        # num_obstacles_range = [0,0]
    )
    env: CppEnv = HumanRendering(env)  # 封装后，只接收render_mode="rgb_array"的env，使得step和reset的时候展示渲染图像

    for _ in range(episodes):
        # env.set_obstacle_range([0,0])
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
                img = env.render()

    env.close()
