from __future__ import annotations

import cv2
import numpy as np
from cpu_apf import cpu_apf_bool  # noqa
from gymnasium.wrappers import HumanRendering

from envs.cpp_env_base import CppEnvBase
from envs.utils import get_map_pasture_larger


class CppEnv(CppEnvBase):
    """
    This env contains mist that the agent must explore to know what's inside.
    """

    def get_maps_and_mask(self) -> tuple[np.ndarray, list[float]]:
        if self.noise_weed and self.np_random.uniform() < self.noise_weed:
            map_weed_ = self.map_weed_noisy
        else:
            map_weed_ = self.map_weed
        maps_list = [
            # np.logical_and(self.map_frontier, self.map_mist),
            self.map_trajectory,
            np.logical_not(self.map_mist),
            self.map_obstacle,
            np.logical_and(map_weed_, np.logical_not(self.map_frontier)),
        ]
        mask = [0., 0., 1., 0.]
        if self.use_traj:
            maps_list.append(self.map_trajectory)
            mask.append(0.)
        maps = np.stack(maps_list, axis=-1)
        return maps, mask

    def render_map(self) -> np.ndarray:
        rendered_map = super(CppEnv, self).render_map()
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
