from __future__ import annotations

import numpy as np
from gymnasium.wrappers import HumanRendering

from envs.cpp_env_base import CppEnvBase


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
