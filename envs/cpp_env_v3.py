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
            map_weed_= self.map_weed_noisy
        else:
            map_weed_ = self.map_weed
        maps_list = [
            np.logical_and(self.map_frontier, self.map_mist),
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
        rendered_map = np.ones((self.dimensions[1], self.dimensions[0], 3), dtype=np.float32) * 255.
        rendered_map = np.where(
            np.expand_dims(self.map_frontier, axis=-1),
            np.array((255., 215., 0.)),
            rendered_map,
        )
        mask_tv_cols = self.map_frontier[1:, :] - self.map_frontier[:-1, :] != 0
        mask_tv_cols = np.pad(mask_tv_cols, pad_width=[[0, 1], [0, 0]], mode='constant')
        mask_tv_rows = self.map_frontier[:, 1:] - self.map_frontier[:, :-1] != 0
        mask_tv_rows = np.pad(mask_tv_rows, pad_width=[[0, 0], [0, 1]], mode='constant')
        mask_tv = np.logical_or(mask_tv_rows, mask_tv_cols)
        rendered_map = np.where(
            np.expand_dims(mask_tv, axis=-1),
            np.array((255., 38., 255.)),
            rendered_map,
        )
        cv2.ellipse(img=rendered_map,
                    center=self.agent.position_discrete,
                    axes=(self.vision_length, self.vision_length),
                    angle=self.agent.direction,
                    startAngle=-self.vision_angle / 2,
                    endAngle=self.vision_angle / 2,
                    color=(192., 192., 192.),
                    thickness=-1,
                    )
        weed_undiscovered = get_map_pasture_larger(np.logical_and(self.map_weed, self.map_frontier))
        weed_discovered = get_map_pasture_larger(np.logical_and(self.map_weed, np.logical_not(self.map_frontier)))
        rendered_map = np.where(
            np.expand_dims(weed_undiscovered, axis=-1),
            np.array((255., 0., 0.)),
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(weed_discovered, axis=-1),
            np.array((0., 255., 0.)),
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(self.map_obstacle, axis=-1),
            128.,
            rendered_map,
        )
        cv2.fillPoly(rendered_map, [self.agent.convex_hull.round().astype(np.int32)], color=(255., 0., 0.))
        rendered_map = np.where(
            np.expand_dims(self.map_trajectory, axis=-1) != 0,
            np.array((0., 255., 255.)),
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(self.map_mist, axis=-1),
            rendered_map,
            (rendered_map * 0.5).astype(np.uint8),
        )
        # cv2.polylines(rendered_map, self.min_area_rect, True, (0, 255, 0), 1)
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
