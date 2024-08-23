from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers import HumanRendering

from envs.utils import get_map_pasture_larger, MowerAgent, NumericalRange


class CppEnvironment(gym.Env):
    metadata = {
        "render_modes": [
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": 50,
    }

    vision_length = 24
    vision_angle = 60
    v_range = NumericalRange(0.0, 3.5)
    w_range = NumericalRange(-28.0, 28.0)
    nvec = (7, 21)

    def __init__(
            self,
            # dimensions: tuple[int, int] = (400, 400),
            action_type: str = 'discrete',
            render_mode: str = None,
            state_pixels: bool = False,
            state_size: tuple[int, int] = (128, 128),
            state_downsize: tuple[int, int] = (128, 128),
            num_obstacles_range: tuple[int, int] = (5, 8),
            use_sgcnn: bool = True,
            use_global_obs: bool = True,
            map_dir: str = 'envs/maps/1-600',
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

    @property
    def state_size_diag(self) -> tuple[int, int]:
        return (np.ceil(np.sqrt(2) * self.state_size[0]).astype(np.int32),
                np.ceil(np.sqrt(2) * self.state_size[1]).astype(np.int32))

    @property
    def dimensions_diag(self) -> tuple[int, int]:
        return (np.ceil(np.sqrt(2) * self.dimensions[0]).astype(np.int32),
                np.ceil(np.sqrt(2) * self.dimensions[1]).astype(np.int32))

    def get_action(self, action: Union[int, Tuple[int, int], Tuple[float, float]]) -> tuple[float, float]:
        if self.action_type == 'discrete':
            acc = action // self.nvec[1]
            linear_velocity = (self.v_range.min
                               + (acc + 1) / (self.nvec[0]) * self.v_range.mode)
            steer = action % self.nvec[1]
            angular_velocity = (self.w_range.min
                                + steer / (self.nvec[1] - 1) * self.w_range.mode)
        elif self.action_type == 'continuous':
            linear_velocity, angular_velocity = action
        elif self.action_type == 'multi_discrete':
            acc = action[0]
            linear_velocity = (self.v_range.min
                               + (acc + 1) / (self.nvec[0]) * self.v_range.mode)
            steer = action[1]
            angular_velocity = (self.w_range.min
                                + steer / (self.nvec[1] - 1) * self.w_range.mode)
        else:
            raise NotImplementedError(
                f'Available action spaces are ["discrete", "continuous", "multi_discrete"], got {self.action_type}'
            )
        return linear_velocity, angular_velocity

    def step(self, action: Union[int, Tuple[int, int], Tuple[float, float]]):
        x_t, y_t = self.agent.position_discrete
        acc, steer = self.get_action(action)
        self.agent.control(acc, steer)
        x_tp1, y_tp1 = self.agent.position_discrete
        cv2.fillPoly(self.map_weed, [self.agent.convex_hull.round().astype(np.int32)], color=(0.,))
        cv2.ellipse(img=self.map_frontier,
                    center=self.agent.position_discrete,
                    axes=(self.vision_length, self.vision_length),
                    angle=self.agent.direction,
                    startAngle=-self.vision_angle / 2,
                    endAngle=self.vision_angle / 2,
                    color=(0.,),
                    thickness=-1,
                    )
        cv2.line(self.map_trajectory, pt1=(x_t, y_t), pt2=(x_tp1, y_tp1), color=(1.,))
        reward, done = self.reward_and_done()
        obs = self.observation()
        return obs, reward, done, False, {}

    def reward_and_done(self) -> tuple[float, bool]:
        convex_hull = self.agent.convex_hull
        crashed: bool = (not (
                (0 < convex_hull[:, 0])
                & (convex_hull[:, 0] < self.dimensions[0])
                & (0 < convex_hull[:, 1])
                & (convex_hull[:, 1] < self.dimensions[1])
        ).all())
        if self.agent.x < 0 or self.agent.x > self.dimensions[0]:
            self.agent.x = float(np.clip(self.agent.x, 0, self.dimensions[0]))
        if self.agent.y < 0 or self.agent.y > self.dimensions[1]:
            self.agent.y = float(np.clip(self.agent.y, 0, self.dimensions[1]))
        if not crashed:
            reward_const = -1.
            reward = (reward_const)
        else:
            reward = -100.
        self.t += 1
        time_out = self.t == 2000
        done = crashed or time_out
        return reward, done

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

    def get_sgcnn_obs(self, obs: np.ndarray):
        sgcnn_size = 16
        # obs_ = obs.transpose(1, 2, 0)
        obs_ = obs
        # obs_ = torch.from_numpy(obs).type(dtype=torch.float32)
        obs_list = []
        center_size = self.state_downsize[0] // 2
        with torch.no_grad():
            for _ in range(4):
                obs_list.append(obs_[
                                :,
                                (center_size - sgcnn_size // 2):(center_size + sgcnn_size // 2),
                                (center_size - sgcnn_size // 2):(center_size + sgcnn_size // 2),
                                ])
                obs_ = F.max_pool2d(torch.from_numpy(obs_), (2, 2), 2).numpy()
                center_size //= 2
            if self.use_global_obs:
                obs_global = np.stack((
                    self.map_frontier,
                    self.map_obstacle,
                    self.map_weed,
                    self.map_trajectory,
                ), axis=-1)
                diag_r = self.dimensions[0] / 2 * np.sqrt(2)
                diag_r_int = np.ceil(diag_r).astype(np.int32)
                obs_global = cv2.copyMakeBorder(obs_global, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                                cv2.BORDER_CONSTANT, value=np.array((0., 1., 0., 0.)), )
                leftmost = round(self.agent.y)
                rightmost = round(self.agent.y + 2 * diag_r_int)
                upmost = round(self.agent.x)
                bottommost = round(self.agent.x + 2 * diag_r_int)
                obs_cropped = obs_global[leftmost:rightmost, upmost:bottommost, :]

                rotation_mat = cv2.getRotationMatrix2D((diag_r, diag_r), 180 + self.agent.direction, 1.0)
                dst_size = 2 * diag_r_int
                delta_leftmost = int(diag_r_int - self.dimensions[0] / 2)
                delta_rightmost = delta_leftmost + self.dimensions[0]
                obs_global = cv2.warpAffine(obs_cropped.astype(np.float32), rotation_mat, (dst_size, dst_size))
                obs_global = obs_global[
                             delta_leftmost:delta_rightmost,
                             delta_leftmost:delta_rightmost,
                             :]
                obs_global = obs_global.transpose(2, 0, 1)
                kernel_size = int(np.round(self.dimensions[0] / sgcnn_size)) - 1
                obs_global = F.max_pool2d(torch.from_numpy(obs_global),
                                          (kernel_size, kernel_size),
                                          kernel_size).numpy()
                obs_list.append(obs_global)
        return np.concatenate(obs_list, axis=0, dtype=np.float32)

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
        # rendered_map = np.where(
        #     np.expand_dims(self.map_distance, axis=-1),
        #     np.expand_dims(self.map_distance, axis=-1) * np.array((255., 0., 0.)),
        #     rendered_map,
        # )
        cv2.fillPoly(rendered_map, [self.agent.convex_hull.round().astype(np.int32)], color=(255., 0., 0.))
        rendered_map = np.where(
            np.expand_dims(self.map_trajectory, axis=-1) != 0,
            np.array((0., 255., 255.)),
            rendered_map,
        )
        cv2.polylines(rendered_map, self.min_area_rect, True, (0, 255, 0), 1)
        rendered_map = rendered_map.repeat(2, axis=0).repeat(2, axis=1)
        return rendered_map

    def render_self(self) -> np.ndarray:
        # obs_rotated_resize = cv2.resize(self.obs_ego_centric, self.state_size_downsize)
        rendered_map = self.render_map()
        diag_r = self.state_size[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)
        obs = cv2.copyMakeBorder(rendered_map, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                 cv2.BORDER_CONSTANT, value=np.array((128., 128., 128.)), )
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
        return obs_rotated

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Parse Options
        position = None
        theta = None
        weed_dist = None
        weed_num = None
        map_id = None
        if isinstance(options, dict):
            if 'position' in options:
                position = options['position']
            if 'theta' in options:
                theta = options['theta']
            if 'weed_dist' in options:
                weed_dist = options['weed_dist']
            if 'weed_num' in options:
                weed_num = options['weed_num']
            if 'map_id' in options:
                map_id = options['map_id']
        if position is None:
            position = (
                self.np_random.uniform(self.dimensions[0] * 0.1, self.dimensions[0] * 0.9),
                self.np_random.uniform(self.dimensions[1] * 0.1, self.dimensions[1] * 0.9),
            )
        if theta is None:
            theta = self.np_random.uniform(0., 360.)
        if weed_dist is None:
            weed_dist = 'uniform'
        if weed_num is None:
            weed_num = 100
        if map_id is None:
            map_id = self.np_random.integers(0, len(self.map_names) - 1)
        # Check Parameters' Range
        agent_occupy = math.hypot(MowerAgent.length, MowerAgent.width)
        assert isinstance(position, tuple) and len(position) == 2
        assert self.dimensions[0] - agent_occupy > 0 and self.dimensions[0] - agent_occupy > agent_occupy
        assert self.dimensions[1] - agent_occupy > 0 and self.dimensions[1] - agent_occupy > agent_occupy
        assert agent_occupy < position[0] < self.dimensions[0] - agent_occupy
        assert agent_occupy < position[1] < self.dimensions[1] - agent_occupy
        assert weed_dist in {'uniform', 'gaussian'}
        assert 0 <= map_id <= len(self.map_names) - 1
        self.map_id = map_id
        self.map_frontier: np.ndarray = (
                cv2.imread(self.map_dir / self.map_names[self.map_id]).sum(axis=-1) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(self.map_frontier, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        box = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])
        # box = cv2.boxPoints(rect)
        start_idx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_idx, 0)
        # (4, 1, 2)
        box = box.reshape((-1, 1, 2)).astype(np.int32)
        self.min_area_rect = [box]
        pos_index = 0 if math.hypot(*(box[1, 0] - box[0, 0])) > math.hypot(*(box[2, 0] - box[1, 0])) else 1
        position = (float(box[pos_index, 0, 0]), float(box[pos_index, 0, 1]))
        theta_vector = box[pos_index + 1, 0] - box[pos_index, 0]
        theta_vector = theta_vector / math.hypot(*theta_vector)
        theta = math.degrees(math.atan2(theta_vector[1], theta_vector[0]))
        # cv2.polylines(image, [box], True, (0, 255, 0), 10)

        # Initialize player
        self.agent.reset(
            position=position,
            direction=theta,
        )
        # Initialize maps
        self.map_trajectory = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        # Randomize obstacles
        self.map_obstacle = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        num_obstacles = self.np_random.integers(*self.num_obstacles_range) if self.num_obstacles_range[1] > 0 else 0
        current_obstacle_num = 0
        # self.obstacles = []
        while current_obstacle_num < num_obstacles:
            o_x = self.np_random.uniform(0 + 100, self.dimensions[0] - 100)
            o_y = self.np_random.uniform(0 + 100, self.dimensions[1] - 100)
            o_len = self.np_random.uniform(10, 80)
            o_wid = self.np_random.uniform(10, 80)
            angle = self.np_random.uniform(0., 360.)
            pts = np.array(
                cv2.RotatedRect(center=(o_x, o_y), size=(o_len, o_wid), angle=angle).points(), dtype=np.int32
            ).reshape((-1, 1, 2))
            dist2player = cv2.pointPolygonTest(pts, self.agent.position, True)
            if dist2player < -2.0 * MowerAgent.length:
                current_obstacle_num += 1
                cv2.fillPoly(self.map_obstacle, [pts], color=(1.))
                pts = np.array(
                    cv2.RotatedRect(center=(o_x, o_y), size=(o_len + 15, o_wid + 15), angle=angle).points(), dtype=np.int32
                ).reshape((-1, 1, 2))
                cv2.fillPoly(self.map_frontier, [pts], color=(0.))
                # cv2.fillPoly(self.map_obstacle, [pts], color=(1.))
                # self.obstacles.append((o_x, o_y, o_len, o_wid))
        self.map_frontier_origin = self.map_frontier
        self.map_weed = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        weed_count = 0
        while weed_count < weed_num:
            if weed_dist == 'uniform':
                weed_x = self.np_random.integers(low=0, high=self.dimensions[0] - 1)
                weed_y = self.np_random.integers(low=0, high=self.dimensions[1] - 1)
            else:
                weed_x = self.np_random.normal(loc=0., scale=1.)
                weed_x = np.round((self.dimensions[1] / 2) * weed_x + self.dimensions[1] / 2).astype(np.int32)
                weed_x = np.clip(weed_x, 0, self.dimensions[0] - 1, dtype=np.int32)
                weed_y = self.np_random.normal(loc=0., scale=1.)
                weed_y = np.round((self.dimensions[1] / 2) * weed_y + self.dimensions[1] / 2).astype(np.int32)
                weed_y = np.clip(weed_y, 0, self.dimensions[1] - 1, dtype=np.int32)
            if self.map_frontier[weed_y, weed_x] and not self.map_weed[weed_y, weed_x]:
                self.map_weed[weed_y, weed_x] = 1
                weed_count += 1
        # Get observation
        obs = self.observation()
        self.t = 1
        return obs, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface(
                (self.state_size[0], self.state_size[1]) if self.state_pixels else (
                    self.dimensions[0] * 2, self.dimensions[1] * 2)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state_pixels:
            img = self.render_self()
        else:
            img = self.render_map()
        surf = pygame.surfarray.make_surface(img)
        self.screen.blit(surf, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == "__main__":
    if_render = True
    episodes = 3
    env = CppEnvironment(
        render_mode='rgb_array' if if_render else None,
        # state_pixels=True,
        state_pixels=False,
    )
    env: CppEnvironment = HumanRendering(env)

    for _ in range(episodes):
        obs, info = env.reset(options={
            'weed_dist': 'gaussian'
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
