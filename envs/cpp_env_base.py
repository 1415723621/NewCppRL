from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple, Union, Any, Sequence

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from cpu_apf import cpu_apf_bool  # noqa
from gymnasium.error import DependencyNotInstalled

from envs.utils import get_map_pasture_larger, MowerAgent, NumericalRange, total_variation, total_variation_mat


class CppEnvBase(gym.Env):
    metadata = {
        "render_modes": [
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": 50,
    }

    vision_length = 28
    vision_angle = 75

    v_range = NumericalRange(0.0, 3.5)
    w_range = NumericalRange(-28.6, 28.6)
    nvec = (7, 21)

    obstacle_size_range = (10, 25)
    sgcnn_size = 16

    render_repeat_times = 1  # 渲染时放缩比例

    render_tv = False
    render_covered_weed = True

    def __init__(
            self,
            action_type: str = 'discrete',
            render_mode: str = None,
            state_pixels: bool = False,  # 是否渲染第一视角
            state_size: tuple[int, int] = (128, 128),
            state_downsize: tuple[int, int] = (128, 128),
            num_obstacles_range: tuple[int, int] = (5, 8),
            use_sgcnn: bool = True,
            use_global_obs: bool = True,
            use_apf: bool = True,
            use_box_boundary: bool = True,  # 在场景外是否绘制障碍物包裹
            use_traj: bool = True,
            noise_position: Optional[float] = 0.0,
            noise_direction: Optional[float] = 0.0,
            noise_weed: Optional[float] = 0.0,
            map_dir: str = 'envs/maps/1-400',
    ):
        super().__init__()
        # Read maps
        self.map_dir = Path(__file__).parent.parent / map_dir
        self.map_names = sorted(os.listdir(self.map_dir))
        # Environmental parameters
        self.action_type = action_type
        self.dimensions = cv2.imread(str(self.map_dir / self.map_names[0])).shape[:-1]  # 先列数再行数
        self.state_size = state_size
        self.state_downsize = state_downsize
        self.num_obstacles_range = num_obstacles_range
        self.use_sgcnn = use_sgcnn
        self.use_global_obs = use_global_obs
        self.use_apf = use_apf
        self.use_box_boundary = use_box_boundary
        self.use_traj = use_traj
        self.noise_position = noise_position
        self.noise_direction = noise_direction
        self.noise_weed = noise_weed
        total_channels = 4 + use_traj
        obs_shape = (total_channels, *self.state_downsize)
        if use_sgcnn:
            obs_shape = (total_channels * (4 + use_global_obs), *(ds // 8 for ds in self.state_downsize))

        # RL parameters
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=0., high=1., shape=obs_shape, dtype=np.float32
            ),
            'vector': gym.spaces.Box(
                low=-1., high=1., dtype=np.float32
            ),
            'weed_ratio': gym.spaces.Box(
                low=0., high=1., dtype=np.float32
            ),
        })
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
        self.map_frontier = None
        self.map_obstacle = None
        self.map_weed = None
        self.map_trajectory = None
        self.t = None
        self.weed_num_t = None
        self.frontier_area_t = None
        self.frontier_tv_t = None
        self.steer_t = None
        # Misc
        self.render_mode = render_mode
        self.state_pixels = state_pixels
        self.screen = None
        self.clock = None
        self.isopen = True

    # state对角尺寸
    @property
    def state_size_diag(self) -> tuple[int, int]:
        return (np.ceil(np.sqrt(2) * self.state_size[0]).astype(np.int32),
                np.ceil(np.sqrt(2) * self.state_size[1]).astype(np.int32))

    # 图片对角尺寸
    @property
    def dimensions_diag(self) -> tuple[int, int]:
        return (np.ceil(np.sqrt(2) * self.dimensions[0]).astype(np.int32),
                np.ceil(np.sqrt(2) * self.dimensions[1]).astype(np.int32))

    def set_action_type(self, action_type: str = "discrete"):
        self.action_type = action_type

    def set_obstacle_range(self, num_obstacles_range: tuple[int, int] = (5, 8)):
        self.num_obstacles_range = num_obstacles_range

    def get_action(self, action: Union[int, Tuple[int, int], Tuple[float, float]]) -> tuple[float, float]:
        if self.action_type == 'discrete':
            acc = action // self.nvec[1]  # //之后在0~6范围
            linear_velocity = (self.v_range.min
                               + (acc + 1) / (self.nvec[0]) * self.v_range.mode)  # 通过+1将速度最低档设置为0.5,+后面是做线性的放缩
            steer = action % self.nvec[1]  # %后在0~20范围
            angular_velocity = (self.w_range.min
                                + steer / (self.nvec[1] - 1) * self.w_range.mode)  # -1去掉中位数，保证线性放缩合理
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
                f'Available action spaces are ["discrete", "continuous", "multi_discrete"], got "{self.action_type}"'
            )
        return linear_velocity, angular_velocity

    def step(self, action: Union[int, Tuple[int, int], Tuple[float, float]]):
        x_t, y_t = self.agent.position_discrete
        acc, steer = self.get_action(action)
        self.agent.control(acc, steer)
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
        cv2.ellipse(img=self.map_mist,
                    center=self.agent.position_discrete,
                    axes=(self.vision_length + 1, self.vision_length + 1),
                    angle=self.agent.direction,
                    startAngle=-self.vision_angle / 2,
                    endAngle=self.vision_angle / 2,
                    color=(1.,),
                    thickness=-1,
                    )
        crashed = self.check_collision()
        x_tp1, y_tp1 = self.agent.position_discrete
        x_t = max(min(x_t, self.dimensions[0] - 1), 0)
        y_t = max(min(y_t, self.dimensions[1] - 1), 0)
        x_tp1 = max(min(x_tp1, self.dimensions[0] - 1), 0)
        y_tp1 = max(min(y_tp1, self.dimensions[1] - 1), 0)
        cv2.line(self.map_trajectory, pt1=(x_t, y_t), pt2=(x_tp1, y_tp1), color=(1.,))
        reward = self.get_reward(steer, x_t, y_t, x_tp1, y_tp1)
        if crashed:
            reward -= 399.
        self.t += 1
        time_out = self.t == 3000
        # finish = self.weed_num_t == 0 and self.frontier_area_t == 0
        finish = self.weed_num_t == 0  # TODO: 这样楚瑜就没办法测试了，因此记得解决，看看能不能解决吧
        if finish:
            reward += 500
        done = crashed or finish
        obs = self.observation()
        return obs, reward, done, time_out, {}

    def get_extra_reward(self,
                         steer_tp1: float,
                         x_t: int,
                         y_t: int,
                         x_tp1: int,
                         y_tp1: int) -> float:
        return 0.

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
        reward_extra = self.get_extra_reward(steer_tp1, x_t, y_t, x_tp1, y_tp1)
        # Summary
        reward = (reward_const
                  + reward_frontier
                  + reward_weed
                  + reward_extra
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

    def check_collision(self) -> tuple[float, bool]:
        convex_hull = self.agent.convex_hull
        map_agent = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        cv2.fillPoly(map_agent, [self.agent.convex_hull.round().astype(np.int32)], color=(1,))
        crashed_bounds: bool = (not (
                (0 < convex_hull[:, 0])
                & (convex_hull[:, 0] < self.dimensions[0])
                & (0 < convex_hull[:, 1])
                & (convex_hull[:, 1] < self.dimensions[1])
        ).all())
        crashed_obstacles: bool = np.logical_and(map_agent, self.map_obstacle).any()
        self.agent.x = float(np.clip(self.agent.x, 0, self.dimensions[0]))
        self.agent.y = float(np.clip(self.agent.y, 0, self.dimensions[1]))
        crashed = np.logical_or(crashed_bounds, crashed_obstacles)
        return crashed

    @staticmethod
    def get_discounted_apf(map_apf: np.ndarray, max_step: int, eps: Optional[float] = None) -> float:
        gamma = (max_step - 1) / max_step
        map_apf = gamma ** map_apf
        if eps is None:
            eps = gamma ** max_step
        return np.where(map_apf < eps, 0., map_apf)

    def get_rotated_obs_(self, maps, mask: Sequence[float]):
        agent_y = self.agent.y
        agent_x = self.agent.x
        if self.noise_position:
            delta_y = np.clip(self.np_random.normal(0, self.noise_position), -self.noise_position, self.noise_position)
            delta_x = np.clip(self.np_random.normal(0, self.noise_position), -self.noise_position, self.noise_position)
            agent_y += delta_y
            agent_x += delta_x
        agent_direction = self.agent.direction
        if self.noise_direction:
            delta_direction = np.clip(self.np_random.normal(0, self.noise_direction),
                                      -self.noise_direction,
                                      self.noise_direction)
            agent_direction += delta_direction
            agent_direction %= 360

        diag_r = self.state_size[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)
        maps = cv2.copyMakeBorder(maps, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                  cv2.BORDER_CONSTANT, value=mask, )
        leftmost = round(agent_y)
        rightmost = round(agent_y + 2 * diag_r_int)
        upmost = round(agent_x)
        bottommost = round(agent_x + 2 * diag_r_int)
        obs_cropped = maps[leftmost:rightmost, upmost:bottommost]

        rotation_mat = cv2.getRotationMatrix2D((diag_r, diag_r), 180 + agent_direction, 1.0)
        dst_size = 2 * diag_r_int
        delta_leftmost = int(diag_r_int - self.state_size[0] / 2)
        delta_rightmost = delta_leftmost + self.state_size[0]
        obs_rotated = cv2.warpAffine(obs_cropped.astype(np.float32), rotation_mat, (dst_size, dst_size))
        obs_rotated = obs_rotated[delta_leftmost:delta_rightmost, delta_leftmost:delta_rightmost]
        if obs_rotated.ndim == 2:
            obs_rotated = obs_rotated.reshape(*obs_rotated.shape, 1)
        return obs_rotated

    def get_rotated_obs(self, maps: np.ndarray, mask: Sequence[float]) -> np.ndarray:
        channel_num = maps.shape[-1]
        ch_beg = 0
        candidates = []
        for i in range(0, channel_num, 4):
            ch_end = min(i + 4, channel_num)
            candidates.append(self.get_rotated_obs_(maps[:, :, ch_beg:ch_end], mask[ch_beg:ch_end]))
            ch_beg = ch_end
        if len(candidates) > 1:
            candidates = np.concatenate(candidates, axis=-1)
        else:
            candidates = candidates[0]
        return candidates

    def get_maps_and_mask(self) -> tuple[np.ndarray, list[float]]:
        raise NotImplementedError

    def observation(self) -> dict[str, np.ndarray | float]:
        maps, mask = self.get_maps_and_mask()
        obs_rotated = self.get_rotated_obs(maps, mask)
        obs_rotated_resize = cv2.resize(obs_rotated, self.state_downsize)
        obs = obs_rotated_resize.transpose(2, 0, 1)
        if self.use_sgcnn:
            if self.use_global_obs:
                obs = self.get_sgcnn_obs(obs, maps, mask)
            else:
                obs = self.get_sgcnn_obs(obs, None, None)
        return {'observation': obs,
                'vector': self.agent.last_steer / self.w_range.max,
                'weed_ratio': 1 - self.weed_num_t / self.weed_num}

    def get_global_obs_(self, maps, mask: Sequence[float]):
        agent_y = self.agent.y
        agent_x = self.agent.x
        if self.noise_position:
            delta_y = np.clip(self.np_random.normal(0, self.noise_position), -self.noise_position, self.noise_position)
            delta_x = np.clip(self.np_random.normal(0, self.noise_position), -self.noise_position, self.noise_position)
            agent_y += delta_y
            agent_x += delta_x
        agent_direction = self.agent.direction
        if self.noise_direction:
            delta_direction = np.clip(self.np_random.normal(0, self.noise_direction),
                                      -self.noise_direction,
                                      self.noise_direction)
            agent_direction += delta_direction
            agent_direction %= 360

        diag_r = self.dimensions[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)
        obs_global = cv2.copyMakeBorder(maps, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                        cv2.BORDER_CONSTANT, value=mask, )
        leftmost = round(agent_y)
        rightmost = round(agent_y + 2 * diag_r_int)
        upmost = round(agent_x)
        bottommost = round(agent_x + 2 * diag_r_int)
        obs_cropped = obs_global[leftmost:rightmost, upmost:bottommost]

        rotation_mat = cv2.getRotationMatrix2D((diag_r, diag_r), 180 + agent_direction, 1.0)
        dst_size = 2 * diag_r_int
        delta_leftmost = int(diag_r_int - self.dimensions[0] / 2)
        delta_rightmost = delta_leftmost + self.dimensions[0]
        obs_global = cv2.warpAffine(obs_cropped.astype(np.float32), rotation_mat, (dst_size, dst_size))
        obs_global = obs_global[delta_leftmost:delta_rightmost, delta_leftmost:delta_rightmost]
        if obs_global.ndim == 2:
            obs_global = obs_global.reshape(*obs_global.shape, 1)
        obs_global = obs_global.transpose(2, 0, 1)
        kernel_size = int(np.round(self.dimensions[0] / self.sgcnn_size)) - 1
        obs_global = F.max_pool2d(torch.from_numpy(obs_global),
                                  (kernel_size, kernel_size),
                                  kernel_size).numpy()
        obs_global = obs_global.transpose(1, 2, 0)
        return obs_global

    def get_global_obs(self, maps: np.ndarray, mask: Sequence[float]) -> np.ndarray:
        channel_num = maps.shape[-1]
        ch_beg = 0
        candidates = []
        for i in range(0, channel_num, 4):
            ch_end = min(i + 4, channel_num)
            candidates.append(self.get_global_obs_(maps[:, :, ch_beg:ch_end], mask[ch_beg:ch_end]))
            ch_beg = ch_end
        if len(candidates) > 1:
            candidates = np.concatenate(candidates, axis=-1)
        else:
            candidates = candidates[0]
        return candidates

    def get_sgcnn_obs(self, obs: np.ndarray,
                      maps: Optional[np.ndarray] = None,
                      mask: Optional[Sequence[float]] = None):  # 在obs的基础上做多尺度图
        obs_ = obs
        obs_list = []
        center_size = self.state_downsize[0] // 2
        with torch.no_grad():
            for _ in range(4):
                obs_list.append(obs_[
                                :,
                                (center_size - self.sgcnn_size // 2):(center_size + self.sgcnn_size // 2),
                                (center_size - self.sgcnn_size // 2):(center_size + self.sgcnn_size // 2),
                                ])
                obs_ = F.max_pool2d(torch.from_numpy(obs_), (2, 2), 2).numpy()
                center_size //= 2
            if self.use_global_obs:
                obs_global = self.get_global_obs(maps, mask)
                obs_global = obs_global.transpose(2, 0, 1)
                obs_list.append(obs_global)
        return np.concatenate(obs_list, axis=0, dtype=np.float32)

    def render_map(self) -> np.ndarray:
        rendered_map = np.ones((self.dimensions[1], self.dimensions[0], 3), dtype=np.float32) * 255
        rendered_map = np.where(
            np.expand_dims(self.map_frontier, axis=-1),
            np.array((112, 173, 71)),
            rendered_map,
        )
        if self.render_tv:
            mask_tv = total_variation_mat(self.map_frontier)
            rendered_map = np.where(
                np.expand_dims(mask_tv, axis=-1),
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
        weed_discovered = get_map_pasture_larger(np.logical_and(self.map_weed, np.logical_not(self.map_frontier)))
        rendered_map = np.where(
            np.expand_dims(weed_undiscovered, axis=-1),
            np.array((56, 86, 35)),
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(weed_discovered, axis=-1),
            np.array((237, 125, 49)),
            rendered_map,
        )
        rendered_map = np.where(
            np.expand_dims(self.map_obstacle, axis=-1),
            np.array((68, 114, 169)),
            rendered_map,
        )
        # if self.render_tv:
        mask_tv = total_variation_mat(self.map_obstacle)
        rendered_map = np.where(
            np.expand_dims(mask_tv, axis=-1),
            np.array((47, 82, 143)),
            rendered_map,
        )
        cv2.fillPoly(rendered_map, [self.agent.convex_hull.round().astype(np.int32)], color=(255., 0., 0.))
        rendered_map = np.where(
            np.expand_dims(self.map_trajectory, axis=-1) != 0,
            np.array((0, 255, 255)),
            rendered_map,
        )
        if self.render_covered_weed:
            weed_covered = get_map_pasture_larger(
                np.logical_and(self.map_weed_ori, np.logical_not(self.map_weed))
            )
            # weed_covered = get_map_pasture_larger(self.map_weed_ori)
            # weed_covered = get_map_pasture_larger(self.map_weed)
            rendered_map = np.where(
                np.expand_dims(weed_covered, axis=-1),
                (
                        0.3 * np.array((48, 48, 48))
                        + 0.7 * rendered_map
                ).astype(np.uint8),
                # np.array((255, 0, 0)),
                rendered_map,
            )
        # cv2.polylines(rendered_map, self.min_area_rect, True, (0, 255, 0), 1)
        return rendered_map

    def render_self(self) -> np.ndarray:
        agent_y = self.agent.y
        agent_x = self.agent.x
        if self.noise_position:
            delta_y = np.clip(self.np_random.normal(0, self.noise_position), -self.noise_position, self.noise_position)
            delta_x = np.clip(self.np_random.normal(0, self.noise_position), -self.noise_position, self.noise_position)
            agent_y += delta_y
            agent_x += delta_x
        agent_direction = self.agent.direction
        if self.noise_direction:
            delta_direction = np.clip(self.np_random.normal(0, self.noise_direction),
                                      -self.noise_direction,
                                      self.noise_direction)
            agent_direction += delta_direction
            agent_direction %= 360

        rendered_map = self.render_map()
        diag_r = self.state_size[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)
        obs = cv2.copyMakeBorder(rendered_map, diag_r_int, diag_r_int, diag_r_int, diag_r_int,
                                 cv2.BORDER_CONSTANT, value=np.array((128., 128., 128.)), )
        leftmost = round(agent_y)
        rightmost = round(agent_y + 2 * diag_r_int)
        upmost = round(agent_x)
        bottommost = round(agent_x + 2 * diag_r_int)
        obs_cropped = obs[leftmost:rightmost, upmost:bottommost, :]

        rotation_mat = cv2.getRotationMatrix2D((diag_r, diag_r), 180 + agent_direction, 1.0)
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
    ) -> tuple[dict[str, np.ndarray | float], dict[Any, Any]]:
        super().reset(seed=seed)
        # Parse Options
        weed_dist = None
        weed_num = None
        map_id = None
        if isinstance(options, dict):
            if 'weed_dist' in options:
                weed_dist = options['weed_dist']
            if 'weed_num' in options:
                weed_num = options['weed_num']
            if 'map_id' in options:
                map_id = options['map_id']
        if weed_dist is None:
            weed_dist = 'uniform'
        if weed_num is None:
            weed_num = 100
        if map_id is None:
            map_id = self.np_random.integers(0, len(self.map_names) - 1)
        # Check Parameters' Range
        assert weed_dist in {'uniform', 'gaussian'}
        assert 0 <= map_id <= len(self.map_names) - 1
        self.map_id = map_id
        self.map_frontier: np.ndarray = (  # map_frontier是农田，这里在计算boudingbox TODO: 有时间细致理解一下这里的代码
                cv2.imread(str(self.map_dir / self.map_names[self.map_id])).sum(axis=-1) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(self.map_frontier, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
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

        # Initialize player
        self.agent.reset(
            position=position,
            direction=theta,
        )
        # Initialize maps
        self.map_trajectory = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        self.map_mist = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        # Randomize obstacles
        if self.use_box_boundary:  # 外部画障碍物
            self.map_obstacle = np.ones((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
            r_center = 0.5 * (box[0, 0] + box[2, 0])
            vecs = [box[0, 0] - box[1, 0], box[1, 0] - box[2, 0]]
            wd_i = 0
            if abs(vecs[1][1]) < abs(vecs[1][0]):
                wd_i = 0
            ht_i = (wd_i + 1) % 2
            angle = math.atan(vecs[wd_i][1] / vecs[wd_i][0]) * 180.0 / math.pi
            width = math.hypot(*vecs[wd_i])
            height = math.hypot(*vecs[ht_i])
            width = max(width * 1.2, width + 60)  # 障碍物空间放大了1.2倍
            height = max(height * 1.2, height + 60)
            rect_expanded = cv2.RotatedRect(r_center, (width, height), angle)
            # sz = rect_expanded.size
            box = rect_expanded.points()
            start_idx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - start_idx, 0)
            # (4, 1, 2)
            box = box.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(self.map_obstacle, [box], color=(0,))
        else:
            self.map_obstacle = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        num_obstacles = self.np_random.integers(*self.num_obstacles_range) if self.num_obstacles_range[1] > 0 else 0
        current_obstacle_num = 0
        while current_obstacle_num < num_obstacles:  # 生成障碍物
            o_x = self.np_random.uniform(0 + 100, self.dimensions[0] - 100)
            o_y = self.np_random.uniform(0 + 100, self.dimensions[1] - 100)
            if self.map_obstacle[int(o_y), int(o_x)]:
                continue
            o_len = self.np_random.uniform(*self.obstacle_size_range)
            o_wid = self.np_random.uniform(*self.obstacle_size_range)
            angle = self.np_random.uniform(0., 360.)
            pts = np.array(
                cv2.RotatedRect(center=(o_x, o_y), size=(o_len, o_wid), angle=angle).points(), dtype=np.int32
            ).reshape((-1, 1, 2))
            dist2player = cv2.pointPolygonTest(pts, self.agent.position, True)
            if dist2player < -2.0 * MowerAgent.length:
                current_obstacle_num += 1
                cv2.fillPoly(self.map_obstacle, [pts], color=(1.,))
                pts = np.array(
                    cv2.RotatedRect(center=(o_x, o_y), size=(o_len + 15, o_wid + 15), angle=angle).points(),  # 扩了15码
                    dtype=np.int32
                ).reshape((-1, 1, 2))
                cv2.fillPoly(self.map_frontier, [pts], color=(0.,))
        self.map_frontier_full = self.map_frontier  # map_frontier_full可用于获得地图的覆盖率
        self.map_weed = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        if self.noise_weed:
            self.map_weed_noisy = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        if isinstance(weed_num, float):  # 可以给比例
            weed_num = math.ceil(self.map_frontier.sum() * weed_num)
        self.weed_num = weed_num
        weed_count = 0
        while weed_count < weed_num:  # TODO，这个有时间的话就改成小簇然后发芽的方法
            if weed_dist == 'uniform':
                weed_x = self.np_random.integers(low=0, high=self.dimensions[0] - 1)
                weed_y = self.np_random.integers(low=0, high=self.dimensions[1] - 1)
                if self.map_frontier[weed_y, weed_x] and not self.map_weed[weed_y, weed_x]:
                    self.map_weed[weed_y, weed_x] = 1
                    weed_count += 1
                    if self.noise_weed:
                        delta_x = self.np_random.integers(low=-1, high=1)
                        delta_y = self.np_random.integers(low=-1, high=1)
                        self.map_weed_noisy[weed_y + delta_y, weed_x + delta_x] = 1
            else:
                weed_x = self.np_random.normal(loc=0., scale=0.35, size=weed_num - weed_count)
                weed_y = self.np_random.normal(loc=0., scale=0.35, size=weed_num - weed_count)
                weed_x = np.round((self.dimensions[1] / 2) * weed_x + self.dimensions[1] / 2).astype(np.int32)
                weed_x = np.clip(weed_x, 0, self.dimensions[0] - 1, dtype=np.int32)
                weed_y = np.round((self.dimensions[1] / 2) * weed_y + self.dimensions[1] / 2).astype(np.int32)
                weed_y = np.clip(weed_y, 0, self.dimensions[1] - 1, dtype=np.int32)
                for i in range(weed_num - weed_count):
                    if self.map_frontier[weed_y[i], weed_x[i]] and not self.map_weed[weed_y[i], weed_x[i]]:
                        self.map_weed[weed_y[i], weed_x[i]] = 1
                        weed_count += 1
                        if self.noise_weed:
                            delta_x = self.np_random.integers(low=-1, high=1)
                            delta_y = self.np_random.integers(low=-1, high=1)
                            self.map_weed_noisy[weed_y[i] + delta_y, weed_x[i] + delta_x] = 1
        cv2.fillPoly(self.map_weed, [self.agent.convex_hull.round().astype(np.int32)], color=(0.,))  # 小车位置boudingbox
        if self.render_covered_weed:
            self.map_weed_ori = self.map_weed.copy()
            # self.map_weed_ori = np.ones((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        # map_frontier和map_mist都要进行一次step的计算
        cv2.ellipse(img=self.map_frontier,
                    center=self.agent.position_discrete,
                    axes=(self.vision_length, self.vision_length),
                    angle=self.agent.direction,
                    startAngle=-self.vision_angle / 2,
                    endAngle=self.vision_angle / 2,
                    color=(0.,),
                    thickness=-1, )
        cv2.ellipse(img=self.map_mist,
                    center=self.agent.position_discrete,
                    axes=(self.vision_length + 1, self.vision_length + 1),  # vision_lenth+1就是迷雾探开范围
                    angle=self.agent.direction,
                    startAngle=-self.vision_angle / 2,
                    endAngle=self.vision_angle / 2,
                    color=(1.,),
                    thickness=-1, )
        if not np.logical_and(self.map_frontier, self.map_mist).any():  # 最开始看不见，开个口看见
            dist2player = cv2.pointPolygonTest(contours[0], self.agent.position, True)
            cv2.circle(img=self.map_mist,
                       center=self.agent.position_discrete,
                       radius=math.ceil(abs(dist2player) + 5),
                       color=(1.,),
                       thickness=-1, )
        # Get observation
        self.weed_num_t = self.map_weed.sum(dtype=np.int32)
        self.frontier_area_t = self.map_frontier.sum(dtype=np.int32)
        self.frontier_tv_t = total_variation(self.map_frontier.astype(np.int32))
        obs = self.observation()
        self.t = 1
        self.steer_t = 0.
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
            if self.state_pixels:
                self.screen = pygame.Surface(
                    (self.state_size[0] * self.render_repeat_times,
                     self.state_size[1] * self.render_repeat_times)
                )
            else:
                self.screen = pygame.Surface(
                    (self.dimensions[0] * self.render_repeat_times,
                     self.dimensions[1] * self.render_repeat_times)
                )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state_pixels:
            img = self.render_self()
        else:
            img = self.render_map()
        # 尝试用cv2和scipy.ndimage插值替换repeat，但是运行速度降低较多，或者插值后图像模糊有锯齿，故保留repeat操作
        img = img.repeat(self.render_repeat_times, axis=0).repeat(self.render_repeat_times, axis=1)  # 按比例缩放render的img
        surf = pygame.surfarray.make_surface(img)
        self.screen.blit(surf, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )  # pygame数据 (width, height, channels) -> numpy数据 (height, width, channels)

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
