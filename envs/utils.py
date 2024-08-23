from __future__ import annotations

import math

import numpy as np


def total_variation(images: np.ndarray) -> int:
    """https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/image_ops_impl.py#L3213-L3282"""
    pixel_dif1 = images[1:, :] - images[:-1, :]
    pixel_dif2 = images[:, 1:] - images[:, :-1]
    tot_var = np.abs(pixel_dif1).sum() + np.abs(pixel_dif2).sum()
    return tot_var


def total_variation_mat(images: np.ndarray) -> np.ndarray:
    mask_tv_cols = images[1:, :] - images[:-1, :] != 0
    mask_tv_cols = np.pad(mask_tv_cols, pad_width=[[0, 1], [0, 0]], mode='constant')
    mask_tv_rows = images[:, 1:] - images[:, :-1] != 0
    mask_tv_rows = np.pad(mask_tv_rows, pad_width=[[0, 0], [0, 1]], mode='constant')
    mask_tv = np.logical_or(mask_tv_rows, mask_tv_cols)
    return mask_tv


def get_map_pasture_larger(map_pasture: np.ndarray):
    map_pasture_larger = map_pasture
    map_pasture_larger = np.logical_or(
        map_pasture_larger,
        np.insert(
            map_pasture[1:, :],
            -1,
            False,
            axis=0
        )
    )
    map_pasture_larger = np.logical_or(
        map_pasture_larger,
        np.insert(
            map_pasture[:-1, :],
            0,
            False,
            axis=0
        )
    )
    map_pasture_larger = np.logical_or(
        map_pasture_larger,
        np.insert(
            map_pasture[:, 1:],
            -1,
            False,
            axis=1
        )
    )
    map_pasture_larger = np.logical_or(
        map_pasture_larger,
        np.insert(
            map_pasture[:, :-1],
            0,
            False,
            axis=1
        )
    )
    return map_pasture_larger


class NumericalRange:
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    @property
    def mode(self):
        return self.max - self.min


class MowerAgent:
    width = 5
    length = 8.5
    occupancy = math.hypot(width, length)
    lw_ratio = math.degrees(math.atan2(width / occupancy, length / occupancy))


    def __init__(
            self,
            position: tuple[float, float] = (None, None),
            direction: float = None,
    ):
        self.x, self.y = position
        self.direction = direction
        self.last_acc, self.last_steer = 0., 0.

    @property
    def position(self):
        return self.x, self.y

    @property
    def position_discrete(self):
        return round(self.x), round(self.y)

    @property
    def convex_hull(self):
        return np.array([
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 0 + self.lw_ratio)),
             self.y + 1.0 * self.width * math.sin(math.radians(self.direction + 0 + self.lw_ratio))),
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 180 - self.lw_ratio)),
             self.y + 1.0 * self.width * math.sin(math.radians(self.direction + 180 - self.lw_ratio))),
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 180 + self.lw_ratio)),
             self.y + 1.0 * self.width * math.sin(math.radians(self.direction + 180 + self.lw_ratio))),
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 0 - self.lw_ratio)),
             self.y + 1.0 * self.width * math.sin(math.radians(self.direction + 0 - self.lw_ratio))),
        ])

    def control(self, acc: float, steer: float):
        self.last_acc, self.last_steer = acc, steer
        self.direction = (self.direction + steer) % 360
        dx = acc * math.cos(math.radians(self.direction))
        dy = acc * math.sin(math.radians(self.direction))
        self.x += dx
        self.y += dy

    def reset(self, position: tuple[float, float], direction: float):
        self.x, self.y = position
        self.last_acc, self.last_steer = 0., 0.
        self.direction = direction
