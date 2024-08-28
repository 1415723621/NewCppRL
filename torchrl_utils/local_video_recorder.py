# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from typing import Optional

import torch
from tensordict import MemoryMappedTensor

_has_tv = importlib.util.find_spec("torchvision", None) is not None


class LocalVideoRecorder:
    def __init__(
            self,
            device: Optional[torch.device] = None,
            max_len: int = 128,
            use_memmap: bool = True,
            nrow: int = None,
            skip: int | None = None,
            center_crop: Optional[int] = None,
            make_grid: bool | None = None,
            fps: int | None = None,
    ) -> None:
        if make_grid:
            assert nrow is not None and nrow > 0
        self.nrow = nrow
        self.fps = 6 if fps is None else fps
        self.iter = 0
        self.skip = 1 if skip is None else skip
        self.count = 0
        self.idx = 0
        self.center_crop = center_crop
        self.make_grid = make_grid
        if center_crop and not _has_tv:
            raise ImportError(
                "Could not load center_crop from torchvision. Make sure torchvision is installed."
            )
        # self.obs = []
        self.use_memmap = use_memmap
        self.max_len = max_len
        self.obs = None
        self.device = device

    def apply(self, observation: torch.Tensor) -> torch.Tensor:
        if not isinstance(observation, torch.Tensor):
            observation_trsf = torch.tensor(observation)
        else:
            observation_trsf = observation
        self.count += 1
        if self.count % self.skip == 0:
            if (
                    observation_trsf.ndim >= 3
                    and observation_trsf.shape[-3] == 3
                    and observation_trsf.shape[-2] > 3
                    and observation_trsf.shape[-1] > 3
            ):
                # permute the channels to the last dim
                observation_trsf = observation_trsf.permute(
                    *range(observation_trsf.ndim - 3), -2, -1, -3
                )
            if not (
                    observation_trsf.shape[-1] == 3 or observation_trsf.ndimension() == 2
            ):
                raise RuntimeError(
                    f"Invalid observation shape, got: {observation.shape}"
                )
            observation_trsf = observation_trsf.clone()

            if observation_trsf.ndimension() == 2:
                observation_trsf = observation_trsf.unsqueeze(-3)
            else:
                if observation_trsf.shape[-1] != 3:
                    raise RuntimeError(
                        "observation_trsf is expected to have 3 dimensions, "
                        f"got {observation_trsf.ndimension()} instead"
                    )
                trailing_dim = range(observation_trsf.ndimension() - 3)
                observation_trsf = observation_trsf.permute(*trailing_dim, -1, -3, -2)
            if self.center_crop:
                if not _has_tv:
                    raise ImportError(
                        "Could not import torchvision, `center_crop` not available."
                        "Make sure torchvision is installed in your environment."
                    )
                from torchvision.transforms.functional import (
                    center_crop as center_crop_fn,
                )

                observation_trsf = center_crop_fn(
                    observation_trsf, [self.center_crop, self.center_crop]
                )
            if self.make_grid and observation_trsf.ndimension() >= 4:
                if not _has_tv:
                    raise ImportError(
                        "Could not import torchvision, `make_grid` not available."
                        "Make sure torchvision is installed in your environment."
                    )
                from torchvision.utils import make_grid

                observation_trsf = make_grid(observation_trsf.flatten(0, -4), nrow=self.nrow)
                self.add(observation_trsf.to(torch.uint8))
            elif observation_trsf.ndimension() >= 4:
                self.add(observation_trsf.to(torch.uint8).flatten(0, -4))
            else:
                self.add(observation_trsf.to(torch.uint8))
        del observation_trsf
        return observation

    def obs_expanded(self, obs_old):
        expaned_len = 2 * self.max_len
        if self.use_memmap:
            obs_new = MemoryMappedTensor.zeros((1, expaned_len, *self.obs_shape), dtype=torch.uint8)
        else:
            obs_new = torch.zeros((1, expaned_len, *self.obs_shape), device=self.device, dtype=torch.uint8)
        obs_new[:, :self.max_len] = obs_old
        del obs_old
        self.max_len = expaned_len
        return obs_new

    def add(self, observation_trsf: torch.Tensor):
        if self.obs is None:
            self.obs_shape = observation_trsf.shape[-3:]
            if self.use_memmap:
                self.obs = MemoryMappedTensor.zeros((1, self.max_len, *self.obs_shape), dtype=torch.uint8)
            else:
                self.obs = torch.zeros((1, self.max_len, *self.obs_shape), device=self.device,
                                       dtype=torch.uint8)
        if observation_trsf.ndimension() < 4:
            num_obs = 1
        elif observation_trsf.ndimension() == 4:
            num_obs = observation_trsf.size(0)
        else:
            raise ValueError("Observation dimension is greater than 4")
        while self.idx + num_obs > self.max_len:
            self.obs = self.obs_expanded(self.obs)
        self.obs[:, self.idx:(self.idx + num_obs)] = observation_trsf
        self.idx += num_obs

    def dump(self, filepath: Optional[str] = None) -> None | torch.Tensor:
        vid_tensor = None
        if self.idx > 0:
            vid_tensor = self.obs[:, :self.idx]
            if filepath is not None:
                import torchvision
                if vid_tensor.shape[-3] not in (3, 1):
                    raise RuntimeError(
                        "expected the video tensor to be of format [T, C, H, W] but the third channel "
                        f"starting from the end isn't in (1, 3) but is {vid_tensor.shape[-3]}."
                    )
                if vid_tensor.ndim > 4:
                    vid_tensor = vid_tensor.flatten(0, vid_tensor.ndim - 4)
                vid_tensor = vid_tensor.permute((0, 2, 3, 1))
                vid_tensor = vid_tensor.expand(*vid_tensor.shape[:-1], 3)
                torchvision.io.write_video(filepath, vid_tensor, fps=self.fps)
        self.iter += 1
        self.count = 0
        self.idx = 0
        return vid_tensor
