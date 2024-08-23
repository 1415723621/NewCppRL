# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from copy import copy
from typing import Optional, Sequence

import torch
from tensordict import NonTensorData, TensorDictBase, MemoryMappedTensor
from tensordict.utils import NestedKey
from torchrl.envs.transforms import ObservationTransform
from torchrl.record.loggers import Logger

_has_tv = importlib.util.find_spec("torchvision", None) is not None


class CustomVideoRecorder(ObservationTransform):
    def __init__(
            self,
            logger: Logger,
            tag: str,
            device: Optional[torch.device] = None,
            max_len: int = 128,
            use_memmap: bool = True,
            nrow: int = None,
            in_keys: Optional[Sequence[NestedKey]] = None,
            skip: int | None = None,
            center_crop: Optional[int] = None,
            make_grid: bool | None = None,
            out_keys: Optional[Sequence[NestedKey]] = None,
            **kwargs,
    ) -> None:
        if in_keys is None:
            in_keys = ["pixels"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        video_kwargs = {"fps": 6}
        video_kwargs.update(kwargs)
        self.nrow = nrow
        self.video_kwargs = video_kwargs
        self.iter = 0
        self.skip = skip
        self.logger = logger
        self.tag = tag
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

    @property
    def make_grid(self):
        make_grid = self._make_grid
        if make_grid is None:
            if self.parent is not None:
                self._make_grid = True
                return True
            self._make_grid = False
            return False
        return make_grid

    @make_grid.setter
    def make_grid(self, value):
        self._make_grid = value

    @property
    def skip(self):
        skip = self._skip
        if skip is None:
            if self.parent is not None:
                self._skip = 2
                return 2
            self._skip = 1
            return 1
        return skip

    @skip.setter
    def skip(self, value):
        self._skip = value

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        if isinstance(observation, NonTensorData):
            observation_trsf = torch.tensor(observation.data)
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

            if observation.ndimension() == 2:
                observation_trsf = observation.unsqueeze(-3)
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

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def dump(self, suffix: Optional[str] = None) -> None:
        """Writes the video to the ``self.logger`` attribute.

        Calling ``dump`` when no image has been stored in a no-op.

        Args:
            suffix (str, optional): a suffix for the video to be recorded
        """
        if self.idx > 0:
            if suffix is None:
                tag = self.tag
            else:
                tag = "_".join([self.tag, suffix])
            if self.logger is not None:
                self.logger.log_video(
                    name=tag,
                    video=self.obs[:, :self.idx],
                    step=self.iter,
                    **self.video_kwargs,
                )
        self.iter += 1
        self.count = 0
        self.idx = 0

    def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self._call(tensordict_reset)
        return tensordict_reset
