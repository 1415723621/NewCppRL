# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey
from torch import nn
from torchrl.data.tensor_specs import TensorSpec

from torchrl.data.utils import _find_action_space

from torchrl.envs.utils import step_mdp
from torchrl.modules.tensordict_module.actors import (
    DistributionalQValueActor,
    QValueActor,
)
from torchrl.modules.tensordict_module.common import ensure_tensordict_compatible

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import TDLambdaEstimator
from torchrl.objectives.value.advantages import TD0Estimator, TD1Estimator
from torchrl.objectives import DQNLoss


def value_rescale(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def value_rescale_inv(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    return torch.sign(x) * (
            ((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )


class CustomDQNLoss(DQNLoss):
    def __init__(
            self,
            value_network: Union[QValueActor, nn.Module],
            *,
            loss_function: Optional[str] = "l2",
            delay_value: bool = True,
            double_dqn: bool = False,
            gamma: float = None,
            action_space: Union[str, TensorSpec] = None,
            priority_key: str = None,
            reduction: str = None,
            value_rescale_eps: float = 1e-2,
    ) -> None:
        super().__init__(value_network=value_network,
                         loss_function=loss_function,
                         delay_value=delay_value,
                         double_dqn=double_dqn,
                         gamma=gamma,
                         action_space=action_space,
                         priority_key=priority_key,
                         reduction=reduction)
        self.value_rescale_eps = value_rescale_eps


    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(**hp, value_network=self.value_network)
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(**hp, value_network=self.value_network)
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp, value_network=self.value_network
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value_target": self.tensor_keys.value_target,
            "value": self.tensor_keys.value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the DQN loss given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network (observations, "done", "terminated", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the DQN loss.

        """
        td_copy = tensordict.clone(False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(self.tensor_keys.action_value)

        if self.action_space == "categorical":
            if action.ndim != pred_val.ndim:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            pred_val_index = torch.gather(pred_val, -1, index=action).squeeze(-1)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1)

        if self.double_dqn:
            step_td = step_mdp(td_copy, keep_other=False)
            step_td_copy = step_td.clone(False)
            # Use online network to compute the action
            with self.value_network_params.data.to_module(self.value_network):
                self.value_network(step_td)
                next_action = step_td.get(self.tensor_keys.action)

            # Use target network to compute the values
            with self.target_value_network_params.to_module(self.value_network):
                self.value_network(step_td_copy)
                next_pred_val = step_td_copy.get(self.tensor_keys.action_value)

            if self.action_space == "categorical":
                if next_action.ndim != next_pred_val.ndim:
                    # unsqueeze the action if it lacks on trailing singleton dim
                    next_action = next_action.unsqueeze(-1)
                next_value = torch.gather(next_pred_val, -1, index=next_action)
            else:
                next_value = (next_pred_val * next_action).sum(-1, keepdim=True)
        else:
            next_value = None
        target_value = self.value_estimator.value_estimate(
            td_copy,
            target_params=self.target_value_network_params,
            next_value=next_value,
        ).squeeze(-1)

        with torch.no_grad():
            priority_tensor = (pred_val_index - target_value).pow(2)
            priority_tensor = priority_tensor.unsqueeze(-1)
        if tensordict.device is not None:
            priority_tensor = priority_tensor.to(tensordict.device)

        tensordict.set(
            self.tensor_keys.priority,
            priority_tensor,
            inplace=True,
        )
        loss = distance_loss(pred_val_index, target_value, self.loss_function)
        loss = _reduce(loss, reduction=self.reduction)
        td_out = TensorDict({"loss": loss}, [])
        return td_out
