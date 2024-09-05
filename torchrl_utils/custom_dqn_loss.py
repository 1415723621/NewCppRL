from __future__ import annotations

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from torchrl.objectives import DQNLoss
from torchrl.objectives.utils import (
    default_value_kwargs,
    ValueEstimators,
)
from torchrl.objectives.value.advantages import TD0Estimator
from torchrl.objectives.value.functional import td0_return_estimate


def value_rescale(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def value_rescale_inv(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    return torch.sign(x) * (
            ((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )


class CustomDQNLoss(DQNLoss):
    def make_value_estimator(self, value_type: ValueEstimators = None, value_rescale_eps: float = 1e-2, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator # 默认的是TD0
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type)) # 设置默认的gama和differentiable参数
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = CustomTD0Estimator(**hp,
                                                       value_network=self.value_network,
                                                       value_rescale_eps=value_rescale_eps)
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
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


class CustomTD0Estimator(TD0Estimator):

    def __init__(
            self,
            *,
            gamma: float | torch.Tensor,
            value_network: TensorDictModule,
            shifted: bool = False,
            average_rewards: bool = False,
            differentiable: bool = False,
            advantage_key: NestedKey = None,
            value_target_key: NestedKey = None,
            value_key: NestedKey = None,
            skip_existing: bool | None = None,
            device: torch.device | None = None,
            value_rescale_eps: float = 1e-2,
    ):
        super().__init__(
            gamma=gamma,
            value_network=value_network,
            shifted=shifted,
            average_rewards=average_rewards,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
            device=device,
        )
        self.value_rescale_eps = value_rescale_eps

    def value_estimate(
            self,
            tensordict,
            target_params: TensorDictBase | None = None,
            next_value: torch.Tensor | None = None,
            **kwargs,
    ):
        """
        对值进行放缩后再进行预测
        """
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        gamma = self.gamma.to(device)
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-5)
            tensordict.set(
                ("next", self.tensor_keys.reward), reward
            )  # we must update the rewards if they are used later in the code
        if next_value is None:
            next_value = self._next_value(tensordict, target_params, kwargs=kwargs)

        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        value_target = td0_return_estimate(
            gamma=gamma,
            next_state_value=value_rescale_inv(next_value, eps=self.value_rescale_eps),
            reward=reward,
            done=done,
            terminated=terminated,
        )
        return value_rescale(value_target, eps=self.value_rescale_eps)
