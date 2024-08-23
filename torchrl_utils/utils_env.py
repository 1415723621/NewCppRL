from pathlib import Path

import yaml
from omegaconf import DictConfig
from torchrl.envs import RewardSum, StepCounter, TransformedEnv, ParallelEnv
import gymnasium as gym
from torchrl.envs.libs.gym import GymWrapper

# ====================================================================
# Environment utils
# --------------------------------------------------------------------

cfg = DictConfig(yaml.load(open(f'{Path(__file__).parent.parent}/configs/env_config.yaml'), Loader=yaml.FullLoader))


def make_env_lambda(
        device="cpu",
        from_pixels=False,
):
    env = gym.make(
        render_mode='rgb_array' if from_pixels else None,
        # save_pixels=from_pixels,
        **cfg.env.params,
    )
    env = GymWrapper(
        env,
        device=device,
        from_pixels=from_pixels,
        pixels_only=False,
    )
    return env


def make_env(
        num_envs=1,
        device="cpu",
        from_pixels=False,
):
    if num_envs == 1:
        env = make_env_lambda(
            device=device,
            from_pixels=from_pixels,
        )
    else:
        env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=lambda: make_env_lambda(
                device=device,
                from_pixels=from_pixels,
            ),
        )
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env
