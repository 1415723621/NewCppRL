from pathlib import Path
from typing import Any

import numpy
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torchrl.envs import ExplorationType, set_exploration_type

from torchrl_utils.custom_evaluator import CustomEvaluator


class SacEvaluator(CustomEvaluator):
    algo_name = 'sac_cont'
    base_dir = Path(__file__).parent.parent.parent
    env_cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))

    def get_actions(self,
                    actor: torch.nn.Module,
                    obss: list[Any]) -> list[int]:
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            observation = []
            vector = []
            for obs in obss:
                if isinstance(obs, dict):
                    observation.append(obs['observation'])
                    vector.append([obs['vector']])
            observation = torch.from_numpy(np.stack(observation, axis=0)).float().to(self.device)
            vector = torch.tensor(numpy.array(vector)).float().to(self.device)
            actions = actor(observation=observation, vector=vector)[2].tolist()
        return actions

    def get_actor(self,
                  pt_path: str) -> torch.nn.Module:
        model = torch.load(pt_path).to(self.device)
        actor = model[0]
        return actor


if __name__ == '__main__':
    evaluator = SacEvaluator(
        episodes=4,
        max_frames=4,
        max_step=1500,
        skip_frames=30,
        video=True,
        device='cpu',
        start_idx=0,
        ckpt_path=None,
        # ckpt_path='../../ckpt/dqn/240901_051858_test',
    )
    evaluator.run()
