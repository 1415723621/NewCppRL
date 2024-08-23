import torch.nn
import torch.optim

from torchrl_utils.custom_video_recorder import CustomVideoRecorder


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, eval_steps=1_000):
    td_test = test_env.rollout(
        policy=actor,
        auto_reset=True,
        auto_cast_to_device=True,
        break_when_any_done=False,
        max_steps=eval_steps,
    )
    test_env.apply(dump_video)
    return td_test


def dump_video(module):
    if isinstance(module, CustomVideoRecorder):
        module.dump()
