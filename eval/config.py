import os
import numpy as np


class Config:
    # 环境参数
    W = 600
    H = 600
    ENV_NAME = "Pasture"
    RENDER_MODE = 'rgb_array'
    ACTION_TYPE = "continuous"
    WEED_COUNT = 50
    GAUSSIAN_WEED = True
    RETURN_MAP = True
    NUM_OBSTACLE_MIN = 0
    NUM_OBSTACLE_MAX = 0
    FARM_VERTICES = np.array([
        [30, 30],
        [370, 30],
        [370, 370],
        [70, 370]
    ])

    # 小车参数
    CAR_WIDTH = 4
    SIGHT_WIDTH = 35
    SIGHT_LENGTH = 50
    TURNING_RADIUS = 7

    # 路径设置
    DATA_DIR = 'path/to/data'
    MODEL_SAVE_PATH = 'path/to/save/model'
    LOG_DIR = 'path/to/logs'

    # 其他配置项
    # TASK_TYPE = "JUMP"
    # TASK_TYPE = "SNAKE"
    # TASK_TYPE = "R_SNAKE"
    TASK_TYPE = "REACT"  # JUMP SNAKE BCP R_SNAKE REACT # REACT不应该设置init_x y
    # TASK_TYPE = "BCP"

    RANDOM_SEED = 42
    DEBUG_MODE = True
