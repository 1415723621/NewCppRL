import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import trange

map_dir = 'envs/maps/1-600'

base_dir = Path(__file__).parent.parent.parent
map_dir = base_dir / map_dir
map_names = sorted(os.listdir(map_dir))
aim_size = cv2.imread(str(map_dir / map_names[0])).shape[0]


aim_size_half = aim_size // 2
aim_size_diag = int(aim_size / (2 ** 0.5))
aim_size_diag_half = int(aim_size / (2 ** 0.5)) // 2
aim_begin = aim_size_half - aim_size_diag_half
aim_end = aim_begin + aim_size_diag

x_min = aim_size
y_min = aim_size
x_max = 0
y_max = 0

for map_id in trange(len(map_names)):
    if map_names[map_id][0] == '_':
        continue
    map_weed: np.ndarray = (cv2.imread(str(map_dir / map_names[map_id])).sum(axis=-1) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(map_weed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    start_idx = box.sum(axis=1).argmin()
    box = np.roll(box, 4 - start_idx, 0)
    for pt in box:
        x_min = min(x_min, pt[0])
        y_min = min(y_min, pt[1])
        x_max = max(x_max, pt[0])
        y_max = max(y_max, pt[1])
    box = box.reshape((-1, 1, 2)).astype(np.int32)
    # map_lines = np.ones((aim_size, aim_size, 3), dtype=np.uint8)
    # cv2.polylines(map_lines, [box], True, (0, 255, 0), 1)
    # cv2.imwrite(str(map_dir / ('_' + map_names[map_id])), map_lines)
print(x_min, y_min)
print(aim_size - x_max, aim_size - y_max)
