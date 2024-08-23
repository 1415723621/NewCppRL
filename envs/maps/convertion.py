import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import trange

aim_size = 400
map_dir = 'envs/maps/1'
output_dir = f'envs/maps/1-{aim_size}'

base_dir = Path(__file__).parent.parent.parent
map_dir = base_dir / map_dir
map_names = sorted(os.listdir(map_dir))
if not Path(base_dir / output_dir).exists():
    os.mkdir(str(base_dir / output_dir))

aim_size_half = aim_size // 2
aim_size_diag = int((aim_size) / (2 ** 0.5) * 1.2)
aim_size_diag_half = aim_size_diag // 2
aim_begin = aim_size_half - aim_size_diag_half
aim_end = aim_begin + aim_size_diag

for map_id in trange(len(map_names)):
    map_weed: np.ndarray = cv2.imread(str(map_dir / map_names[map_id])).sum(axis=-1) > 0
    dimensions = map_weed.shape
    x_min, x_max, y_min, y_max = dimensions[0], 0, dimensions[1], 0
    for i in range(dimensions[0]):
        x_any = False
        for j in range(dimensions[1]):
            if map_weed[i, j]:
                x_any = True
                y_min = min(y_min, j)
                y_max = max(y_max, j)
        if x_any:
            x_min = min(x_min, i)
            x_max = max(x_max, i)
    x_len = x_max - x_min
    y_len = y_max - y_min
    max_len = max(x_len, y_len)
    x_begin = (max_len - x_len) // 2
    x_end = x_begin + x_len
    y_begin = (max_len - y_len) // 2
    y_end = y_begin + y_len
    map_weed_cropped = np.zeros((max_len, max_len))
    map_weed_cropped[x_begin:x_end, y_begin:y_end] = map_weed[x_min:x_max, y_min:y_max]
    map_weed_resize = cv2.resize(map_weed_cropped, (aim_size_diag, aim_size_diag))
    map_weed_full = np.zeros((aim_size, aim_size), dtype=np.uint8)
    map_weed_full[aim_begin:aim_end, aim_begin:aim_end] = map_weed_resize
    contours, _ = cv2.findContours(map_weed_full, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    x_min = box[:, 0].min()
    x_max = box[:, 0].max()
    y_min = box[:, 1].min()
    y_max = box[:, 1].max()
    # print(x_min, x_max, y_min, y_max)
    if 20 < x_min and x_max < aim_size - 20 and 20 < y_min and y_max < aim_size - 20:
        cv2.imwrite(str(base_dir / output_dir / map_names[map_id]), map_weed_full)
