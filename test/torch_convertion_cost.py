import time

import numpy as np
import torch
from gymnasium.experimental.wrappers.numpy_to_torch import numpy_to_torch

iters = 1000
size = (400, 400)
count = []
for i in range(iters):
    random_data = np.random.random(size=size)
    tic = time.time()
    converted_data = numpy_to_torch(random_data)
    toc = time.time()
    count.append(toc - tic)
print(f'Numpy_to_Torch {np.mean(count)}ms')
count = []
for i in range(iters):
    random_data = np.random.random(size=size)
    tic = time.time()
    converted_data = torch.from_numpy(random_data)
    toc = time.time()
    count.append(toc - tic)
print(f'Numpy_to_Torch {np.mean(count)}ms')
