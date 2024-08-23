import time
from collections import deque
from queue import Queue

import numpy as np
import torch


def value_rescale(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def value_rescale_inv(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    return torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)


data = torch.normal(0, 100, size=(100,))
print(data.sum())
data_h = value_rescale(data)
print(data_h.sum())
data_h_ = value_rescale_inv(data_h)
print(data_h_.sum())
data_delta = torch.abs(data - data_h_)
print(data_delta.sum())
print(data_delta.mean())
# data = torch.normal(0, 1, size=[8, 5])
# # data_index = torch.nn.functional.one_hot(data.argmax(dim=1), 5)
# # data_selected = (data * data_index).sum(dim=1)
# data_index = data.argmax(dim=1, keepdim=True)
# data_selected = data.gather(dim=1, index=data_index)
# torch.nn.functional.huber_loss()
# print(data.shape)
# print(data_index.shape)
# print(data_selected.shape)
# print(data)
# print(data_selected)
#
# map_obstacle = np.zeros([256, 256])
# map_goal = np.zeros([256, 256])
# map_dist = np.zeros([256, 256])
#
# map_obstacle[20:40, 20:40] = 1
# map_goal[5, 5] = 1
# queue = deque()
# queue.append((200, 200, 0))
# directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
# tic = time.time()
#
# while len(queue) > 0:
#     x, y, dist = queue.popleft()
#     if map_obstacle[y, x] or map_dist[y, x]:
#         continue
#     map_dist[y, x] = dist + 1
#     # print(len(queue))
#     for dx, dy in directions:
#         nx, ny, ndist = x + dx, y + dy, dist + 1
#         if 0 <= nx < 256 and 0 <= ny < 256:
#             queue.append((nx, ny, ndist))
# toc = time.time()
# print(f'cost: {toc - tic}ms')
