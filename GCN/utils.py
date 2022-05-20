import time

import torch


edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6],
                           [2, 5, 3, 2, 3, 5, 0, 1, 4, 6, 0, 1, 6, 4, 2, 3, 5, 0, 1, 4, 6, 2, 3, 5]])
x = torch.tensor([[1, 0, 0, 0, 0, 0, 1],
     [0, 1, 0, 0, 0, 0, 1],
     [0, 0, 1, 0, 0, 0, 1],
     [0, 0, 0, 1, 0, 0, 1],
     [0, 0, 0, 0, 1, 0, 1],
     [0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 0, 1]])


# get neighbor list and num of vertex
def get_neighborlist(edge, x):
    num_neighbors = dict()
    idxs = []
    count = 1
    current = edge[0][0]
    idxs.append(current.item())
    start, end = 0, 0
    for i in edge[0][1:]:
        if i == current:
            count += 1
        else:
            end = count
            num_neighbors[current.item()] = (start, end)
            current = i
            idxs.append(current.item())
            count += 1
            start = end
    num_neighbors[current.item()] = (start, count)
    num_v = x.shape[0]
    return num_neighbors, num_v, idxs

num_neighbors, num_v, idxs = get_neighborlist(edge_index, x)
print(num_neighbors[-1])
