import time

import torch
from torch import tensor
import numpy as np
import math
from torch_geometric.datasets import Reddit, PPI
from operator import itemgetter
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.loader import ClusterData, ClusterLoader

# dataset = PPI(root='../data/ppi/')

# cluster_data = ClusterData(dataset, num_parts=1500, recursive=False,
#                            save_dir='../data/reddit/')
# train_loader = ClusterLoader(cluster_data, batch_size=5, shuffle=True)
# edge_sou = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5])
# edge_des = tensor([1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 1, 3, 0, 1, 2, 0, 1, 0, 1])
# edge_data = torch.stack((edge_sou, edge_des), 0)


# calculate the degree of each node
def get_degree(edge):
    edge_list = edge[0].numpy().tolist()
    dic = {}
    for key in edge_list:
        dic[key] = dic.get(key, 0) + 1
    return dic


# get the attribution of edge
def get_attribution(edge):
    num_neighbors = []
    vertex_index = []
    num_neighbors.append(0)
    count = 1
    current = edge[0][0]
    vertex_index.append(current.item())
    for i in edge[0][1:]:
        if i == current:
            count += 1
        else:
            num_neighbors.append(count)
            current = i
            vertex_index.append(current.item())
            count += 1
    num_neighbors.append(count)
    num_v = len(num_neighbors) - 1
    return num_neighbors, vertex_index, num_v


# get the redundancy of two nodes
def redundancy(edge):
    num_neighbors, vertex_index, num_v = get_attribution(edge)
    pair_redundancy = dict()
    for v in range(num_v):
        neigh = edge[1][num_neighbors[v]:num_neighbors[v + 1]].tolist()

        # traverse the current vertex's neighbors, if two neighbors have an edge then they are a pair can be aggregated
        for iu, u in enumerate(neigh):
            for w in neigh[iu + 1:]:
                if (u, w) not in pair_redundancy:
                    pair_redundancy[(u, w)] = 1
                else:
                    pair_redundancy[(u, w)] += 1

    # exact the edges with their weights, which bigger than 1
    heap = {k: v for k, v in pair_redundancy.items() if v > 2}
    # reorder the weight for aggregation selection then select biggest pair in the heap
    heap_sorted = sorted(heap.items(), key=itemgetter(1, 0), reverse=True)

    return heap_sorted


# calculate the contribution of all nodes to max_heap
def get_contribution_2(edge, degrees, vertex):
    contribution_list = []
    for key in vertex:
        root_degree = degrees.get(key, 0)
        root_degree2 = math.pow(root_degree, 2)
        degree_mul = 1
        neighbors = edge[1][np.where(edge[0] == key)[0]].numpy().tolist()
        for neighbor in neighbors:
            degree = degrees.get(neighbor, 0)
            degree_mul *= degree
        # print(root_degree3, degree2_sum)
        final = math.log(degree_mul, math.e)
        contribution = final/root_degree2
        contribution_list.append(contribution)
    return contribution_list


def get_contribution(edge, degrees, vertex):
    contribution_list = []
    for key in vertex:
        root_degree = degrees.get(key, 0)
        root_degree2 = math.pow(root_degree, 2)
        degree2_sum = 0
        neighbors = edge[1][np.where(edge[0] == key)[0]].numpy().tolist()
        for neighbor in neighbors:
            degree = degrees.get(neighbor, 0)
            degree2 = math.pow(degree, 2)
            degree2_sum += degree2
        # print(root_degree3, degree2_sum)
        influence = root_degree2/degree2_sum
        contribution = 1/influence
        contribution_list.append(contribution)
    return contribution_list


# get the rest of edge after filtering with threshold
def get_rest_edge(edge, vertex_index, contribution_list, threshold):

    indices = np.where(np.array(contribution_list) >= threshold)[0]
    rest_source_nodes = np.array(vertex_index)[indices]
    rest_source_tensor = edge[0][np.isin(edge[0], rest_source_nodes)]
    rest_des_tensor = edge[1][np.isin(edge[0], rest_source_nodes)]
    rest_tensor = torch.stack((rest_source_tensor, rest_des_tensor), 0)

    return rest_tensor

# i = 0
# for data in dataset:
#     print(data)
#     # set parameters
#     threshold = 100
#     overlap_boundary = int(data.x.shape[0] / 4)
#
#
#     degree_dic = get_degree(data.edge_index)
#     num_neighbors, vertex_index, num_v = get_attribution(data.edge_index)
#     contribution_list = get_contribution(data.edge_index, degree_dic, vertex_index)
#     rest_tensor = get_rest_edge(data.edge_index, vertex_index, contribution_list, threshold)
#
#     t1 = time.time()
#     pair_redundancy = redundancy(data.edge_index)
#     t2 = time.time()
#     topk_pair_redundancy = pair_redundancy[:overlap_boundary]
#     print("time before reduction: ", t2-t1)
#
#     t3 = time.time()
#     pair_redundancy_after = redundancy(rest_tensor)
#     t4 = time.time()
#     topk_pair_redundancy_after = pair_redundancy_after[:overlap_boundary]
#     print("time after reduction: ", t4-t3)
#
#     pair_list = []
#     for pair in topk_pair_redundancy:
#         pair_list.append(pair[0])
#
#     pair_after_list = []
#     for pair in topk_pair_redundancy_after:
#         pair_after_list.append(pair[0])
#
#     count = 0
#     for pair in pair_after_list:
#         if pair in pair_list:
#             count += 1
#     print("acc: ", count/overlap_boundary)
#     i += 1
#     if i == 5:
#         break
