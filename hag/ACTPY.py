import ctypes

import torch
import operator
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Reddit, PPI
from torch_geometric.loader import DataLoader

# torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None)
import time
from threshold import get_attribution, get_contribution, get_contribution_2, get_rest_edge, get_degree

from operator import itemgetter


# get the redundancy of two nodes
def redundancy(edge):
    num_neighbors = []
    vertex_index = []
    num_neighbors.append(0)
    count = 1
    current = edge[0][0]
    vertex_index.append(current.item())
    for i in edge[0][1:]:
        if i == current:
            count += 1
        elif i != current and i != (current + 1):
            num_neighbors.append(count)
            for j in range(current + 1, i):
                num_neighbors.append(count)
                vertex_index.append(j)
                current += 1
            current += 1
            vertex_index.append(current.item())
            count += 1
        else:
            num_neighbors.append(count)
            current = i
            vertex_index.append(current.item())
            count += 1
    num_neighbors.append(count)
    num_v = len(num_neighbors) - 1
    # vertex_index = np.unique(data.edge_index[0].tolist())
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

    return pair_redundancy, num_neighbors, vertex_index


# return the max redundancy of pair of nodes
def argmax_redundancy(pair_redundancy, num_v, capacity):
    # record the edges of aggregated pairs of nodes, (u,v) containing the two nodes of the edge
    edge_to_aggr = []

    # exact the edges with their weights, which bigger than 1
    H = {k: v for k, v in pair_redundancy.items() if v > 2}
    number_of_va = 0

    # reorder the weight for aggregation selection then select biggest pair in the heap
    H_sorted = sorted(H.items(), key=itemgetter(1, 0), reverse=True)
    # print("H_sorted: ", H_sorted[:10])
    S = np.ones(num_v)
    cap = 0
    for (u, v), weight in H_sorted:
        if not (S[u] and S[v]):
            continue
        S[u] = 0
        S[v] = 0
        edge_to_aggr.append((u, v))
        cap += 1
        if len(edge_to_aggr) == int(num_v/2) or cap == capacity:
            break

    return edge_to_aggr


# construct the aggregated GNN_graph
def constrcut_gnn_graph(edge_to_aggr, num_v, num_neighbors, vertex_index, data):
    ret_feat = torch.vstack((torch.zeros(data.x.size()), torch.zeros((len(edge_to_aggr)+1, data.x.shape[1]), dtype=torch.int)))
    edge_index0 = data.edge_index[0]
    edge_index1 = data.edge_index[1]
    # print(ret_feat)
    ret_feat[:data.x.shape[0]] = data.x
    idx = 0
    total_agg_num = 0
    # prepare I_edges here, after identifying the large-weight edges
    i_edges = dict()
    t1 = time.time()
    for (aggr1, aggr2) in edge_to_aggr:
        index1 = vertex_index.index(aggr1)
        index2 = vertex_index.index(aggr2)
        neigh1 = data.edge_index[1][num_neighbors[index1]:num_neighbors[index1 + 1]]
        neigh2 = data.edge_index[1][num_neighbors[index2]:num_neighbors[index2 + 1]]
        # find the nodes whose neighbors containing (aggr1, aggr2)
        i_edges[(aggr1, aggr2)] = np.intersect1d(neigh1, neigh2, assume_unique=True)
    t2 = time.time()
    # print("intersection time: ", t2 - t1)
    loc_time = 0
    aggregated_node_set = []
    for (aggr1, aggr2) in edge_to_aggr:
        # add the new aggregated node, compute the sum of pair of nodes
        v_root = i_edges[(aggr1, aggr2)]
        # ret_feat[num_v + idx] = ret_feat[aggr1] + ret_feat[aggr2]
        # construct the graph after aggregation
        for v in v_root:
            t1 = time.time()
            neigh = np.array(edge_index1[num_neighbors[v]:num_neighbors[v + 1]])
            i1 = np.where(neigh == aggr1)[0][0]
            i2 = np.where(neigh == aggr2)[0][0]
            t2 = time.time()
            loc_time += t2 - t1
            # insert the aggregated node on the first node
            edge_index1[num_neighbors[v] + i1] = num_v + idx
            edge_index1[num_neighbors[v] + i2] = -1
            total_agg_num += 1
        idx += 1
        aggregated_node_set.append((num_v + idx, (aggr1, aggr2)))
    # print("loc_time:{}".format(loc_time))
    # delete where index1 == -1
    final_edge0 = []
    final_edge1 = []
    for i, j in zip(edge_index0, edge_index1):
        if j == -1:
            continue
        final_edge0.append(i)
        final_edge1.append(j)
    data.x = ret_feat
    data.edge_index = torch.stack((torch.tensor(final_edge0), torch.tensor(final_edge1)), 0)
    return data, total_agg_num, aggregated_node_set


def findIndex(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx


def obtain_aggregated_gnn(data, capacity):
    zero_loc = np.where(data.edge_index[0] == 0)[0][0]
    c1 = data.edge_index[0][:zero_loc]
    c2 = data.edge_index[0][zero_loc:]
    # print(c1.size(), c2.size())
    data.edge_index[0] = torch.cat((c2, c1), 0)
    c3 = data.edge_index[1][:zero_loc]
    c4 = data.edge_index[1][zero_loc:]
    data.edge_index[1] = torch.cat((c4, c3), 0)

    t1 = time.time()
    degree_dic = get_degree(data.edge_index)
    num_neighbors, vertex_index, num_v = get_attribution(data.edge_index)
    contribution_list = get_contribution_2(data.edge_index, degree_dic, vertex_index)
    contribution_list.sort(reverse=True)
    # print(contribution_list[:int(1*len(contribution_list)/2)])
    rest_tensor = get_rest_edge(data.edge_index, vertex_index, contribution_list,
                                threshold=contribution_list[:int(1*len(contribution_list)/2)][-1])
    t2 = time.time()
    exact_time = t2 - t1
    print("exact pre-train time: ", exact_time)

    t1 = time.time()
    pair_redundancy, num_neighbors, vertex_index = redundancy(data.edge_index)
    t2 = time.time()
    redundancy_time1 = t2 - t1
    print("redundancy time1: ", redundancy_time1)
    num_neighbors_initial = num_neighbors.copy()
    data_initial = data.clone()
    t1 = time.time()
    pair_redundancy_after, num_neighbors_after, vertex_index_after = redundancy(rest_tensor)
    t2 = time.time()
    redundancy_time2 = t2 - t1
    print("redundancy time2: ", redundancy_time2)

    # print(num_neighbors)
    num_v = data.x.shape[0]
    t1 = time.time()
    edge_to_aggr1 = argmax_redundancy(pair_redundancy, num_v, capacity)
    t2 = time.time()
    max_time1 = t2 - t1
    print("max time1: ", max_time1)
    t1 = time.time()
    edge_to_aggr2 = argmax_redundancy(pair_redundancy_after, num_v, capacity)
    t2 = time.time()
    max_time2 = t2 - t1
    print("max time2: ", max_time2)

    # print(edge_to_aggr)
    t1 = time.time()
    aggregated_data1, total_agg_num1, aggregated_nodes_set1 = constrcut_gnn_graph(edge_to_aggr1, num_v, num_neighbors, vertex_index, data)
    t2 = time.time()
    construct_time1 = t2 - t1
    print("construct time1: ", construct_time1)

    t1 = time.time()
    aggregated_data2, total_agg_num2, aggregated_nodes_set2 = constrcut_gnn_graph(edge_to_aggr2, num_v, num_neighbors_initial, vertex_index, data_initial)
    t2 = time.time()
    construct_time2 = t2 - t1
    print("construct time2: ", construct_time2)

    count = 0
    for pair in edge_to_aggr2:
        if pair in edge_to_aggr1:
            count += 1
    print("acc: ", count/len(edge_to_aggr2))

    # print(aggregated_data.edge_index)
    print("Aggregate finish!")
    aggregate_acc = total_agg_num2/total_agg_num1
    print("Aggregation acc: ", aggregate_acc)
    total_time1 = redundancy_time1 + max_time1 + construct_time1
    total_time2 = redundancy_time2 + max_time2 + construct_time2
    print("total time compare:{}, {}" .format(total_time1, total_time2))
    return aggregated_data1, aggregated_data2, aggregated_nodes_set1, aggregated_nodes_set2, redundancy_time1, redundancy_time2, total_time1, total_time2, aggregate_acc



