from torch import tensor
import torch
import operator
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Reddit, PPI, Yelp, AmazonProducts, Flickr
from torch_geometric.data import Data
from torch_scatter import segment_coo, segment_csr
from torch_geometric.loader import DataLoader
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.loader import ClusterData, ClusterLoader
import sys
sys.path.append(r'../hag')
sys.path.append(r'../GCN')
from gcn import GCN, GCNHAG, GraphSAGE, GraphSAGEHAG
from sklearn.metrics import f1_score, recall_score, precision_score
# torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None)
import time
from threshold import get_attribution, get_contribution, get_contribution_2, get_rest_edge, get_degree
from operator import itemgetter
torch.set_printoptions(threshold=np.inf)


# add pair count
def add_pair_count(u, v, H_sorted):
    if (u, v) not in H_sorted and (v, u) not in H_sorted:
        H_sorted[(u, v)] = 1
    else:
        if (u, v) in H_sorted:
            H_sorted[(u, v)] += 1
        else:
            H_sorted[(v, u)] += 1
    return H_sorted

# sub pair count
def sub_pair_count(u, v, H_sorted):
    if (u, v) in H_sorted:
        H_sorted[(u, v)] -= 1
    elif (v, u) in H_sorted:
        H_sorted[(v, u)] -= 1
    return H_sorted


# find item's idx in the array
def findIndex(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx[1]


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
    last_index = current.item()
    num_v = x.shape[0]
    return num_neighbors, num_v, idxs, last_index


# get the redundancy of two nodes
def get_redundancy_heap(edge, x):
    num_neighbors, num_v, idxs, last = get_neighborlist(edge, x)
    # vertex_index = np.unique(data.edge_index[0].tolist())
    pair_redundancy = dict()
    for v in idxs:
        neigh = edge[1][num_neighbors[v][0]: num_neighbors[v][1]].tolist()
        # traverse the current vertex's neighbors, if two neighbors have an edge then they are a pair can be aggregated
        for iu, u in enumerate(neigh):
            for w in neigh[iu + 1:]:
                temp1 = u
                temp2 = w
                if u > w:
                    temp1 = w
                    temp2 = u
                if (temp1, temp2) not in pair_redundancy:
                    pair_redundancy[(temp1, temp2)] = 1
                else:
                    pair_redundancy[(temp1, temp2)] += 1

    # exact the edges with their weights, which bigger than 1
    # H = {k: v for k, v in pair_redundancy.items() if v > 2}

    # reorder the weight for aggregation selection then select biggest pair in the heap
    # H_sorted = sorted(pair_redundancy.items(), key=itemgetter(1), reverse=True)

    return pair_redundancy, num_v, num_neighbors, idxs, last


# get the simplified computation graph
def transfer_graph(H_sorted, num_v, num_neighbors, idxs, last, data, maxDepth, maxWidth):
    edge_index0 = data.edge_index[0]
    edge_index1 = data.edge_index[1]
    x = data.x
    total_redundancy_eliminated = 0
    v = num_v
    print("number of v: ", num_v)
    depths = [None] * maxWidth
    ranges = []
    ranges_list = []
    total_process = 0
    initial_len = num_v
    actual_len = len(idxs)
    search_time = 0
    print("length of heap before contribution: ", len(H_sorted))
    while v < num_v + maxWidth:
        # H_sorted = dict(H_sorted)
        if not H_sorted:
            break
        t1 = time.time()
        top = max(H_sorted.values())
        t2 = time.time()
        search_time += (t2 - t1)
        for pair, value in H_sorted.items():
            if value == top:
                break
        aggr1 = min(pair[0], pair[1])
        aggr2 = max(pair[0], pair[1])
        redundancy_eliminated = H_sorted.pop(pair)
        # print("redundancy eliminated: ", redundancy_eliminated)
        if redundancy_eliminated < 2:
            break
        total_redundancy_eliminated += redundancy_eliminated
        preDepth = 0 if aggr1 < initial_len else depths[aggr1 - initial_len]
        if aggr2 >= initial_len:
            preDepth = max(preDepth, depths[aggr2 - initial_len])
        if preDepth >= maxDepth:
            continue
        depths[v - initial_len] = preDepth + 1
        for j in idxs:
            neigh = np.array(edge_index1[num_neighbors[j][0]: num_neighbors[j][1]])
            if aggr1 in neigh and aggr2 in neigh:
                i1 = np.where(neigh == aggr1)[0][0]
                i2 = np.where(neigh == aggr2)[0][0]
                edge_index1[num_neighbors[j][0] + i1] = v
                edge_index1[num_neighbors[j][0] + i2] = -1
                neigh = np.delete(neigh, [i1, i2])
                t3 = time.time()
                for k in neigh:
                    if k != -1:
                        sub_pair_count(k, aggr1, H_sorted)
                        sub_pair_count(k, aggr2, H_sorted)
                        add_pair_count(k, v, H_sorted)
                t4 = time.time()
                search_time += (t4 - t3)
        edge_index0 = torch.cat((edge_index0, tensor([v, v])), dim=0)
        edge_index1 = torch.cat((edge_index1, tensor([aggr1, aggr2])), dim=0)
        # print(edge_index0)
        # print(edge_index1)
        idxs.append(v)
        num_neighbors[v] = (num_neighbors[last][1], num_neighbors[last][1] + 2)
        last = v
        v += 1
        # H_sorted = sorted(H_sorted.items(), key=itemgetter(1), reverse=True)
        total_process += 1
    print("Search time: ", search_time)
    # print(depths)
    # reorder vertices by their depths
    newNumV = v
    newIds = dict()
    for i in range(actual_len):
        newIds[idxs[i]] = idxs[i]
    nextId = initial_len
    d = 1
    while d <= maxDepth:
        rangeLeft = nextId
        i = actual_len
        while i < (actual_len + total_process):
            if depths[i - actual_len] == d:
                newIds[idxs[i]] = nextId
                # print("depths and newid: ", d, nextId)
                nextId += 1
            i += 1
        d += 1
        rangeRight = nextId - 1
        if rangeRight >= rangeLeft:
            ranges.append((rangeLeft, rangeRight))
            ranges_list.append(torch.tensor([2 * i for i in range(0, rangeRight + 1 - rangeLeft + 1)]))
    # newIds.append(-1)
    ranges_list = torch.cat(ranges_list, dim=0)
    # print(ranges_list)

    # delete -1 in tensors
    opt_edge0 = []
    opt_edge1 = []
    for i, j in zip(edge_index0, edge_index1):
        if j == -1:
            continue
        opt_edge0.append(i)
        opt_edge1.append(j)
    opt_edge_index = torch.stack((torch.tensor(opt_edge0), torch.tensor(opt_edge1)), 0)

    # create space for new nodes
    ret_feat = torch.vstack(
        (torch.zeros(x.size()), torch.zeros((newNumV - num_v, data.x.shape[1]), dtype=torch.int)))
    ret_feat[:x.shape[0]] = x

    num_neighbors, num_v, idxs, _ = get_neighborlist(opt_edge_index, ret_feat)

    # construct optimize list according to newIds
    opt_edge_index0 = []
    opt_edge_index1 = []
    for i in idxs:
        neigh = np.array(opt_edge1[num_neighbors[newIds[i]][0]:num_neighbors[newIds[i]][1]])
        for node in neigh:
            opt_edge_index0.append(i)
            opt_edge_index1.append(newIds[node])

    opt_edge_index = torch.stack((torch.tensor(opt_edge_index0), torch.tensor(opt_edge_index1)), 0)

    num_neighbors, num_v, idxs, _ = get_neighborlist(opt_edge_index, ret_feat)

    # aggregate intermediate results
    off = 0
    start = 0
    for range_ in ranges:
        begin, end = range_[0], range_[1]
        off += end + 1 - begin
        # range_index = range_index.view(-1)
        emb = ret_feat[opt_edge_index[1][num_neighbors[begin][0]:num_neighbors[end][1]]]
        # print(begin, end, start, off, ranges_list[start: off + 1], emb.shape)
        ret_feat[begin: end + 1] = segment_csr(emb, ranges_list[start: off + 1], reduce='sum')
        start = off + 1
        off += 1

    return newIds, ranges, ranges_list, opt_edge_index, ret_feat, num_neighbors, idxs, total_redundancy_eliminated, search_time


# get the simplified computation graph
def transfer_graph_contribution(aggregated_pairs, depths, num_v, num_neighbors, idxs, last, data, maxDepth):
    edge_index0 = data.edge_index[0]
    edge_index1 = data.edge_index[1]
    x = data.x
    v = num_v
    print("number of v: ", num_v)
    ranges = []
    ranges_list = []
    total_process = 0
    total_redundancy_eliminated = 0
    initial_len = num_v
    actual_len = len(idxs)
    for aggregated_pair in aggregated_pairs:
        aggr1 = aggregated_pair[0]
        aggr2 = aggregated_pair[1]
        # print("redundancy eliminated: ", redundancy_eliminated)
        for j in idxs:
            neigh = np.array(edge_index1[num_neighbors[j][0]:num_neighbors[j][1]])
            if aggr1 in neigh and aggr2 in neigh:
                i1 = np.where(neigh == aggr1)[0][0]
                i2 = np.where(neigh == aggr2)[0][0]
                edge_index1[num_neighbors[j][0] + i1] = v
                edge_index1[num_neighbors[j][0] + i2] = -1
                total_redundancy_eliminated += 1
        edge_index0 = torch.cat((edge_index0, tensor([v, v])), dim=0)
        edge_index1 = torch.cat((edge_index1, tensor([aggr1, aggr2])), dim=0)
        # print(edge_index0)
        # print(edge_index1)
        idxs.append(v)
        num_neighbors[v] = (num_neighbors[last][1], num_neighbors[last][1] + 2)
        last = v
        v += 1
        total_process += 1

    print(depths)
    print("total redundancy eliminated: ", total_redundancy_eliminated)
    # reorder vertices by their depths
    newNumV = v
    newIds = dict()
    for i in range(actual_len):
        newIds[idxs[i]] = idxs[i]
    nextId = initial_len
    d = 1
    while d <= maxDepth:
        rangeLeft = nextId
        i = actual_len
        while i < (actual_len + total_process):
            if depths[i - actual_len] == d:
                newIds[idxs[i]] = nextId
                # print("depths and newid: ", d, nextId)
                nextId += 1
            i += 1
        d += 1
        rangeRight = nextId - 1
        if rangeRight >= rangeLeft:
            ranges.append((rangeLeft, rangeRight))
            ranges_list.append(torch.tensor([2 * i for i in range(0, rangeRight + 1 - rangeLeft + 1)]))
    # newIds.append(-1)
    ranges_list = torch.cat(ranges_list, dim=0)

    # delete -1 in tensors
    opt_edge0 = []
    opt_edge1 = []
    for i, j in zip(edge_index0, edge_index1):
        if j == -1:
            continue
        opt_edge0.append(i)
        opt_edge1.append(j)
    opt_edge_index = torch.stack((torch.tensor(opt_edge0), torch.tensor(opt_edge1)), 0)

    # create space for new nodes
    ret_feat = torch.vstack(
        (torch.zeros(x.size()), torch.zeros((newNumV - num_v, data.x.shape[1]), dtype=torch.int)))
    ret_feat[:x.shape[0]] = x

    num_neighbors, num_v, idxs, _ = get_neighborlist(opt_edge_index, ret_feat)

    # construct optimize list according to newIds
    opt_edge_index0 = []
    opt_edge_index1 = []
    for i in idxs:
        neigh = np.array(opt_edge1[num_neighbors[newIds[i]][0]:num_neighbors[newIds[i]][1]])
        for node in neigh:
            opt_edge_index0.append(i)
            opt_edge_index1.append(newIds[node])

    opt_edge_index = torch.stack((torch.tensor(opt_edge_index0), torch.tensor(opt_edge_index1)), 0)

    num_neighbors, num_v, idxs, _ = get_neighborlist(opt_edge_index, ret_feat)

    # aggregate intermediate results
    off = 0
    start = 0
    for range_ in ranges:
        begin, end = range_[0], range_[1]
        off += end + 1 - begin
        # range_index = range_index.view(-1)
        emb = ret_feat[opt_edge_index[1][num_neighbors[begin][0]:num_neighbors[end][1]]]
        # print(begin, end, start, off, range_index[start: off], emb.shape)
        # print(begin, end, emb.shape, start, off, range_index[start: off], range_index)
        ret_feat[begin: end + 1] = segment_csr(emb, ranges_list[start: off + 1], reduce='sum')
        start = off + 1
        off += 1

    return newIds, ranges, ranges_list, opt_edge_index, ret_feat, num_neighbors, idxs, total_redundancy_eliminated


# get aggregated pairs with a small graph created with contribution
def get_aggregated_pairs(data, maxDepth, maxWidth, ratio):
    degree_dic = get_degree(data.edge_index)
    num_neighbors, vertex_index, num_v = get_attribution(data.edge_index)
    contribution_list = get_contribution_2(data.edge_index, degree_dic, vertex_index)
    sorted_list = contribution_list.copy()
    sorted_list.sort(reverse=True)
    rest_tensor = get_rest_edge(data.edge_index, vertex_index, contribution_list,
                                threshold=sorted_list[:int(ratio * len(contribution_list))][-1])
    H_sorted, num_v, num_neighbors, idxs, last = get_redundancy_heap(rest_tensor, data.x)
    print("length of heap after contribution: ", len(H_sorted))
    edge_index0 = rest_tensor[0]
    edge_index1 = rest_tensor[1]
    aggregated_pairs_list = []
    v = num_v
    print("number of v: ", num_v)
    depths = [None] * maxWidth
    initial_len = num_v
    while v < num_v + maxWidth:
        H_sorted = dict(H_sorted)
        if not H_sorted:
            break
        top = max(H_sorted.values())
        for pair, value in H_sorted.items():
            if value == top:
                break
        aggr1 = min(pair[0], pair[1])
        aggr2 = max(pair[0], pair[1])
        redundancy_eliminated = H_sorted.pop(pair)
        # print("redundancy eliminated: ", redundancy_eliminated)
        if redundancy_eliminated < 2:
            break
        preDepth = 0 if aggr1 < initial_len else depths[aggr1 - initial_len]
        if aggr2 >= initial_len:
            preDepth = max(preDepth, depths[aggr2 - initial_len])
        if preDepth >= maxDepth:
            continue
        depths[v - initial_len] = preDepth + 1
        for j in idxs:
            neigh = np.array(edge_index1[num_neighbors[j][0]:num_neighbors[j][1]])
            if aggr1 in neigh and aggr2 in neigh:
                i1 = np.where(neigh == aggr1)[0][0]
                i2 = np.where(neigh == aggr2)[0][0]
                edge_index1[num_neighbors[j][0] + i1] = v
                edge_index1[num_neighbors[j][0] + i2] = -1
                neigh = np.delete(neigh, [i1, i2])
                for k in neigh:
                    if k != -1:
                        sub_pair_count(k, aggr1, H_sorted)
                        sub_pair_count(k, aggr2, H_sorted)
                        add_pair_count(k, v, H_sorted)
        v += 1
        aggregated_pairs_list.append(pair)
    return aggregated_pairs_list, depths


class Dataset(object):
    def __init__(self, edge, x, y):
        self.edge_index = edge
        self.x = x
        self.y = y


# yelp, amazon, PPI, (flick, reddit)

path = '../data/reddit'
dataset = Reddit(path)
# print(dataset)
# train_loader = []
# for i, data in enumerate(dataset):
#     train_loader.append(data)
#     print(i, data, max(data.edge_index[0]))

# for graphsaint/gcn
train_loader = GraphSAINTRandomWalkSampler(dataset[0], batch_size=200, walk_length=2,
                                           num_steps=5, sample_coverage=100,
                                           save_dir='../data/reddit/')

# for graphsage
# train_loader = NeighborLoader(dataset[0], num_neighbors=[25, 10], batch_size=100, shuffle=False)

# for clustergcn
# cluster_data = ClusterData(dataset[0], num_parts=1500, recursive=False,
#                            save_dir='../data/reddit/')
# train_loader = ClusterLoader(cluster_data, batch_size=5, shuffle=False)

train_data = []
aggregated_train_data, aggregated_train_data_contribution1, aggregated_train_data_contribution2, aggregated_train_data_contribution3, aggregated_train_data_contribution4 = [], [], [], [], []
original_data_size = []
ranges_list, ranges_list_contribution1,  ranges_list_contribution2, ranges_list_contribution3, ranges_list_contribution4 = [], [], [], [], []
ranges_index_list, ranges_index_list_contribution1,  ranges_index_list_contribution2, ranges_index_list_contribution3, ranges_index_list_contribution4 = [], [], [], [], []
neighbors_list, neighbors_list_contribution1,  neighbors_list_contribution2, neighbors_list_contribution3, neighbors_list_contribution4 = [], [], [], [], []
newids_list, newids_list_contribution1, newids_list_contribution2, newids_list_contribution3, newids_list_contribution4 = [], [], [], [], []
idxs_list, idxs_list_contribution1, idxs_list_contribution2, idxs_list_contribution3, idxs_list_contribution4 = [], [], [], [], []
i = 0
total_search_time, total_search_time_contribution1, total_search_time_contribution2, total_search_time_contribution3, total_search_time_contribution4 = 0, 0, 0, 0, 0
total_construct_time, total_construct_time_contribution1, total_construct_time_contribution2, total_construct_time_contribution3, total_construct_time_contribution4 = 0, 0, 0, 0, 0
total_redundancy_eliminated, total_redundancy_contribution1, total_redundancy_contribution2, total_redundancy_contribution3, total_redundancy_contribution4 = 0, 0, 0, 0, 0
for data in train_loader:
    i += 1

    original_data = data.clone()
    contribution_data1 = data.clone()
    contribution_data2 = data.clone()
    contribution_data3 = data.clone()
    contribution_data4 = data.clone()
    original_size = data.x.shape[0]
    original_data_size.append(original_size)
    original_edge = data.edge_index.shape[1]
    train_data.append(original_data)
    print(data.x.shape, data.edge_index.shape, data.y.shape[0])

    # HAG
    t1 = time.time()
    H_sorted, num_v, num_neighbors, idxs, last = get_redundancy_heap(data.edge_index, data.x)
    t2 = time.time()
    total_search_time += (t2 - t1)
    print("get aggregated edge time: ", t2 - t1)
    newids, ranges, ranges_index, opt_edge_index, x, num_neighbors, idxs, redundancy_eliminated, search_time = \
        transfer_graph(H_sorted, num_v, num_neighbors, idxs, last, data, 4, int(data.x.shape[0]/4))
    total_search_time += search_time
    t3 = time.time()
    total_construct_time += (t3 - t2 - search_time)
    print("constructing HAG cost: ", t3 - t2)
    y = data.y
    simplified_dataset = Data(x=x, edge_index=opt_edge_index, y=y)
    aggregated_train_data.append(simplified_dataset)
    ranges_list.append(ranges)
    ranges_index_list.append(ranges_index)
    neighbors_list.append(num_neighbors)
    print(simplified_dataset.x.shape, simplified_dataset.edge_index.shape)
    newids_list.append(newids)
    idxs_list.append(idxs)
    total_redundancy_eliminated += redundancy_eliminated

    # HAG with contribution 0.5
    t3 = time.time()
    aggregated_pairs, depths = get_aggregated_pairs(contribution_data1, 4, int(data.x.shape[0]/4), 0.5)
    t4 = time.time()
    total_search_time_contribution1 += (t4 - t3)
    print("get aggregated edge time: ", t4 - t3)
    H_sorted, num_v, num_neighbors, idxs, last = get_redundancy_heap(contribution_data1.edge_index, contribution_data1.x)
    newids, ranges, ranges_index, opt_edge_index, x, num_neighbors, idxs, redundancy_eliminated = \
        transfer_graph_contribution(aggregated_pairs, depths, num_v, num_neighbors, idxs, last, contribution_data1, 4)
    t5 = time.time()
    total_construct_time_contribution1 += (t5 - t4)
    print("constructing HAG cost: ", t5 - t4)
    y = data.y
    simplified_dataset1 = Data(x=x, edge_index=opt_edge_index, y=y)
    aggregated_train_data_contribution1.append(simplified_dataset1)
    ranges_list_contribution1.append(ranges)
    ranges_index_list_contribution1.append(ranges_index)
    neighbors_list_contribution1.append(num_neighbors)
    print(simplified_dataset1.x.shape, simplified_dataset1.edge_index.shape)
    newids_list_contribution1.append(newids)
    idxs_list_contribution1.append(idxs)
    total_redundancy_contribution1 += redundancy_eliminated

    # HAG with contribution 0.6
    t5 = time.time()
    aggregated_pairs, depths = get_aggregated_pairs(contribution_data2, 4, int(data.x.shape[0] / 4), 0.6)
    t6 = time.time()
    total_search_time_contribution2 += (t6 - t5)
    print("get aggregated edge time: ", t6 - t5)
    H_sorted, num_v, num_neighbors, idxs, last = get_redundancy_heap(contribution_data2.edge_index, contribution_data2.x)
    newids, ranges, ranges_index, opt_edge_index, x, num_neighbors, idxs, redundancy_eliminated = \
        transfer_graph_contribution(aggregated_pairs, depths, num_v, num_neighbors, idxs, last, contribution_data2, 4)
    t7 = time.time()
    total_construct_time_contribution2 += (t7 - t6)
    print("constructing HAG cost: ", t7 - t6)
    y = data.y
    simplified_dataset2 = Data(x=x, edge_index=opt_edge_index, y=y)
    aggregated_train_data_contribution2.append(simplified_dataset2)
    ranges_list_contribution2.append(ranges)
    ranges_index_list_contribution2.append(ranges_index)
    neighbors_list_contribution2.append(num_neighbors)
    print(simplified_dataset2.x.shape, simplified_dataset2.edge_index.shape)
    newids_list_contribution2.append(newids)
    idxs_list_contribution2.append(idxs)
    total_redundancy_contribution2 += redundancy_eliminated

    # HAG with contribution 0.7
    t7 = time.time()
    aggregated_pairs, depths = get_aggregated_pairs(contribution_data3, 4, int(data.x.shape[0] / 4), 0.7)
    t8 = time.time()
    total_search_time_contribution3 += (t8 - t7)
    print("get aggregated edge time: ", t8 - t7)
    H_sorted, num_v, num_neighbors, idxs, last = get_redundancy_heap(contribution_data3.edge_index, contribution_data3.x)
    newids, ranges, ranges_index, opt_edge_index, x, num_neighbors, idxs, redundancy_eliminated = \
        transfer_graph_contribution(aggregated_pairs, depths, num_v, num_neighbors, idxs, last, contribution_data3, 4)
    t9 = time.time()
    total_construct_time_contribution3 += (t9 - t8)
    print("constructing HAG cost: ", t9 - t8)
    y = data.y
    simplified_dataset3 = Data(x=x, edge_index=opt_edge_index, y=y)
    aggregated_train_data_contribution3.append(simplified_dataset3)
    ranges_list_contribution3.append(ranges)
    ranges_index_list_contribution3.append(ranges_index)
    neighbors_list_contribution3.append(num_neighbors)
    print(simplified_dataset3.x.shape, simplified_dataset3.edge_index.shape)
    newids_list_contribution3.append(newids)
    idxs_list_contribution3.append(idxs)
    total_redundancy_contribution3 += redundancy_eliminated

    # HAG with contribution 0.8
    t9 = time.time()
    aggregated_pairs, depths = get_aggregated_pairs(contribution_data4, 4, int(data.x.shape[0] / 4), 0.8)
    t10 = time.time()
    total_search_time_contribution4 += (t10 - t9)
    print("get aggregated edge time: ", t10 - t9)
    H_sorted, num_v, num_neighbors, idxs, last = get_redundancy_heap(contribution_data4.edge_index, contribution_data4.x)
    newids, ranges, ranges_index, opt_edge_index, x, num_neighbors, idxs, redundancy_eliminated = \
        transfer_graph_contribution(aggregated_pairs, depths, num_v, num_neighbors, idxs, last, contribution_data4, 4)
    t11 = time.time()
    total_construct_time_contribution4 += (t11 - t10)
    print("constructing HAG cost: ", t11 - t10)
    y = data.y
    simplified_dataset4 = Data(x=x, edge_index=opt_edge_index, y=y)
    aggregated_train_data_contribution4.append(simplified_dataset4)
    ranges_list_contribution4.append(ranges)
    ranges_index_list_contribution4.append(ranges_index)
    neighbors_list_contribution4.append(num_neighbors)
    print(simplified_dataset4.x.shape, simplified_dataset4.edge_index.shape)
    newids_list_contribution4.append(newids)
    idxs_list_contribution4.append(idxs)
    total_redundancy_contribution4 += redundancy_eliminated

    if i == 20:
        break

print("search time:{}, contribution 0.5 search time: {}, contribution 0.6 search time: {}, contribution 0.7 search time: {}, contribution 0.8 search time: {}"
      .format(total_search_time, total_search_time_contribution1, total_search_time_contribution2, total_search_time_contribution3, total_search_time_contribution4))
print("hag time: {}, hag with contribution 0.5 time:{}, hag with contribution 0.6 time:{}, hag with contribution 0.7 time:{}, hag with contribution 0.8 time:{}"
      .format(total_construct_time, total_construct_time_contribution1, total_construct_time_contribution2, total_construct_time_contribution3, total_construct_time_contribution4))
print("redudancy eliminated: {}, {}, {}, {}, {}" .format(total_redundancy_eliminated,
                                                         total_redundancy_contribution1, total_redundancy_contribution2, total_redundancy_contribution3, total_redundancy_contribution4))

device = torch.device('cuda:0')
model = GCN(data.x.shape[1], hidden_channels=128, out_channels=dataset.num_classes).to(device)
model0 = GCN(data.x.shape[1], hidden_channels=128, out_channels=dataset.num_classes).to(device)
model1 = GCN(data.x.shape[1], hidden_channels=128, out_channels=dataset.num_classes).to(device)
model2 = GCN(data.x.shape[1], hidden_channels=128, out_channels=dataset.num_classes).to(device)
model3 = GCN(data.x.shape[1], hidden_channels=128, out_channels=dataset.num_classes).to(device)
model4 = GCN(data.x.shape[1], hidden_channels=128, out_channels=dataset.num_classes).to(device)
print("Model has been transferred to GPU")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
optimizer0 = torch.optim.Adam(model0.parameters(), lr=0.001, weight_decay=5e-4)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=5e-4)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=5e-4)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001, weight_decay=5e-4)
optimizer4 = torch.optim.Adam(model4.parameters(), lr=0.001, weight_decay=5e-4)
loss_op = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()

    total_loss = total_correct = total_examples = 0
    ys, preds = [], []
    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data, original_size=data.x.shape[0])
        loss = F.nll_loss(out, data.y)
        # ys.append(data.y)
        # preds.append((out > 0).float())
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
        loss.backward()
        optimizer.step()

    # y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    # f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return total_loss / total_examples, total_correct / total_examples


def train_with_rr():
    model0.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    ys, preds = [], []
    for data in aggregated_train_data:
        data = data.to(device)
        original_size = original_data_size[i]
        optimizer0.zero_grad()
        out = model0(data, original_size)
        loss = F.nll_loss(out, data.y)
        # ys.append(data.y)
        # preds.append((out > 0).float())
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * original_size
        total_examples += original_size
        loss.backward()
        optimizer0.step()
        i += 1

    # y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    # f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return total_loss / total_examples, total_correct / total_examples


def train_with_rr_contribution1():
    model1.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    ys, preds = [], []
    for data in aggregated_train_data_contribution1:
        data = data.to(device)
        original_size = original_data_size[i]
        optimizer1.zero_grad()
        out = model1(data, original_size)
        loss = F.nll_loss(out, data.y)
        # ys.append(data.y)
        # preds.append((out > 0).float())
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * original_size
        total_examples += original_size
        loss.backward()
        optimizer1.step()
        i += 1

    # y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    # f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return total_loss / total_examples, total_correct / total_examples


def train_with_rr_contribution2():
    model2.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    ys, preds = [], []
    for data in aggregated_train_data_contribution2:
        data = data.to(device)
        original_size = original_data_size[i]
        optimizer2.zero_grad()
        out = model2(data, original_size)
        loss = F.nll_loss(out, data.y)
        # ys.append(data.y)
        # preds.append((out > 0).float())
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * original_size
        total_examples += original_size
        loss.backward()
        optimizer2.step()
        i += 1

    # y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    # f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return total_loss / total_examples, total_correct / total_examples


def train_with_rr_contribution3():
    model3.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    ys, preds = [], []
    for data in aggregated_train_data_contribution3:
        data = data.to(device)
        original_size = original_data_size[i]
        optimizer3.zero_grad()
        out = model3(data, original_size)
        loss = F.nll_loss(out, data.y)
        # ys.append(data.y)
        # preds.append((out > 0).float())
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * original_size
        total_examples += original_size
        loss.backward()
        optimizer3.step()
        i += 1

    # y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    # f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return total_loss / total_examples, total_correct / total_examples


def train_with_rr_contribution4():
    model4.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    ys, preds = [], []
    for data in aggregated_train_data_contribution4:
        data = data.to(device)
        original_size = original_data_size[i]
        optimizer4.zero_grad()
        out = model4(data, original_size)
        loss = F.nll_loss(out, data.y)
        # ys.append(data.y)
        # preds.append((out > 0).float())
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * original_size
        total_examples += original_size
        loss.backward()
        optimizer4.step()
        i += 1

    # y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    # f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return total_loss / total_examples, total_correct / total_examples
#
#
# # test for ppi dataset
# # def test(loader):
# #     model.eval()
# #
# #     ys, preds = [], []
# #     for data in loader:
# #         ys.append(data.y)
# #         out = model(data.to(device), original_size=data.x.shape[0])
# #         preds.append((out > 0).float().cpu())
# #
# #     y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
# #     print(y, pred)
# #     return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
#
# # test for reddit dataset
# # def test():
# #     model.eval()
# #     logits, accs = model.inference(dataset.x, subgraph_loader, device), []
# #     y = dataset.y.to(logits.device)
# #     for _, mask in dataset('val_mask', 'test_mask'):
# #         accs.append(int((logits[mask] == y[mask]).sum()) / int(mask.sum()))
# #     return accs


t1 = time.time()
max_acc1 = 0
for epoch in range(350):
    loss, acc = train()
    # train_acc, val_acc, test_acc = test()
    if acc > max_acc1:
        max_acc1 = acc
    # print('Epoch: {:02d}, Loss: {:.4f}, f1/acc: {:.4f}'.format(
    #     epoch, loss, acc))
t2 = time.time()
torch.cuda.empty_cache()
print("total training time: ", t2 - t1)

t3 = time.time()
max_acc2 = 0
for epoch in range(350):
    loss, acc = train_with_rr()
    if acc > max_acc2:
        max_acc2 = acc
    # print('Epoch: {:02d}, Loss: {:.4f}, f1/acc: {:.4f}'.format(
    #     epoch, loss, acc))
t4 = time.time()
torch.cuda.empty_cache()
print("total training time with rr: ", t4 - t3)

t5 = time.time()
max_acc3 = 0
for epoch in range(350):
    loss, acc = train_with_rr_contribution1()
    if acc > max_acc3:
        max_acc3 = acc
    # print('Epoch: {:02d}, Loss: {:.4f}, f1/acc: {:.4f}'.format(
    #     epoch, loss, acc))
t6 = time.time()
torch.cuda.empty_cache()
print("total training time with rr and contribution: ", t6 - t5)

t7 = time.time()
max_acc4 = 0
for epoch in range(350):
    loss, acc = train_with_rr_contribution2()
    if acc > max_acc4:
        max_acc4 = acc
    # print('Epoch: {:02d}, Loss: {:.4f}, f1/acc: {:.4f}'.format(
    #     epoch, loss, acc))
t8 = time.time()
torch.cuda.empty_cache()
print("total training time with rr and contribution: ", t8 - t7)

t9 = time.time()
max_acc5 = 0
for epoch in range(350):
    loss, acc = train_with_rr_contribution3()
    if acc > max_acc5:
        max_acc5 = acc
    # print('Epoch: {:02d}, Loss: {:.4f}, f1/acc: {:.4f}'.format(
    #     epoch, loss, acc))
t10 = time.time()
torch.cuda.empty_cache()
print("total training time with rr and contribution: ", t10 - t9)

t11 = time.time()
max_acc6 = 0
for epoch in range(350):
    loss, acc = train_with_rr_contribution4()
    if acc > max_acc6:
        max_acc6 = acc
    # print('Epoch: {:02d}, Loss: {:.4f}, f1/acc: {:.4f}'.format(
    #     epoch, loss, acc))
t12 = time.time()
torch.cuda.empty_cache()
print("total training time with rr and contribution: ", t12 - t11)

print("time compare: ", t2 - t1, t4 - t3, t6 - t5, t8 - t7, t10 - t9, t12 - t11)
print("acc compare:", max_acc1, max_acc2, max_acc3, max_acc4, max_acc5, max_acc6)