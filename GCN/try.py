from torch import tensor
import torch
import operator
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Reddit, PPI, Yelp, AmazonProducts, Flickr
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
from sortedcollections import ValueSortedDict
from operator import itemgetter
torch.set_printoptions(threshold=np.inf)
from collections import OrderedDict

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
    num_v = x.shape[0]
    return num_neighbors, num_v, idxs


# get the redundancy of two nodes
def get_redundancy_heap(edge, x):
    num_neighbors, num_v, idxs = get_neighborlist(edge, x)
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
    H_sorted = sorted(pair_redundancy.items(), key=itemgetter(1), reverse=True)

    return H_sorted, num_v, num_neighbors, idxs


# get aggregated pairs with a small graph created with contribution
def get_aggregated_pairs(data, maxDepth, maxWidth):
    t1 = time.time()
    degree_dic = get_degree(data.edge_index)
    num_neighbors, vertex_index, num_v = get_attribution(data.edge_index)
    contribution_list = get_contribution_2(data.edge_index, degree_dic, vertex_index)
    sorted_list = contribution_list.copy()
    sorted_list.sort(reverse=True)
    rest_tensor = get_rest_edge(data.edge_index, vertex_index, contribution_list,
                                threshold=sorted_list[:int(1 * len(contribution_list) / 2)][-1])
    H_sorted, num_v, num_neighbors, idxs = get_redundancy_heap(rest_tensor, data.x)
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
        top, pair = list(H_sorted.values())[0], list(H_sorted.keys())[0]
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
        H_sorted = sorted(H_sorted.items(), key=itemgetter(1), reverse=True)
    t2 = time.time()
    exact_time = t2 - t1
    print("exact pre-process time: ", exact_time)
    return aggregated_pairs_list, depths


# get the simplified computation graph
def transfer_graph(H_sorted, num_v, num_neighbors, idxs, data, maxDepth, maxWidth):
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
    v_list = []
    while v < num_v + maxWidth:
        H_sorted = dict(H_sorted)
        if not H_sorted:
            break
        top, pair = list(H_sorted.values())[0], list(H_sorted.keys())[0]
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
        edge_index0 = torch.cat((edge_index0, tensor([v, v])), dim=0)
        edge_index1 = torch.cat((edge_index1, tensor([aggr1, aggr2])), dim=0)
        # print(edge_index0)
        # print(edge_index1)
        idxs.append(v)
        num_neighbors[v] = (num_neighbors[v-1][1], num_neighbors[v-1][1] + 2)
        v += 1
        H_sorted = sorted(H_sorted.items(), key=itemgetter(1), reverse=True)
        total_process += 1

    print(depths)
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
            ranges_list.append(torch.tensor([i for i in range(rangeRight + 1 - rangeLeft) for k in range(2)]))
            v_list.append(torch.tensor([1] * (rangeRight - rangeLeft + 1) * 2, dtype=torch.float32))
    # newIds.append(-1)
    ranges_list = torch.cat(ranges_list, dim=0)

    print(v_list)

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

    num_neighbors, num_v, idxs = get_neighborlist(opt_edge_index, ret_feat)

    print("opt1: ", opt_edge_index)
    # construct optimize list according to newIds
    opt_edge_index0 = []
    opt_edge_index1 = []
    print("new id: ", newIds)
    print("id: ", idxs)
    for i in idxs:
        neigh = np.array(opt_edge1[num_neighbors[newIds.index(i)][0]:num_neighbors[newIds.index(i)][1]])
        print("vertex: ", i)
        for node in neigh:
            opt_edge_index0.append(i)
            opt_edge_index1.append(newIds[idxs.index(node)])
            print(newIds[idxs.index(node)])

    opt_edge_index = torch.stack((torch.tensor(opt_edge_index0), torch.tensor(opt_edge_index1)), 0)

    print("opt2: ", opt_edge_index)
    print("total redundancy eliminate: ", total_redundancy_eliminated)

    num_neighbors, num_v, idxs = get_neighborlist(opt_edge_index, ret_feat)

    return newIds, ranges, ranges_list, v_list, opt_edge_index, ret_feat, num_neighbors, idxs, total_redundancy_eliminated


# get the simplified computation graph
def transfer_graph_contribution(aggregated_pairs, depths, num_v, num_neighbors, idxs, data, maxDepth, maxWidth):
    edge_index0 = data.edge_index[0]
    edge_index1 = data.edge_index[1]
    x = data.x
    v = num_v
    print("number of v: ", num_v)
    ranges = []
    ranges_list = []
    total_process = 0
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
                edge_index1[num_neighbors[j][1] + i2] = -1
                neigh = np.delete(neigh, [i1, i2])
        edge_index0 = torch.cat((edge_index0, tensor([v, v])), dim=0)
        edge_index1 = torch.cat((edge_index1, tensor([aggr1, aggr2])), dim=0)
        # print(edge_index0)
        # print(edge_index1)
        idxs.append(v)
        num_neighbors.append(num_neighbors[idxs.index(v)] + 2)
        v += 1
        total_process += 1

    print(depths)
    # reorder vertices by their depths
    newNumV = v
    newIds = [None] * (actual_len + total_process)
    for i in range(actual_len):
        newIds[i] = idxs[i]
    nextId = initial_len
    d = 1
    while d <= maxDepth:
        rangeLeft = nextId
        i = actual_len
        while i < (actual_len + total_process):
            if depths[i - actual_len] == d:
                newIds[i] = nextId
                # print("depths and newid: ", d, nextId)
                nextId += 1
            i += 1
        d += 1
        rangeRight = nextId - 1
        if rangeRight >= rangeLeft:
            ranges.append((rangeLeft, rangeRight))
            ranges_list.append(torch.tensor([i for i in range(rangeRight + 1 - rangeLeft) for k in range(2)]))
    # newIds.append(-1)
    ranges_list = torch.cat(ranges_list, dim=0)
    print("range list: ", ranges_list)

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

    num_neighbors, num_v, idxs = get_neighborlist(opt_edge_index, ret_feat)

    print("opt1: ", opt_edge_index)
    # construct optimize list according to newIds
    opt_edge_index0 = []
    opt_edge_index1 = []
    print("new id: ", newIds)
    print("id: ", idxs)
    for i in idxs:
        neigh = np.array(opt_edge1[num_neighbors[newIds.index(i)]:num_neighbors[newIds.index(i) + 1]])
        print("vertex: {}, neigh: {}" .format(i,  neigh))
        for node in neigh:
            opt_edge_index0.append(i)
            opt_edge_index1.append(newIds[idxs.index(node)])
            print(newIds[idxs.index(node)])

    opt_edge_index = torch.stack((torch.tensor(opt_edge_index0), torch.tensor(opt_edge_index1)), 0)

    print("opt2: ", opt_edge_index)

    num_neighbors, num_v, idxs = get_neighborlist(opt_edge_index, ret_feat)

    return newIds, ranges, ranges_list, opt_edge_index, ret_feat, num_neighbors, idxs


edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6],
                           [2, 5, 3, 2, 3, 5, 0, 1, 4, 6, 0, 1, 6, 4, 2, 3, 5, 0, 1, 4, 6, 2, 3, 5]])
x = torch.tensor([[1, 0, 0, 0, 0, 0, 1],
     [0, 1, 0, 0, 0, 0, 1],
     [0, 0, 1, 0, 0, 0, 1],
     [0, 0, 0, 1, 0, 0, 1],
     [0, 0, 0, 0, 1, 0, 1],
     [0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 0, 1]])


class Dataset(object):
    def __init__(self, edge, x):
        self.edge_index = edge
        self.x = x


data = Dataset(edge_index, x)

aggregated_pairs, depths = get_aggregated_pairs(data, 2, 5)
print(aggregated_pairs, depths)

H_sorted, num_v, num_neighbors, idxs = get_redundancy_heap(data.edge_index, data.x)
newIds, ranges, range_list, v_list, opt_edge_index, ret_feat, num_neighbors, idxs, _ = \
    transfer_graph(H_sorted, num_v, num_neighbors, idxs, data, 3, 5)
print(opt_edge_index)

simple_data = Dataset(opt_edge_index, ret_feat)

edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6],
                           [2, 5, 3, 2, 3, 5, 0, 1, 4, 6, 0, 1, 6, 4, 2, 3, 5, 0, 1, 4, 6, 2, 3, 5]])
x = torch.tensor([[1, 0, 0, 0, 0, 0, 1],
     [0, 1, 0, 0, 0, 0, 1],
     [0, 0, 1, 0, 0, 0, 1],
     [0, 0, 0, 1, 0, 0, 1],
     [0, 0, 0, 0, 1, 0, 1],
     [0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 0, 1]])

# edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6],
#                            [2, 5, 3, 2, 3, 5, 0, 1, 4, 6, 0, 1, 6, 4, 2, 3, 5, 0, 1, 4, 6, 2, 3, 5]])
# x = torch.tensor([[1, 0, 0, 0, 0, 0, 1],
#      [0, 1, 0, 0, 0, 0, 1],
#      [0, 0, 1, 0, 0, 0, 1],
#      [0, 0, 0, 1, 0, 0, 1],
#      [0, 0, 0, 0, 1, 0, 1],
#      [0, 0, 0, 0, 0, 1, 1],
#      [0, 0, 0, 0, 0, 0, 1]])


def conv_sum(edge_index, x):
    current_node = edge_index[0][0]
    aggregated_emb = torch.zeros(x.shape, dtype=torch.float)
    for dst, src in zip(edge_index[0], edge_index[1]):
        if dst == current_node:
            aggregated_emb[current_node] += x[src]
        else:
            current_node = dst
            aggregated_emb[current_node] += x[src]
    return x + aggregated_emb


def model1(edge_index, x):
    layers = 2
    i = 0
    while i < layers:
        x = conv_sum(edge_index, x)
        print("Round: ", i)
        print(x)
        i += 1
    return x


from torch_scatter import segment_coo

original_size = 6

zero_tensor = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0])

def model2(edge_index, x, ranges, range_index, neighbors, newids, idxs):
    layers = 2
    i = 0
    print("ranges: ", ranges)
    print("range_index: ", range_index)
    while i < layers:
        off = 0
        start = 0
        # aggregate intermediate results
        for range_ in ranges:
            begin, end = range_[0], range_[1]
            off += (end + 1 - begin) * 2
            range_index = range_index.view(-1)
            emb = x[edge_index[1][neighbors[idxs.index(begin)]:neighbors[idxs.index(end) + 1]]]
            x[begin: end + 1] = segment_coo(emb, range_index[start: off], reduce='sum')
            start = off
        x = conv_sum(edge_index, x)
        print("Round: ", i)
        print(x)
        i += 1
    return x

def model3(edge_index, x, ranges, v_list, neighbors, newids, idxs):
    layers = 2
    i = 0
    print("ranges: ", ranges)
    print("v_list: ", v_list)
    while i < layers:
        # aggregate intermediate results
        k = 0
        for range_ in ranges:
            begin, end = range_[0], range_[1]
            src = edge_index[0][neighbors[idxs.index(begin)]:neighbors[idxs.index(end) + 1]]
            dst = edge_index[1][neighbors[idxs.index(begin)]:neighbors[idxs.index(end) + 1]]
            # create values
            v = v_list[k]
            # v = torch.tensor([1] * len(src), dtype=torch.float32)
            # create sparse_coo_tensor
            index = torch.stack((torch.tensor(src), torch.tensor(dst)), 0)
            print(index, v)
            sparse_tensor = torch.sparse_coo_tensor(index, v, [end + 1, end + 1])
            dense_tensor = sparse_tensor.to_dense()
            x[begin: end + 1] = torch.mm(dense_tensor, x[:end + 1], out=None)[begin: end + 1]
            k += 1
        x = conv_sum(edge_index, x)
        print("Round: ", i)
        print(x)
        i += 1
    return x

t1 = time.time()
out1 = model1(edge_index, x)
t2 = time.time()
out2 = model2(opt_edge_index, ret_feat, ranges, range_list, num_neighbors, newIds, idxs)
t3 = time.time()
out3 = model3(opt_edge_index, ret_feat, ranges, v_list, num_neighbors, newIds, idxs)
t4 = time.time()
print(out1, t2 - t1)
print(out2, t3 - t2)
print(out3, t4 - t3)
#
#
#
#
#
