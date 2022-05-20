from torch import tensor
import torch
import operator
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Reddit, PPI, Yelp, AmazonProducts, Flickr
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
    num_neighbors = []
    idxs = []
    num_neighbors.append(0)
    count = 1
    current = edge[0][0]
    idxs.append(current.item())
    for i in edge[0][1:]:
        if i == current:
            count += 1
        else:
            num_neighbors.append(count)
            current = i
            idxs.append(current.item())
            count += 1
    num_neighbors.append(count)
    num_v = x.shape[0]
    return num_neighbors, num_v, idxs


# get the redundancy of two nodes
def get_redundancy_heap(edge, x, capacity):

    edge_to_aggr = []

    num_neighbors, num_v, idxs = get_neighborlist(edge, x)
    # vertex_index = np.unique(data.edge_index[0].tolist())
    pair_redundancy = dict()
    for v in idxs:
        neigh = edge[1][num_neighbors[idxs.index(v)]:num_neighbors[idxs.index(v)+1]].tolist()
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
        if len(edge_to_aggr) == int(num_v / 2) or cap == capacity:
            break

    return edge_to_aggr, num_v, num_neighbors, idxs


# get the simplified computation graph
def transfer_graph(edge_to_aggr, num_v, num_neighbors, idxs, data, maxDepth):
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
    depths = [1] * len(edge_to_aggr)
    i_edges = dict()
    t1 = time.time()
    for (aggr1, aggr2) in edge_to_aggr:
        index1 = idxs.index(aggr1)
        index2 = idxs.index(aggr2)
        neigh1 = np.array(edge_index1[num_neighbors[index1]:num_neighbors[index1 + 1]])
        neigh2 = np.array(edge_index1[num_neighbors[index2]:num_neighbors[index2 + 1]])
        # find the nodes whose neighbors containing (aggr1, aggr2)
        i_edges[(aggr1, aggr2)] = np.intersect1d(neigh1, neigh2, assume_unique=True)

    for (aggr1, aggr2) in edge_to_aggr:
        v_root = i_edges[(aggr1, aggr2)]
        # print("redundancy eliminated: ", redundancy_eliminated)
        for j in v_root:
            neigh = np.array(edge_index1[num_neighbors[idxs.index(j)]:num_neighbors[idxs.index(j) + 1]])
            if aggr1 in neigh and aggr2 in neigh:
                i1 = np.where(neigh == aggr1)[0][0]
                i2 = np.where(neigh == aggr2)[0][0]
                edge_index1[num_neighbors[idxs.index(j)] + i1] = v
                edge_index1[num_neighbors[idxs.index(j)] + i2] = -1
                total_redundancy_eliminated += 1
        edge_index0 = torch.cat((edge_index0, tensor([v, v])), dim=0)
        edge_index1 = torch.cat((edge_index1, tensor([aggr1, aggr2])), dim=0)
        # print(edge_index0)
        # print(edge_index1)
        idxs.append(v)
        num_neighbors.append(num_neighbors[idxs.index(v)] + 2)
        v += 1
        total_process += 1
    t2 = time.time()
    print("step1: ", t2 - t1)

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
    # print(ranges_list)
    t3 = time.time()
    print("step2: ", t3 - t2)

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

    # construct optimize list according to newIds
    opt_edge_index0 = []
    opt_edge_index1 = []
    for i in idxs:
        neigh = np.array(opt_edge1[num_neighbors[newIds.index(i)]:num_neighbors[newIds.index(i) + 1]])
        for node in neigh:
            opt_edge_index0.append(i)
            opt_edge_index1.append(newIds[idxs.index(node)])

    t4 = time.time()
    print("step3: ", t4 - t3)

    opt_edge_index = torch.stack((torch.tensor(opt_edge_index0), torch.tensor(opt_edge_index1)), 0)

    num_neighbors, num_v, idxs = get_neighborlist(opt_edge_index, ret_feat)

    return newIds, ranges, ranges_list, opt_edge_index, ret_feat, num_neighbors, idxs, total_redundancy_eliminated


# get the simplified computation graph
def transfer_graph_contribution(aggregated_pairs, depths, num_v, num_neighbors, idxs, data):
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
    i_edges = dict()
    t1 = time.time()
    for (aggr1, aggr2) in aggregated_pairs:
        index1 = idxs.index(aggr1)
        index2 = idxs.index(aggr2)
        neigh1 = np.array(edge_index1[num_neighbors[index1]:num_neighbors[index1 + 1]])
        neigh2 = np.array(edge_index1[num_neighbors[index2]:num_neighbors[index2 + 1]])
        # find the nodes whose neighbors containing (aggr1, aggr2)
        i_edges[(aggr1, aggr2)] = np.intersect1d(neigh1, neigh2, assume_unique=True)

    for (aggr1, aggr2) in aggregated_pairs:
        v_root = i_edges[(aggr1, aggr2)]
        # print("redundancy eliminated: ", redundancy_eliminated)
        for j in v_root:
            neigh = np.array(edge_index1[num_neighbors[idxs.index(j)]:num_neighbors[idxs.index(j) + 1]])
            if aggr1 in neigh and aggr2 in neigh:
                i1 = np.where(neigh == aggr1)[0][0]
                i2 = np.where(neigh == aggr2)[0][0]
                edge_index1[num_neighbors[idxs.index(j)] + i1] = v
                edge_index1[num_neighbors[idxs.index(j)] + i2] = -1
                total_redundancy_eliminated += 1
        edge_index0 = torch.cat((edge_index0, tensor([v, v])), dim=0)
        edge_index1 = torch.cat((edge_index1, tensor([aggr1, aggr2])), dim=0)
        # print(edge_index0)
        # print(edge_index1)
        idxs.append(v)
        num_neighbors.append(num_neighbors[idxs.index(v)] + 2)
        v += 1
        total_process += 1
    t2 = time.time()
    print("step1: ", t2 - t1)

    print(depths)
    print("total redundancy eliminated: ", total_redundancy_eliminated)
    # reorder vertices by their depths
    newNumV = v
    newIds = [None] * (actual_len + total_process)
    for i in range(actual_len + total_process):
        newIds[i] = idxs[i]
    rangeLeft = initial_len
    rangeRight = initial_len + total_process - 1
    ranges.append((rangeLeft, rangeRight))
    ranges_list.append(torch.tensor([i for i in range(rangeRight + 1 - rangeLeft) for k in range(2)]))
    # newIds.append(-1)
    ranges_list = torch.cat(ranges_list, dim=0)
    t3 = time.time()
    print("step2: ", t3 - t2)

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

    # construct optimize list according to newIds
    opt_edge_index0 = []
    opt_edge_index1 = []

    for i in idxs:
        neigh = np.array(opt_edge1[num_neighbors[newIds.index(i)]:num_neighbors[newIds.index(i) + 1]])
        for node in neigh:
            opt_edge_index0.append(i)
            opt_edge_index1.append(newIds[idxs.index(node)])
    t4 = time.time()
    print("step3: ", t4 - t3)

    opt_edge_index = torch.stack((torch.tensor(opt_edge_index0), torch.tensor(opt_edge_index1)), 0)


    num_neighbors, num_v, idxs = get_neighborlist(opt_edge_index, ret_feat)

    return newIds, ranges, ranges_list, opt_edge_index, ret_feat, num_neighbors, idxs, total_redundancy_eliminated


# get aggregated pairs with a small graph created with contribution
def get_aggregated_pairs(data, maxWidth, ratio):
    ta = time.time()
    degree_dic = get_degree(data.edge_index)
    num_neighbors, vertex_index, num_v = get_attribution(data.edge_index)
    contribution_list = get_contribution_2(data.edge_index, degree_dic, vertex_index)
    sorted_list = contribution_list.copy()
    sorted_list.sort(reverse=True)
    rest_tensor = get_rest_edge(data.edge_index, vertex_index, contribution_list,
                                threshold=sorted_list[:int(ratio * len(contribution_list))][-1])
    tb = time.time()
    edge_to_aggr, num_v, num_neighbors, idxs = get_redundancy_heap(rest_tensor, data.x, int(data.x.shape[0]/4))
    tc = time.time()
    depths = [1] * maxWidth
    print("extra-time: ", tb - ta)
    print("get aggregated nodes time: ", tc - tb)
    return edge_to_aggr, depths


class Dataset(object):
    def __init__(self, edge, x, y):
        self.edge_index = edge
        self.x = x
        self.y = y


# yelp, amazon, PPI, (flick, reddit)

path = '../data/cora'
dataset = Planetoid(path, 'Cora')
print(dataset)
# train_loader = []
# for i, data in enumerate(dataset):
#     train_loader.append(data)
#     print(i, data, max(data.edge_index[0]))

# for graphsaint/gcn
# train_loader = GraphSAINTRandomWalkSampler(dataset[0], batch_size=1000, walk_length=2,
#                                            num_steps=5, sample_coverage=100,
#                                            save_dir='../data/reddit/')

# for graphsage
# train_loader = NeighborLoader(dataset[0], num_neighbors=[25, 10], batch_size=100, shuffle=False)

# for clustergcn
# cluster_data = ClusterData(dataset[0], num_parts=1500, recursive=False,
#                            save_dir='../data/reddit/')
# train_loader = ClusterLoader(cluster_data, batch_size=5, shuffle=False)

train_data = []
aggregated_train_data, aggregated_train_data_contribution = [], []
original_data_size = []
ranges_list, ranges_list_contribution = [], []
ranges_index_list, ranges_index_list_contribution = [], []
neighbors_list, neighbors_list_contribution = [], []
newids_list, newids_list_contribution = [], []
idxs_list, idxs_list_contribution = [], []
i = 0
total_construct_time, total_construct_time_contribution = 0, 0
total_extra_time = 0
total_redundancy_eliminated, total_redundancy_contribution = 0, 0
for data in dataset:
    original_data = data.clone()
    contribution_data = data.clone()
    original_size = data.x.shape[0]
    original_data_size.append(original_size)
    original_edge = data.edge_index.shape[1]
    train_data.append(original_data)
    print(data.x.shape, data.edge_index.shape, data.y.shape[0])

    # HAG
    t1 = time.time()
    edge_to_aggr, num_v, num_neighbors, idxs = get_redundancy_heap(data.edge_index, data.x, capacity=int(data.x.shape[0]/4))
    t2 = time.time()
    print("get aggregated edge time: ", t2 - t1)
    newids, ranges, ranges_index, opt_edge_index, x, num_neighbors, idxs, redundancy_eliminated = \
        transfer_graph(edge_to_aggr, num_v, num_neighbors, idxs, data, 4)
    t3 = time.time()
    total_construct_time += (t3 - t2)
    print("constructing act cost: ", t3 - t2)
    y = data.y
    simplified_dataset1 = Dataset(opt_edge_index, x, y)
    aggregated_train_data.append(simplified_dataset1)
    ranges_list.append(ranges)
    ranges_index_list.append(ranges_index)
    neighbors_list.append(num_neighbors)
    print(simplified_dataset1.x.shape, simplified_dataset1.edge_index.shape)
    newids_list.append(newids)
    idxs_list.append(idxs)
    total_redundancy_eliminated += redundancy_eliminated

    # HAG with contribution
    t3 = time.time()
    aggregated_pairs, depths = get_aggregated_pairs(contribution_data, int(data.x.shape[0]/4), 0.5)
    t4 = time.time()
    total_extra_time += t4 - t3
    print("get aggregated edge time: ", t4 - t3)
    num_neighbors, num_v, idxs = get_neighborlist(contribution_data.edge_index, contribution_data.x)
    newids, ranges, ranges_index, opt_edge_index, x, num_neighbors, idxs, redundancy_eliminated = \
        transfer_graph_contribution(aggregated_pairs, depths, num_v, num_neighbors, idxs, contribution_data)
    t5 = time.time()
    total_construct_time_contribution += (t5 - t4)
    print("constructing act cost: ", t5 - t4)
    y = data.y
    simplified_dataset2 = Dataset(opt_edge_index, x, y)
    aggregated_train_data_contribution.append(simplified_dataset2)
    ranges_list_contribution.append(ranges)
    ranges_index_list_contribution.append(ranges_index)
    neighbors_list_contribution.append(num_neighbors)
    print(simplified_dataset2.x.shape, simplified_dataset2.edge_index.shape)
    newids_list_contribution.append(newids)
    idxs_list_contribution.append(idxs)
    total_redundancy_contribution += redundancy_eliminated

    # if i == 1:
    #     break

print("act time: {}, hag with contribution time:{}, contribution extra time: {}"
      .format(total_construct_time, total_construct_time_contribution, total_extra_time))
print("redudancy eliminated: {}, {}" .format(total_redundancy_eliminated, total_redundancy_contribution))

device = torch.device('cuda:0')
model1 = GCN(data.x.shape[1], hidden_channels=16, out_channels=dataset.num_classes).to(device)
model2 = GCNHAG(data.x.shape[1], hidden_channels=16, out_channels=dataset.num_classes).to(device)
model3 = GCNHAG(data.x.shape[1], hidden_channels=16, out_channels=dataset.num_classes).to(device)
print("Model has been transferred to GPU")
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=5e-4)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.01, weight_decay=5e-4)
# loss_op = torch.nn.BCEWithLogitsLoss()

def train_with_rr():
    model2.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    for data in aggregated_train_data:
        data.edge_index = data.edge_index.to(device)
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        original_size = original_data_size[i]
        ranges = ranges_list[i]
        ranges_index = ranges_index_list[i].to(device)
        neighbors = neighbors_list[i]
        idxs = idxs_list[i]
        optimizer2.zero_grad()
        out = model2(data, ranges, ranges_index, original_size, neighbors, idxs)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        # print("batch's correct time: ", int((out.argmax(dim=-1) == data.y).sum()))
        optimizer2.step()
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * original_size
        total_examples += original_size
        i += 1

    return total_loss / total_examples, total_correct / total_examples


def train_with_rr_contribution():
    model3.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    for data in aggregated_train_data_contribution:
        data.edge_index = data.edge_index.to(device)
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        original_size = original_data_size[i]
        ranges = ranges_list_contribution[i]
        ranges_index = ranges_index_list_contribution[i].to(device)
        neighbors = neighbors_list_contribution[i]
        idxs = idxs_list_contribution[i]
        optimizer3.zero_grad()
        out = model3(data, ranges, ranges_index, original_size, neighbors, idxs)
        loss = F.nll_loss(out, data.y)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * original_size
        total_examples += original_size
        loss.backward()
        optimizer3.step()
        i += 1

    return total_loss / total_examples, total_correct / total_examples


def train():
    model1.train()

    total_loss = total_correct = total_examples = 0
    for data in train_data:
        data = data.to(device)
        optimizer1.zero_grad()
        out = model1(data, original_size=data.x.shape[0])
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer1.step()
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes

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
#
#
t3 = time.time()
max_acc1 = 0
for epoch in range(350):
    loss, acc = train()
    # train_acc, val_acc, test_acc = test()
    if acc > max_acc1:
        max_acc1 = acc
    print('Epoch: {:02d}, Loss: {:.4f}, acc: {:.4f}'.format(
        epoch, loss, acc))
t4 = time.time()
torch.cuda.empty_cache()
print("total training time: ", t4 - t3)

t5 = time.time()
max_acc2 = 0
for epoch in range(350):
    loss, acc = train_with_rr()
    if acc > max_acc2:
        max_acc2 = acc
    print('Epoch: {:02d}, Loss: {:.4f}, acc: {:.4f}'.format(
        epoch, loss, acc))
t6 = time.time()
torch.cuda.empty_cache()
print("total training time with rr: ", t6 - t5)

t7 = time.time()
max_acc3 = 0
for epoch in range(350):
    loss, acc = train_with_rr_contribution()
    if acc > max_acc3:
        max_acc3 = acc
    print('Epoch: {:02d}, Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, loss, acc))
t8 = time.time()
torch.cuda.empty_cache()
print("total training time with rr and contribution: ", t8 - t7)

print("acc compare:", max_acc1, max_acc2, max_acc3)

