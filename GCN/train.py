import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Reddit, PPI
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.loader import ClusterData, ClusterLoader
import time
import copy

import sys
sys.path.append(r'..\hag')
sys.path.append(r'..\GCN')
from ACTPY import obtain_aggregated_gnn
from gcn import GCN, GraphSAGE, GAT, GCNRR, GraphSAGERR
from threshold import get_attribution, get_contribution, get_rest_edge

from sklearn.metrics import f1_score


# load the data and do the aggregation of repeated nodes
path = '../data/ppi'
dataset = PPI(path, split='train')
print(dataset)
train_loader = []
for i, data in enumerate(dataset):
    if i == 9:
        train_loader.append(data)
        print(i, data, max(data.edge_index[0]))
# train_dataset = Reddit(path)
# val_dataset = PPI(path, split='val')
# test_dataset = PPI(path, split='test')
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# for graphsaint/gcn
# train_loader = GraphSAINTRandomWalkSampler(dataset[0], batch_size=1000, walk_length=2,
#                                      num_steps=5, sample_coverage=100,
#                                      save_dir='../data/reddit/')

# subgraph_loader = NeighborLoader(copy.copy(dataset), input_nodes=None,
#                                  num_neighbors=[-1], shuffle=False, batch_size=16)

# No need to maintain these features during evaluation:
# del subgraph_loader.data.x, subgraph_loader.data.y
# # Add global node index information.
# subgraph_loader.data.num_nodes = dataset.num_nodes
# subgraph_loader.data.n_id = torch.arange(dataset.num_nodes)

# for graphsage
# train_loader = NeighborLoader(dataset, num_neighbors=[25, 10], batch_size=3, directed=False)

# for clustergcn
# cluster_data = ClusterData(dataset[0], num_parts=1500, recursive=False,
#                            save_dir='../data/reddit/')
# train_loader = ClusterLoader(cluster_data, batch_size=5, shuffle=True)

# for fastgcn


# train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
# record the processed data, original data size and processing time
aggregated_train_data1 = []
aggregated_train_data2 = []
aggregated_nodes1 = []
aggregated_nodes2 = []
train_data = []
original_data_size = []
i = 0
redundancy_time1_set = []
redundancy_time2_set = []
total_time1_set = []
total_time2_set = []
aggregate_acc_set = []
for data in train_loader:
    i += 1
    print(i)
    original_data = data.clone()
    original_size = data.x.shape[0]
    # only for GraphSAGE
    # edge_index0 = data.edge_index[1]
    # edge_index1 = data.edge_index[0]
    # data.edge_index = torch.stack((edge_index0, edge_index1), 0)

    train_data.append(original_data)
    print(data.x.shape, data.edge_index.shape, data.y.shape[0])
    # obtain the processed data using function 'obtain_aggregated_gnn'
    aggregated_data1, aggregated_data2, aggregated_nodes_set1, aggregated_nodes_set2, redundancy_time1, redundancy_time2, total_time1, total_time2, aggregate_acc = obtain_aggregated_gnn(data, data.x.shape[0]/4)
    redundancy_time1_set.append(redundancy_time1)
    redundancy_time2_set.append(redundancy_time2)
    aggregated_nodes1.append(aggregated_nodes_set1)
    aggregated_nodes2.append(aggregated_nodes_set2)
    total_time1_set.append(total_time1)
    total_time2_set.append(total_time2)
    aggregate_acc_set.append(aggregate_acc)
    print(aggregated_data1.x.shape, aggregated_data1.edge_index.shape)
    print(aggregated_data2.x.shape, aggregated_data2.edge_index.shape)
    aggregated_train_data1.append(aggregated_data1)
    aggregated_train_data2.append(aggregated_data2)
    original_data_size.append(original_size)
    if i == 20:
        break

redundancy_time1_avg = sum(redundancy_time1_set)/len(redundancy_time1_set)
redundancy_time2_avg = sum(redundancy_time2_set)/len(redundancy_time2_set)
total_time1_avg = sum(total_time1_set)/len(total_time1_set)
total_time2_avg = sum(total_time2_set)/len(total_time2_set)
aggregate_acc_avg = sum(aggregate_acc_set)/len(aggregate_acc_set)
print(redundancy_time1_avg, redundancy_time2_avg, total_time1_avg, total_time2_avg, aggregate_acc_avg)

device = torch.device('cuda:0')
print(data.y.shape[0])
model1 = GCN(data.x.shape[1], hidden_channels=128, out_channels=dataset.num_classes).to(device)
model2 = GCN(data.x.shape[1], hidden_channels=128, out_channels=dataset.num_classes).to(device)
model3 = GCN(data.x.shape[1], hidden_channels=128, out_channels=dataset.num_classes).to(device)
print("Model has been transferred to GPU")
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=5e-4)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.01, weight_decay=5e-4)
loss_op = torch.nn.BCEWithLogitsLoss()

def train_with_rr():
    model3.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    ys, preds = [], []

    for data in aggregated_train_data1:
        data = data.to(device)
        original_size = original_data_size[i]
        optimizer3.zero_grad()
        out = model3(data, original_size)
        loss = loss_op(out, data.y.float())
        # ys.append(data.y)
        # preds.append((out > 0).float())
        # total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
        loss.backward()
        optimizer3.step()
        i += 1

    # y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    # f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return total_loss / total_examples, 0


def train_with_rr_contribution():
    model2.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    ys, preds = [], []

    for data in aggregated_train_data2:
        data = data.to(device)
        original_size = original_data_size[i]
        optimizer2.zero_grad()
        out = model2(data, original_size)
        loss = loss_op(out, data.y.float())
        # ys.append(data.y)
        # preds.append((out > 0).float())
        # total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
        loss.backward()
        optimizer2.step()
        i += 1

    # y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    # f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return total_loss / total_examples, 0

def train():
    model1.train()

    total_loss = total_correct = total_examples = 0
    i = 0
    ys, preds = [], []

    for data in train_data:
        data = data.to(device)
        original_size = original_data_size[i]
        optimizer1.zero_grad()
        out = model1(data, original_size)
        loss = loss_op(out, data.y.float())
        # ys.append(data.y)
        # preds.append((out > 0).float())
        # total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
        loss.backward()
        optimizer1.step()
        i += 1

    # y, pred = torch.cat(ys, dim=0).cpu().numpy(), torch.cat(preds, dim=0).cpu().numpy()
    # f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return total_loss / total_examples, 0


# test for ppi dataset
# def test(loader):
#     model.eval()
#
#     ys, preds = [], []
#     for data in loader:
#         ys.append(data.y)
#         out = model(data.to(device), original_size=data.x.shape[0])
#         preds.append((out > 0).float().cpu())
#
#     y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
#     print(y, pred)
#     return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

# test for reddit dataset
# def test():
#     model.eval()
#     logits, accs = model.inference(dataset.x, subgraph_loader, device), []
#     y = dataset.y.to(logits.device)
#     for _, mask in dataset('val_mask', 'test_mask'):
#         accs.append(int((logits[mask] == y[mask]).sum()) / int(mask.sum()))
#     return accs


t3 = time.time()
max_acc1 = 0
for epoch in range(350):
    loss, acc = train()
    # train_acc, val_acc, test_acc = test()
    if acc > max_acc1:
        max_acc1 = acc
    print('Epoch: {:02d}, Loss: {:.4f}, train: {:.4f}'.format(
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
    print('Epoch: {:02d}, Loss: {:.4f}, Acc: {:.4f}'.format(
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

print("time compare: ", t4 - t3, t6 - t5, t8 - t7)
print("acc compare:", max_acc1, max_acc2, max_acc3)
