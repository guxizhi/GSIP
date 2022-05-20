import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Reddit, PPI
from torch_geometric.loader import DataLoader
import time
from torch_scatter import segment_coo, segment_csr
from gcnconhag import GCNConvHAG

import sys
sys.path.append(r'..\hag')



class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 256, heads=4)
        self.lin1 = torch.nn.Linear(in_channels, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, out_channels, heads=6,
                             concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, out_channels)

    def forward(self, data, original_size):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x[:original_size]


class GCN(torch.nn.Module):
    '''
    a gcn with two gcn layers and a layer of softmax
    '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=True))

    def forward(self, data, original_size):
        x, edge_index = data.x, data.edge_index

        t1 = time.time()
        for i, conv in enumerate(self.convs):

            # for j, tensor_X in enumerate(x):
            #     print(j, tensor_X[0])

            x = conv(x, edge_index)
            # if i < len(self.convs) - 1:
            #     x = x.relu_()
            #     x = F.dropout(x, training=self.training)
        # x = F.log_softmax(x, dim=1)
        t2 = time.time()
        # print("each epoch training time: ", t2 - t1)

        return x[:original_size]


class GCNRR(torch.nn.Module):
    '''
    a gcn with two gcn layers and a layer of softmax
    '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=True))

    def forward(self, data, aggregated_nodes_set, original_size):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):

            # aggregate intermediate results
            # for aggregated_node, aggregated_pair in aggregated_nodes_set:
            #     # print(aggregated_node, aggregated_pair)
            #     x[aggregated_node] = x[aggregated_pair[0]] + x[aggregated_pair[1]]
            #
            # for j, tensor_X in enumerate(x):
            #     print(j, tensor_X[0])
            #     if j == (original_size-1):
            #         print("aggregated embedding: ")

            x = conv(x, edge_index)
            # if i < len(self.convs) - 1:
            #     x = x.relu_()
            #     x = F.dropout(x, training=self.training)
        # x = F.log_softmax(x, dim=1)

        return x

class GCNHAG(torch.nn.Module):
    '''
    a gcn with two gcn layers and a layer of softmax
    '''

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConvHAG(in_channels, hidden_channels, normalize=True))
        self.convs.append(GCNConvHAG(hidden_channels, out_channels, normalize=True))

    def forward(self, data, original_size):
        x, edge_index = data.x, data.edge_index

        # print("ranges:{}, range_index:{}" .format(ranges, range_index))
        t1 = time.time()
        for i, conv in enumerate(self.convs):

            x = conv(x, edge_index)

            # if i < len(self.convs) - 1:
            #     x = x.relu_()
            #     x = F.dropout(x, training=self.training)
        x = F.log_softmax(x, dim=1)
        t2 = time.time()
        # print("each epoch training time: ", t2 - t1)

        return x[:original_size]


    # @torch.no_grad()
    # def inference(self, x_all, subgraph_loader, device):
    #     for i, conv in enumerate(self.convs):
    #         xs = []
    #         for batch in subgraph_loader:
    #             x = x_all[batch.n_id].to(device)
    #             x = conv(x, batch.edge_index.to(device))
    #             if i < len(self.convs) - 1:
    #                 x = x.relu_()
    #             xs.append(x[:batch.batch_size].cpu())
    #         x_all = torch.cat(xs, dim=0)
    #     return x_all


class GraphSAGERR(torch.nn.Module):
    '''
    a GraphSAGE with defined sageconv layer, relu and dropout layer
    '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGERR, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(2-1):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, data, aggregated_nodes_set, original_size):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # aggregate intermediate results
            # for aggregated_node, aggregated_pair in aggregated_nodes_set:
            #     # print(aggregated_node, aggregated_pair)
            #     x[aggregated_node] = x[aggregated_pair[0]] + x[aggregated_pair[1]]

            # if i != 1:
            #     x = x.relu()
            #     x = F.dropout(x, p=0.5, training=self.training)
        # x = F.log_softmax(x, dim=1)
        return x[:original_size]


class GraphSAGE(torch.nn.Module):
    '''
    a GraphSAGE with defined sageconv layer, relu and dropout layer
    '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(2-1):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, data, original_size):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # if i != 1:
            #     x = x.relu()
            #     x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(x, dim=1)
        return x[:original_size]


class GraphSAGEHAG(torch.nn.Module):
    '''
    a GraphSAGE with defined sageconv layer, relu and dropout layer
    '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(2-1):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, data, ranges, original_size, original_edge, neighbors, newids, idxs):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):

            # aggregate intermediate results
            for range in ranges:
                begin, end = range[0], range[1]
                for aggregated_node in newids[begin: end + 1]:
                    neigh = edge_index[1][
                            neighbors[idxs.index(aggregated_node)]:neighbors[idxs.index(aggregated_node) + 1]]
                    x[aggregated_node] = x[neigh[0]] + x[neigh[1]]

            x = conv(x, edge_index)
            # if i != 1:
            #     x = x.relu()
            #     x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(x, dim=1)
        return x[:original_size]

