import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


parser = argparse.ArgumentParser(description='Train a graph classification model.')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--pr', type=float, default=0.8)
parser.add_argument('--batch-size', type=int, default=60)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()


dataset = TUDataset("data/PROTEINS", name='PROTEINS')
dataset = dataset.shuffle()
n = len(dataset) // 10
test_dataset = dataset[:n]
train_dataset = dataset[n:]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)


class Net(torch.nn.Module):
    def __init__(self, hidden_channels=128, dropout=0.5, pooling_ratio=0.8):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GraphConv(dataset.num_features, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=pooling_ratio)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=pooling_ratio)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.pool3 = TopKPooling(hidden_channels, ratio=pooling_ratio)

        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(args.hidden, args.dropout, args.pr).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, args.epochs + 1):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, '
          f'Test Acc: {test_acc:.5f}')
