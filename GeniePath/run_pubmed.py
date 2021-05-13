# coding: utf-8
"""
@author: Nelson Zhao
@date:   2021/5/13 10:28 PM
@email:  dutzhaoyeyu@163.com
"""

import torch
import torch.nn.functional as F
from dgl.data import PubmedGraphDataset
from dgl import DGLGraph

from .geniepath import GeniePath

pub_data = PubmedGraphDataset()

features = torch.FloatTensor(pub_data.features)
labels = torch.LongTensor(pub_data.labels)
train_mask = torch.BoolTensor(pub_data.train_mask)
val_mask = torch.BoolTensor(pub_data.val_mask)
test_mask = torch.BoolTensor(pub_data.test_mask)
g = DGLGraph(pub_data.graph)
g = g.add_self_loop()

# hyper-parameters
in_dim = features.size()[-1]
h_dim = 8
out_dim = pub_data.num_classes
depth = 3
lazy = True

# build model
net = GeniePath(in_dim, h_dim, out_dim, depth, lazy)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)

# main loop
dur = []
epoch_losses = []
best_acc = 0
best_epoch = 0

for epoch in range(100):

    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])
    epoch_losses.append(loss)

    # Compute prediction
    pred = logits.argmax(1)

    # Compute accuracy on training/validation/test
    train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
    val_acc = (pred[val_mask] == labels[val_mask]).float().mean()

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch {:05d} | Loss {:.4f} | Train: {:.4f} | Val: {:.4f}".format(
        epoch, loss.item(), train_acc, val_acc))


# test
