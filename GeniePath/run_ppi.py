# coding: utf-8
"""
@author: Nelson Zhao
@date:   2021/5/13 10:28 PM
@email:  dutzhaoyeyu@163.com
"""

import torch
from dgl.data import PPIDataset
from sklearn.metrics import f1_score

data = PPIDataset(mode="train")
val_data = PPIDataset(mode="valid")
test_data = PPIDataset(mode="test")
g = data.graph

loss_op = torch.nn.BCEWithLogitsLoss()

# hyper-parameters
in_dim = data.features.shape[-1]
h_dim = 8
out_dim = data.labels.shape[-1]
depth = 3
lazy = True

net = GeniePath(in_dim, h_dim, out_dim, depth, lazy)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)

# main loop
dur = []
epoch_losses = []
# best_f1 = 0
# best_epoch = 0
for epoch in range(5000):
    logits = net(g, torch.FloatTensor(data.features))
    torch.nn.BCEWithLogitsLoss()
    loss = loss_op(logits, torch.FloatTensor(data.labels))
    epoch_losses.append(loss)

    # Compute prediction
    pred_val = net(val_data.graph, torch.FloatTensor(val_data.features))
    pred_val = (pred_val.detach().numpy() > 0).astype(int)

    micro_f1_val = f1_score(val_data.labels, pred_val, average='micro')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch {:05d} | Loss {:.4f} | Val: {:.4f} | Test: {:.4f}".format(
        epoch, loss.item(), micro_f1_val, 0))