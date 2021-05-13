# coding: utf-8
"""
@author: Nelson Zhao
@date:   2021/5/13 10:25 PM
@email:  dutzhaoyeyu@163.com
"""

import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

import warnings

warnings.filterwarnings("ignore")


class AdaptiveBreadthLayer(nn.Module):
    def __init__(self, in_dim, h_dim, num_heads=1):
        super(AdaptiveBreadthLayer, self).__init__()
        self.gat = dgl.nn.GATConv(in_feats=in_dim,
                                  out_feats=h_dim,
                                  num_heads=num_heads,
                                  activation=torch.tanh)

    def forward(self, g, h):
        h = self.gat(g, h)
        h = h.mean(dim=1)
        return h


class AdaptiveDepthLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AdaptiveDepthLayer, self).__init__()
        # input gate
        self.input_gate = nn.Linear(in_dim, out_dim)
        # forget gate
        self.forget_gate = nn.Linear(in_dim, out_dim)
        # output gate
        self.output_gate = nn.Linear(in_dim, out_dim)
        # state C
        self.state = nn.Linear(in_dim, out_dim)

    def forward(self, c, h):
        # input gate
        i = torch.sigmoid(self.input_gate(h))
        f = torch.sigmoid(self.forget_gate(h))
        o = torch.sigmoid(self.output_gate(h))
        c_tilde = torch.tanh(self.state(h))

        c = f * c + i * c_tilde
        h = o * torch.tanh(c)

        return c, h


class GeniePath(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, depth, lazy_mode=True):
        """
        @in_dim: 输入维度
        @h_dim: 隐藏层的维度
        @out_dim: 输出维度
        @depth: 迭代深度
        """
        super(GeniePath, self).__init__()
        self.depth = depth
        self.lazy_mode = lazy_mode

        self.wx = nn.Linear(in_dim, h_dim)

        self.breath_fn = torch.nn.ModuleList()
        self.depth_fn = torch.nn.ModuleList()
        for _ in range(depth):
            self.breath_fn.append(AdaptiveBreadthLayer(h_dim, h_dim))
            if not self.lazy_mode:
                self.depth_fn.append(AdaptiveDepthLayer(h_dim, h_dim))
            else:
                self.depth_fn.append(AdaptiveDepthLayer(2 * h_dim, h_dim))

        # 输出层
        self.out_layer = nn.Linear(h_dim, out_dim)

    def forward(self, g, x):
        h0 = self.wx(x)
        h = h0

        # standard模式
        if not self.lazy_mode:
            c = torch.zeros_like(h)
            for i in range(self.depth):
                h = self.breath_fn[i](g, h)
                c, h = self.depth_fn[i](c, h)

            out = torch.relu(self.out_layer(h))
            return out

        # lazy模式
        else:
            collector = []
            for i in range(self.depth):
                h = self.breath_fn[i](g, h)
                collector.append(h)

            mu = h0
            c = torch.zeros_like(mu)
            for i in range(self.depth):
                h_mu = torch.cat([collector[i], mu], dim=1)
                c, mu = self.depth_fn[i](c, h_mu)

            out = torch.relu(self.out_layer(mu))
            return out
