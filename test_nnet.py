import numpy as np
from acasxu_smtverifier_helper.NNet.python.nnet import NNet


import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNet(nn.Module):
    def __init__(self, net):
        self.mins = net.mins
        self.maxes = net.maxes
        self.means = net.means
        self.ranges = net.ranges

        weights = net.weights
        biases = net.biases
        super().__init__()
        layers = []
        for w, b in zip(weights[:-1], biases[:-1]):
            in_features = w.shape[1]
            out_features = w.shape[0]
            layer = nn.Linear(in_features, out_features)
            layer.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32))
            layer.bias = nn.Parameter(torch.tensor(b, dtype=torch.float32))
            layers.append(layer)
            layers.append(nn.ReLU())

        # last layer (no ReLU)
        w, b = weights[-1], biases[-1]
        layer = nn.Linear(w.shape[1], w.shape[0])
        layer.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        layer.bias = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        layers.append(layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)





nn0 = NNet("/home/lucav/thesis/ACASXu-20251018T095242Z-1-001/ACASXu/rectangular-coordinates/networks/tmp/HCAS_rect_v6_pra0_tau00_25HU_3000.nnet")
nnout = nn0.evaluate_network([40000.0,40000.0,1.0])
input_norms = [0.35963060758035714, 0.35714285714285715, 0.15915495087284556]
seq = FeedForwardNet(nn0)
with torch.no_grad():
    outputs = seq.net(torch.tensor(input_norms))
outputs = outputs * nn0.ranges[-1] + nn0.means[-1]
# print(model)
#
# [(40000.0 - (-278.628049)) /112000.0,
# (40000.0 - 0.0) /112000.0,
# (1.0 - 0.0) / 6.283185]
#
# lb=-56000,ub=56000
#
# (x - (-278.628049)) /112000.0
# (y - 0.0) /112000.0
# (psi - 0.0) / 6.283185