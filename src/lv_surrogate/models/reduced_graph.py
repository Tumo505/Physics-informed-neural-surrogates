"""Fixed-geometry graph encoder and train-only POD full-field decoder.

Only load response is assessed; no unseen-geometry claim is supported.
"""
import torch
from torch import nn
from .message_passing import GraphSurrogate

class ReducedGraphSurrogate(nn.Module):
    def __init__(self, mean, basis, coefficient_scale, node_features, edge_features, topology):
        super().__init__()
        rank=basis.shape[0]
        self.encoder=GraphSurrogate(node_features.shape[-1],1,width=16,steps=2)
        self.head=nn.Sequential(nn.Linear(4,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,rank))
        for name,value in [('field_mean',mean),('basis',basis),('coefficient_scale',coefficient_scale),('node_features',node_features),('edge_features',edge_features),('topology',topology)]:self.register_buffer(name,value)

    def forward(self, pressures):
        # Fixed geometry encoded once per batch, trainable graph context.
        context=self.encoder(self.node_features,self.node_features.new_zeros(1),self.edge_features,self.topology).mean(0)
        coefficients=self.head(torch.cat((pressures[:,None],context[None].expand(len(pressures),-1)),1))*self.coefficient_scale
        return self.field_mean+torch.einsum('br,rnc->bnc',coefficients,self.basis)
