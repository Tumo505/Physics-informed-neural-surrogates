"""Geometry-conditioned graph encoder with train-only POD field decoder.

Shared nodal correspondence is verified by the cohort's exact affine maps.
No transfer to arbitrary mesh topology or anatomical shape families is implied.
"""
import torch
from torch import nn
from .multiscale import MultiscaleSurrogate

class GeometryPOD(nn.Module):
    def __init__(self,mean,basis,coefficient_scale):
        super().__init__()
        self.encoder=MultiscaleSurrogate(node_dim=9,global_dim=3,width=32,steps=2)
        # The point decoder is unused; remove it so parameter counts are honest.
        del self.encoder.decoder
        self.head=nn.Sequential(nn.Linear(35,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,len(basis)))
        for name,value in [('field_mean',mean),('basis',basis),('coefficient_scale',coefficient_scale)]:self.register_buffer(name,torch.as_tensor(value,dtype=torch.float64))
    def forward(self,coarse,global_features,edges,topology):
        context=self.encoder.encode(coarse,global_features,edges,topology).mean(1)
        c=self.head(torch.cat((context,global_features),-1))*self.coefficient_scale
        return self.field_mean+torch.einsum('br,rnc->bnc',c,self.basis)
