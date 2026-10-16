"""Causal MLP/POD acceleration-increment model; fixed geometry only."""
import torch
from torch import nn

class TemporalPOD(nn.Module):
    def __init__(self,stats):
        super().__init__()
        for k in ['basis','delta_mean','coefficient_scale','feature_mean','feature_std']:
            self.register_buffer(k,torch.as_tensor(stats[k],dtype=torch.float64))
        rank=len(self.basis)
        self.net=nn.Sequential(nn.Linear(3*rank+1,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,rank))
    def forward(self,features):
        coefficients=self.net((features-self.feature_mean)/self.feature_std)*self.coefficient_scale
        return self.delta_mean+coefficients@self.basis
