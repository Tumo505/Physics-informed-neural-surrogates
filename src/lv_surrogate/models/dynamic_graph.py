"""Bounded causal graph/POD recurrence, trained on its own complete histories.

Bounded coefficients prevent unbounded displacement drift, but do not enforce
positive element Jacobians or equilibrium; both require independent evaluation.
"""
import torch
from torch import nn
from .multiscale import MultiscaleSurrogate

class DynamicGraph(nn.Module):
    def __init__(self,stats,width=64):
        super().__init__()
        self.encoder=MultiscaleSurrogate(node_dim=9,global_dim=3,width=32,steps=2)
        del self.encoder.decoder
        for name in ['basis','coefficient_scale','coefficient_bound']:
            self.register_buffer(name,torch.as_tensor(stats[name],dtype=torch.float32))
        rank=len(self.basis)
        self.cell=nn.GRUCell(32+rank+2,width)
        self.output=nn.Linear(width,rank)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,coarse,globals,edges,topology,pressure):
        # Geometry context is computed once, with no target or future load input.
        context=self.encoder.encode(coarse,globals,edges,topology).mean(1)
        hidden=pressure.new_zeros((len(pressure),self.cell.hidden_size))
        q=pressure.new_zeros((len(pressure),len(self.basis)));previous=pressure.new_zeros(len(pressure));states=[]
        for p in pressure.unbind(1):
            hidden=self.cell(torch.cat((context,q,p[:,None],previous[:,None]),-1),hidden)
            q=self.coefficient_bound*torch.tanh(self.output(hidden));states.append(q);previous=p
        return torch.stack(states,1)
    def decode(self,q):return (q*self.coefficient_scale)@self.basis

def newmark_operators(n,dt):
    """Linear causal maps from displacement coefficients to source-scheme histories."""
    import numpy as np
    eye=np.eye(n);old=np.zeros(n);v=np.zeros(n);a=np.zeros(n);vs=[];acs=[]
    for row in eye:
        anew=(row-old-dt*v-(.5-.36)*dt**2*a)/(.36*dt**2)
        v=v+dt*(.3*a+.7*anew);a=anew;old=row
        vs.append(v.copy());acs.append(a.copy())
    return np.array(vs),np.array(acs)
