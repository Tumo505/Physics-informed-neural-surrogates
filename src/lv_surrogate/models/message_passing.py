"""Small supervised graph surrogate; native PyTorch index_add avoids extension dependency."""
import torch
from torch import nn

def mlp(din,dout,width):return nn.Sequential(nn.Linear(din,width),nn.SiLU(),nn.Linear(width,dout))
class GraphSurrogate(nn.Module):
 def __init__(self,node_dim,global_dim,width=32,steps=2):
  super().__init__();self.node=mlp(node_dim+global_dim,width,width);self.edge=mlp(4,width,width)
  self.messages=nn.ModuleList([mlp(3*width,width,width) for _ in range(steps)])
  self.updates=nn.ModuleList([mlp(2*width,width,width) for _ in range(steps)])
  self.decoder=mlp(width,3,width)
 def forward(self,x,g,edge,topology):
  s,r=topology.T;v=self.node(torch.cat((x,g.expand(len(x),-1)),dim=-1));e=self.edge(edge)
  degree=torch.bincount(r,minlength=len(x)).clamp_min(1).to(x.dtype)[:,None]
  for msg,upd in zip(self.messages,self.updates):
   m=msg(torch.cat((e,v[s],v[r]),-1));agg=torch.zeros_like(v).index_add(0,r,m)/degree
   v=v+upd(torch.cat((v,agg),-1));e=e+m
  return self.decoder(v)
