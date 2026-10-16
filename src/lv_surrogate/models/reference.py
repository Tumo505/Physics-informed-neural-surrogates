"""Numerical adapter for released NumPy-valued Flax FrozenDict weights.

Restricted unpickler permits only NumPy arrays and FrozenDict -> dict.
Forward equations follow external/models.py; LayerNorm epsilon=1e-6.
This is a compatibility adapter, not a claim of original-environment replication.
"""
import pickle
import numpy as np

class WeightReader(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('flax.core.frozen_dict','FrozenDict'):return dict
  if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='_reconstruct':return np._core.multiarray._reconstruct
  if module=='numpy' and name in ('dtype','ndarray'):return getattr(np,name)
  raise pickle.UnpicklingError(f'Unapproved checkpoint class: {module}.{name}')

def read_weights(path):
 with open(path,'rb') as f:return WeightReader(f).load()

def mlp(x,p):
 n=sum(k.startswith('Dense_') for k in p)
 for i in range(n):
  d=p[f'Dense_{i}'];x=x@d['kernel']+d['bias']
  if i<n-1:x=np.tanh(x)
 if 'LayerNorm_0' in p:
  d=p['LayerNorm_0'];mean=x.mean(-1,keepdims=True);var=np.maximum(0,(x*x).mean(-1,keepdims=True)-mean*mean)
  x=(x-mean)/np.sqrt(var+1e-6)*d['scale']+d['bias']
 return x

def fixed_predict(theta,encoder,decoder,latents):
 z=mlp(theta,encoder['params']);rep=np.concatenate((np.broadcast_to(z,(len(latents),z.shape[-1])),latents),-1)
 return np.concatenate([mlp(rep,decoder['params'][f'FlaxMLP_{i}']) for i in range(3)],-1)
