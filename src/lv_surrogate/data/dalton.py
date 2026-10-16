"""Lazy raw Dalton case access. Original topology is graph edges, never FE elements."""
from pathlib import Path
import numpy as np

class Dalton:
 def __init__(self,root,split):
  self.root=Path(root);self.split=split;self.base=self.root/'rawData'/split
  self.arrays={k:np.load(self.base/f'{k}.npy',mmap_mode='r') for k in ['real-node-coords','real-node-features','global-features','shape-coeffs','real-node-displacement']}
  self.edges=np.load(self.root/'topologyData/real-node-topology.npy').astype(np.int64)
 def __len__(self):return len(self.arrays['real-node-displacement'])
 def __getitem__(self,i):return {k:np.array(a[i if len(a)>1 else 0],dtype=np.float32) for k,a in self.arrays.items()}

class AuthorPreprocess:
 """Use released clustering and summary statistics without refitting on evaluation data."""
 def __init__(self,fixed_root):
  r=Path(fixed_root);self.stats=r/'normalisationStatistics';t=r/'topologyData'
  self.layers=np.load(t/'node-layer-labels.npy');self.edges=np.load(t/'sparse-topology.npy').astype(np.int64)
  self.labels=np.load(t/'kmeans-labels-list.npy',allow_pickle=True)
 def norm(self,a,name):
  mean=np.load(self.stats/f'{name}-mean.npy');std=np.load(self.stats/f'{name}-std.npy');assert np.all(std>0)
  return ((a-mean)/std).astype('float32')
 def __call__(self,case):
  coords=case['real-node-coords'];feat=case['real-node-features'];C=[coords];V=[feat]
  for layer,labels in enumerate(self.labels):
   labels=np.array(labels,dtype=int);n=int((self.layers==layer+1).sum());count=np.bincount(labels,minlength=n)[:,None]
   assert np.all(count>0)
   def agg(x):
    out=np.zeros((n,x.shape[-1]),dtype='float32');np.add.at(out,labels,x);return out/count
   C.append(agg(C[-1]));V.append(agg(V[-1]))
  c=np.concatenate(C).astype('float32');v=np.concatenate(V).astype('float32');s,r=self.edges.T
  delta=c[r]-c[s];e=np.concatenate((delta,np.linalg.norm(delta,axis=-1,keepdims=True)),-1)
  return self.norm(v,'augmented-node-features'),self.norm(e,'edge-features'),self.norm(case['global-features'][None],'global-features'),self.norm(case['shape-coeffs'][None],'shape-coeffs')[0,:32]
