"""Rebuild only input-derived pooling/topology; no FE target labels are fitted."""
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans2
P=Path(__file__).resolve().parents[1];out=P/'data/processed/cardiac_geometry_ml';out.mkdir(exist_ok=True);base=np.load(P/'data/processed/cardiac/cohort.npz')
if (out/'cluster_labels.npy').exists() and (out/'topology.npy').exists():
 labels=np.load(out/'cluster_labels.npy');edges=np.load(out/'topology.npy');assert len(labels)==len(base['coords']) and np.bincount(labels,minlength=64).min()>0;assert edges.ndim==2 and edges.shape[1]==2;print('REUSED input-only64-node hierarchy',flush=True)
else:
 labels=kmeans2(base['coords'],64,iter=30,minit='++',seed=2027)[1];C=base['cells'];edges=np.concatenate([C[:,[i,j]] for i in range(4) for j in range(i+1,4)]);edges=labels[edges];edges=edges[edges[:,0]!=edges[:,1]];edges=np.unique(np.sort(edges,axis=1),axis=0);edges=np.r_[edges,edges[:,::-1]]
 for name,value in [('cluster_labels.npy',labels),('topology.npy',edges)]:
  if (out/name).exists():np.testing.assert_array_equal(np.load(out/name),value)
  else:np.save(out/name,value)
 print('CREATED input-only64-node hierarchy',flush=True)
