"""Build finer author-clustering graph inputs; reuse immutable original node arrays."""
from pathlib import Path
import json,os,hashlib
import numpy as np
from scipy.sparse import csr_matrix
P=Path(__file__).resolve().parents[1];base=P/'data/processed/full';out=P/'data/processed/full_level1';out.mkdir(exist_ok=True)
topo=P/'external/passive-lv-gnn-emul/data/lvDataFixedGeom/topologyData'
labels=np.load(topo/'kmeans-labels-list.npy',allow_pickle=True);parents=np.asarray(labels[0],int)
layers=np.load(topo/'node-layer-labels.npy');e=np.load(topo/'sparse-topology.npy');n=int((layers==1).sum());offset=int((layers<1).sum())
edges=e[(layers[e[:,0]]==1)&(layers[e[:,1]]==1)]-offset;edges=np.concatenate([edges,edges[:,::-1]])
pool=csr_matrix((1/np.bincount(parents)[parents],(parents,np.arange(len(parents)))),shape=(n,len(parents))).astype('float32')
np.save(out/'parents.npy',parents);np.save(out/'topology.npy',edges)
stats=dict(np.load(base/'normalizers.npz'))
for split in ['train','validation','test']:
 for key in ['points','globals','targets']:
  dst=out/f'{split}_{key}.npy'
  if not dst.exists():os.link(base/dst.name,dst)
 pts=np.load(base/f'{split}_points.npy',mmap_mode='r');count=len(pts)
 coarse=np.lib.format.open_memmap(out/f'{split}_coarse.npy',mode='w+',dtype='float32',shape=(count,n,13))
 edge=np.lib.format.open_memmap(out/f'{split}_edges.npy',mode='w+',dtype='float32',shape=(count,len(edges),4))
 for i in range(count):
  raw=pts[i]*stats['points_std']+stats['points_mean'];c=pool@raw;delta=c[edges[:,1],:3]-c[edges[:,0],:3]
  coarse[i]=c;edge[i]=np.c_[delta,np.linalg.norm(delta,axis=-1)]
 if split=='train':
  for name,a in [('coarse',coarse),('edges',edge)]:
   total=np.zeros(a.shape[-1]);sq=total.copy();num=0
   for chunk in np.array_split(a,64):
    b=np.array(chunk,dtype='float64').reshape(-1,a.shape[-1]);total+=b.sum(0);sq+=(b*b).sum(0);num+=len(b)
   mean=total/num;stats[name+'_mean']=mean.astype('float32');stats[name+'_std']=np.maximum(np.sqrt(np.maximum(sq/num-mean**2,0)),1e-6).astype('float32')
  np.savez(out/'normalizers.npz',**stats)
 for name,a in [('coarse',coarse),('edges',edge)]:
  for i in range(count):a[i]=(a[i]-stats[name+'_mean'])/stats[name+'_std']
  a.flush()
 print(split,count,n,len(edges),flush=True)
(P/'experiments/manifests/fine_graph_data.json').write_text(json.dumps({'coarse_nodes':n,'directed_edges':len(edges),'normalization_fit':'original training2250 only','shared_arrays':'hardlinks to immutable normalized points, globals and physical targets; do not mutate','normalizers_sha256':hashlib.sha256((out/'normalizers.npz').read_bytes()).hexdigest()},indent=2))
