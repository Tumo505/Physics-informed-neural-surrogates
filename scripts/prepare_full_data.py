"""Cache all author training cases; fit every transformation using train only."""
from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.sparse import csr_matrix
from lv_surrogate.data.dalton import Dalton
P=Path(__file__).resolve().parents[1]
OUT=P/'data/processed/full';OUT.mkdir(parents=True,exist_ok=True)
TOPO=P/'external/passive-lv-gnn-emul/data/lvDataFixedGeom/topologyData'
labels=np.load(TOPO/'kmeans-labels-list.npy',allow_pickle=True)
layers=np.load(TOPO/'node-layer-labels.npy');full_edges=np.load(TOPO/'sparse-topology.npy')
label1=np.asarray(labels[0],int);label2=np.asarray(labels[1],int);parents=label2[label1]
n1=int((layers==1).sum());n2=int((layers==2).sum())
p1=csr_matrix((1/np.bincount(label1)[label1],(label1,np.arange(len(label1)))),shape=(n1,len(label1)))
p2=csr_matrix((1/np.bincount(label2)[label2],(label2,np.arange(len(label2)))),shape=(n2,n1))
pool=(p2@p1).astype('float32');offset=int((layers<2).sum())
edges=full_edges[(layers[full_edges[:,0]]==2)&(layers[full_edges[:,1]]==2)]-offset
edges=np.concatenate((edges,edges[:,::-1]));np.save(OUT/'topology.npy',edges);np.save(OUT/'parents.npy',parents)
np.savez(OUT/'pool_csr.npz',data=pool.data,indices=pool.indices,indptr=pool.indptr,shape=pool.shape)
stats={};start=time.perf_counter()
for split in ['train','validation','test']:
 d=Dalton(P/'data/raw/lvData',split);n=len(d);arrays={
 'points':np.lib.format.open_memmap(OUT/f'{split}_points.npy','w+',dtype='float32',shape=(n,6784,13)),
 'coarse':np.lib.format.open_memmap(OUT/f'{split}_coarse.npy','w+',dtype='float32',shape=(n,n2,13)),
 'globals':np.lib.format.open_memmap(OUT/f'{split}_globals.npy','w+',dtype='float32',shape=(n,36)),
 'edges':np.lib.format.open_memmap(OUT/f'{split}_edges.npy','w+',dtype='float32',shape=(n,len(edges),4)),
 'targets':np.lib.format.open_memmap(OUT/f'{split}_targets.npy','w+',dtype='float32',shape=(n,6784,3))}
 for i in range(n):
  c=d[i];x=np.concatenate((c['real-node-coords'],c['real-node-features']),-1);coarse=pool@x;delta=coarse[edges[:,1],:3]-coarse[edges[:,0],:3]
  arrays['points'][i]=x;arrays['coarse'][i]=coarse;arrays['globals'][i]=np.r_[c['global-features'],c['shape-coeffs'][:32]];arrays['edges'][i]=np.c_[delta,np.linalg.norm(delta,axis=-1)];arrays['targets'][i]=c['real-node-displacement']
 if split=='train':
  for name,a in arrays.items():
   total=np.zeros(a.shape[-1],dtype='float64');square=total.copy();count=0
   for chunk in np.array_split(a,32):
    b=np.asarray(chunk,dtype='float64').reshape(-1,a.shape[-1]);total+=b.sum(0);square+=(b*b).sum(0);count+=len(b)
   mean=total/count;std=np.sqrt(np.maximum(square/count-mean**2,0));stats[name+'_mean']=mean.astype('float32');stats[name+'_std']=np.maximum(std,1e-6).astype('float32')
  np.savez(OUT/'normalizers.npz',**stats)
 for name,a in arrays.items():
  if name!='targets':
   for i in range(n):a[i]=(a[i]-stats[name+'_mean'])/stats[name+'_std']
  a.flush()
 print(split,n,'cached',flush=True)
manifest={'train':2250,'validation':150,'test':567,'normalizers_fit':'original train only, 2250 cases','coarse_nodes':n2,'coarse_directed_edges':len(edges),'full_output_nodes':6784,'source_topology_sha256':hashlib.sha256((TOPO/'sparse-topology.npy').read_bytes()).hexdigest(),'normalizers_sha256':hashlib.sha256((OUT/'normalizers.npz').read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-start}
(P/'experiments/manifests/full_data.json').write_text(json.dumps(manifest,indent=2));print(manifest)
