"""Geometry-separated preprocessing and POD fit using TRAINING fields only."""
from pathlib import Path
import json,sys,hashlib,argparse
import numpy as np
from scipy.cluster.vq import kmeans2
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['all','training','evaluation'],default='all');args=ap.parse_args()
manifest=json.loads((P/'experiments/manifests/cardiac_geometry_v1.json').read_text())
if not manifest['complete'] and args.phase!='training':raise RuntimeError('Geometry cohort incomplete')
cfg=manifest['config'];root=P/'data/processed'/cfg['name'];out=P/'data/processed/cardiac_geometry_ml';out.mkdir(exist_ok=True)
base=np.load(P/'data/processed/cardiac/cohort.npz');X=base['coords']
_,labels=kmeans2(X,64,iter=30,minit='++',seed=2027)
np.save(out/'cluster_labels.npy',labels)
cells=base['cells'];edge=np.concatenate([cells[:,[i,j]] for i in range(4) for j in range(i+1,4)]);edge=labels[edge];edge=edge[edge[:,0]!=edge[:,1]];edge=np.unique(np.sort(edge,axis=1),axis=0);edge=np.r_[edge,edge[:,::-1]]
np.save(out/'topology.npy',edge);counts=np.bincount(labels,minlength=64)
def pool(x):
 z=np.zeros((64,x.shape[1]));np.add.at(z,labels,x);return z/counts[:,None]
prepared={};rank=12
if args.phase=='evaluation':prepared=json.loads((P/'experiments/manifests/geometry_ml.json').read_text())['splits']
if args.phase=='evaluation':stats=dict(np.load(out/'normalizers.npz'))
for split,(lo,hi) in cfg['splits'].items():
 if args.phase=='training' and split not in ['train','validation']:continue
 if args.phase=='evaluation' and split in ['train','validation']:continue
 values={k:[] for k in ['coarse','globals','edges','targets','geometry_ids','load_ids','pressures']}
 for g in range(lo,hi):
  with np.load(root/f'geometry_{g:03d}.npz') as a:
   c=pool(np.c_[a['coords'],a['fiber'],a['sheet']]);delta=c[edge[:,1],:3]-c[edge[:,0],:3];ef=np.c_[delta,np.linalg.norm(delta,axis=-1)]
   for j,p in enumerate(a['pressures']):
    for k,v in [('coarse',c),('edges',ef),('globals',np.r_[a['geometry_scales'],p]),('targets',a['displacements'][j]),('geometry_ids',g),('load_ids',j),('pressures',p)]:values[k].append(v)
 arrays={k:np.array(v) for k,v in values.items()}
 if split=='train':
  stats={}
  for k in ['coarse','edges','globals']:
   flat=arrays[k].reshape(-1,arrays[k].shape[-1]);stats[k+'_mean']=flat.mean(0);stats[k+'_std']=np.maximum(flat.std(0),1e-10)
  mean=arrays['targets'].mean(0);_,singular,basis=np.linalg.svd((arrays['targets']-mean).reshape(len(arrays['targets']),-1),full_matrices=False);rank=12;basis=basis[:rank].reshape(rank,*mean.shape);coeff=np.einsum('bnc,rnc->br',arrays['targets']-mean,basis);stats.update(field_mean=mean,basis=basis,coefficient_scale=np.maximum(coeff.std(0),1e-12),u_scale=np.sqrt(np.mean((arrays['targets']-mean)**2)),singular_values=singular)
  np.savez(out/'normalizers.npz',**stats)
 for k in ['coarse','edges','globals']:arrays[k]=(arrays[k]-stats[k+'_mean'])/stats[k+'_std']
 np.savez_compressed(out/f'{split}.npz',**arrays);prepared[split]={'geometry_ids':[lo,hi],'case_count':len(arrays['targets'])}
(P/'experiments/manifests/geometry_ml.json').write_text(json.dumps({'splits':prepared,'normalizers_fit':'training20 geometries/80 startup load cases only','pod_rank':rank,'coarse_nodes':64,'directed_edges':len(edge),'normalizer_sha256':hashlib.sha256((out/'normalizers.npz').read_bytes()).hexdigest(),'node_correspondence':'exact source affine mapping; not an assumption across unrelated anatomies','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print('GEOMETRY ML DATA READY',prepared,flush=True)
