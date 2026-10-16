"""Transform fresh numerical geometries using immutable original training statistics."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
P=Path(__file__).resolve().parents[1]
source=P/'data/processed/cardiac_geometry_ml';out=P/'data/processed/cardiac_confirmation_ml';out.mkdir(exist_ok=True)
m=json.loads((P/'experiments/manifests/cardiac_confirmation_v1.json').read_text())
if not m['complete']:raise RuntimeError('Incomplete confirmation cohort')
original=json.loads((P/'experiments/manifests/geometry_ml.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if sha(source/'normalizers.npz')!=original['normalizer_sha256']:raise RuntimeError('Training normalizers changed')
for name in ['normalizers.npz','cluster_labels.npy','topology.npy']:
 if (out/name).exists() and sha(out/name)!=sha(source/name):raise RuntimeError('Immutable transformation changed')
 if not (out/name).exists():shutil.copyfile(source/name,out/name)
stats=dict(np.load(source/'normalizers.npz'));labels=np.load(source/'cluster_labels.npy');edges=np.load(source/'topology.npy');counts=np.bincount(labels,minlength=64)
def pool(x):
 z=np.zeros((64,x.shape[1]));np.add.at(z,labels,x);return z/counts[:,None]
splits={}
for split,(lo,hi) in m['config']['splits'].items():
 values={k:[] for k in ['coarse','globals','edges','targets','geometry_ids','load_ids','pressures']}
 for g in range(lo,hi):
  with np.load(P/f'data/processed/cardiac_confirmation_v1/geometry_{g:03d}.npz') as a:
   c=pool(np.c_[a['coords'],a['fiber'],a['sheet']]);delta=c[edges[:,1],:3]-c[edges[:,0],:3];ef=np.c_[delta,np.linalg.norm(delta,axis=-1)]
   for j,p in enumerate(a['pressures']):
    for k,v in [('coarse',c),('edges',ef),('globals',np.r_[a['geometry_scales'],p]),('targets',a['displacements'][j]),('geometry_ids',g),('load_ids',j),('pressures',p)]:values[k].append(v)
 arrays={k:np.array(v) for k,v in values.items()}
 for k in ['coarse','edges','globals']:arrays[k]=(arrays[k]-stats[k+'_mean'])/stats[k+'_std']
 np.savez_compressed(out/f'{split}.npz',**arrays);splits[split]={'geometry_ids':[lo,hi],'case_count':len(arrays['targets']),'sha256':sha(out/f'{split}.npz')}
result={'normalizers_fit':'NONE: frozen original 20 training geometries; no confirmation labels used for transformations','parent_normalizer_sha256':sha(source/'normalizers.npz'),'source_cohort_manifest_sha256':sha(P/'experiments/manifests/cardiac_confirmation_v1.json'),'source_sha256':sha(Path(__file__)),'splits':splits}
(P/'experiments/manifests/confirmation_ml.json').write_text(json.dumps(result,indent=2));print('CONFIRMATION TRANSFORM COMPLETE',splits,flush=True)
