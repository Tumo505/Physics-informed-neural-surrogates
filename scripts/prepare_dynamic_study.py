"""Train-only displacement POD; no calibration/test trajectory labels opened."""
from pathlib import Path
import json,hashlib,numpy as np,sys,argparse
from scipy.linalg import svd
P=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--regime',choices=['full','limited'],default='full');args=ap.parse_args();cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());out=P/f'data/processed/cardiac_dynamic_ml_{args.regime}';out.mkdir(exist_ok=True)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
train=[c for c in cfg['cases'] if c['split']=='train'];train=train if args.regime=='full' else train[:2];allowed=[*train,*[c for c in cfg['cases'] if c['split']=='validation']]
for c in allowed:
 m=json.loads((P/f'data/processed/cardiac_dynamic_v1/{c["id"]}/manifest.json').read_text());assert m['complete']
labels=np.load(P/'data/processed/cardiac_geometry_ml/cluster_labels.npy');edges=np.load(P/'data/processed/cardiac_geometry_ml/topology.npy');counts=np.bincount(labels,minlength=64)
np.save(out/'topology.npy',edges)
trajectories={c['id']:dict(np.load(P/f'data/processed/cardiac_dynamic_v1/{c["id"]}/trajectory.npz')) for c in allowed}
fields=np.concatenate([trajectories[c['id']]['u'].reshape(len(trajectories[c['id']]['u']),-1) for c in train]);normpath=out/'normalizers.npz'
if normpath.exists():stats=dict(np.load(normpath))
else:
 # Uncentered POD includes exact zero initial state without an offset field.
 _,s,V=svd(fields,full_matrices=False,lapack_driver='gesdd');basis=V[:12];coeff=fields@basis.T;scale=np.maximum(np.sqrt(np.mean(coeff**2,axis=0)),1e-8);q=coeff/scale;bound=1.5*np.max(abs(q),axis=0)+.1
 stats=dict(basis=basis,coefficient_scale=scale,coefficient_bound=bound,singular_values=s,field_scale=np.sqrt(np.mean(fields**2)),dq_scale=np.maximum(np.std(np.diff(q.reshape(len(train),-1,12),axis=1),axis=(0,1)),.01),ddq_scale=np.maximum(np.std(np.diff(q.reshape(len(train),-1,12),n=2,axis=1),axis=(0,1)),.01));np.savez(normpath,**stats)
for split in ['train','validation']:
 chosen=train if split=='train' else [c for c in allowed if c['split']==split];values={k:[] for k in ['coarse','globals','edges','pressure','q','u','u_old','v_old','a_old','case_ids']}
 for c in chosen:
  geom=np.load(P/f'data/processed/cardiac_dynamic_v1/{c["id"]}/geometry.npz');x=np.c_[geom['coords']/.1,geom['fiber'],geom['sheet']];pooled=np.zeros((64,9));np.add.at(pooled,labels,x);pooled/=counts[:,None];delta=pooled[edges[:,1],:3]-pooled[edges[:,0],:3];a=trajectories[c['id']]
  row=dict(coarse=pooled,globals=np.r_[(np.array(c['scales'])-1)/.05,0.],edges=np.c_[delta,np.linalg.norm(delta,axis=-1)],pressure=a['pressure']/16070.954,q=a['u'].reshape(len(a['u']),-1)@stats['basis'].T/stats['coefficient_scale'],case_ids=c['id'],**{k:a[k] for k in ['u','u_old','v_old','a_old']})
  for k,v in row.items():values[k].append(v)
 np.savez_compressed(out/f'{split}.npz',**{k:np.array(v) for k,v in values.items()})
manifest={'regime':args.regime,'train_cases':[c['id'] for c in train],'validation_cases':[c['id'] for c in allowed if c['split']=='validation'],'normalizer_sha256':sha(normpath),'feature_scaling':'reference length0.1m and documented source pressure maximum; geometry scales normalized by preregistered numerical box; no held-out labels','POD':'rank12 uncentered, training trajectories only; exact zero initial displacement','normalizers_fit_cases':[c['id'] for c in train],'source_sha256':sha(Path(__file__))}
(P/f'experiments/manifests/dynamic_ml_{args.regime}.json').write_text(json.dumps(manifest,indent=2));print('DYNAMIC DATA READY',args.regime,flush=True)
