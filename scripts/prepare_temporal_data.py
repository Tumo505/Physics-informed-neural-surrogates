"""Train-only POD and feature scaling for a contiguous temporal holdout."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parents[1];cfg=json.loads((P/'configs/physics/temporal_model.json').read_text());source=P/'data/processed/cardiac_passive_trajectory_v1';out=P/'data/processed/cardiac_temporal_ml';out.mkdir(exist_ok=True)
metadata=json.loads((P/'experiments/metrics/cardiac_passive_trajectory_v1.json').read_text())
if not metadata['complete']:raise RuntimeError('Trajectory incomplete')
values={k:[] for k in ['u','u_old','v_old','a_old','pressure','time']}
for i in range(metadata['n_steps']):
 with np.load(source/f'step_{i:04d}.npz') as a:
  for k in values:values[k].append(a[k])
a={k:np.array(v) for k,v in values.items()};dt=metadata['dt_s'];beta=.36;anew=(a['u']-a['u_old']-dt*a['v_old']-(.5-beta)*dt**2*a['a_old'])/(beta*dt**2);delta=(anew-a['a_old']).reshape(len(anew),-1);lo,hi=cfg['train_steps'];mean=delta[lo:hi].mean(0);_,singular,V=np.linalg.svd(delta[lo:hi]-mean,full_matrices=False);basis=V[:cfg['pod_rank']];coeff=(delta[lo:hi]-mean)@basis.T
features=np.c_[*[a[k].reshape(len(anew),-1)@basis.T for k in ['u_old','v_old','a_old']],a['pressure']]
stats={'basis':basis,'delta_mean':mean,'coefficient_scale':np.maximum(coeff.std(0),1e-12),'feature_mean':features[lo:hi].mean(0),'feature_std':np.maximum(features[lo:hi].std(0),1e-12),'delta_scale':np.sqrt(np.mean((delta[lo:hi]-mean)**2)),'dt':dt,'singular_values':singular}
np.savez(out/'normalizers.npz',**stats)
for split in ['train','validation','test']:
 lo,hi=cfg[split+'_steps'];np.savez_compressed(out/f'{split}.npz',features=features[lo:hi],delta=delta[lo:hi],ids=np.arange(lo,hi),**{k:v[lo:hi] for k,v in a.items()})
(P/'experiments/manifests/temporal_data.json').write_text(json.dumps({'config':cfg,'normalization_fit':'first600 training timesteps only','split_unit':'contiguous time blocks within ONE cycle; no exchangeability or independent trajectory claim','dt_s':dt,'pod_rank':len(basis)},indent=2));print('TEMPORAL DATA READY',flush=True)
