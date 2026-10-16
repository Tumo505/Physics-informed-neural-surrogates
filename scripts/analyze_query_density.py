"""Validation-only comparison of matched point-query density continuations."""
from pathlib import Path
import json, sys, hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.multiscale import MultiscaleSurrogate
D=P/'data/processed/full'
torch.set_num_threads(4)
norm=np.load(D/'normalizers.npz')
topology=torch.from_numpy(np.load(D/'topology.npy'))
parents=torch.from_numpy(np.load(D/'parents.npy'))
train_ids=np.random.default_rng(2027).choice(2250,16,replace=False).tolist()
selections={'train':train_ids,'validation':list(range(16))}
arrays={split:{name:np.load(D/f'{split}_{name}.npy',mmap_mode='r') for name in ['points','coarse','globals','edges','targets']} for split in selections}

def evaluate(path):
 ck=torch.load(path,map_location='cpu',weights_only=False)
 model=MultiscaleSurrogate().eval();model.load_state_dict(ck['model'])
 result={'epoch':ck['epoch'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 with torch.inference_mode():
  for split,ids in selections.items():
   data=arrays[split];cases=[]
   for i in ids:
    batch={k:torch.from_numpy(np.array(v[i:i+1],copy=True)) for k,v in data.items()}
    pred=model(batch['points'],batch['coarse'],batch['globals'],batch['edges'],topology,parents)
    diff=(pred*torch.from_numpy(norm['targets_std'])+torch.from_numpy(norm['targets_mean'])-batch['targets'])[0]
    cases.append({'case':i,'component_mse_cm2':float(diff.square().mean()),'mean_nodal_l2_cm':float(diff.norm(dim=-1).mean())})
   result[split]={'component_rmse_cm':float(np.sqrt(np.mean([c['component_mse_cm2'] for c in cases]))),'mean_nodal_l2_cm':float(np.mean([c['mean_nodal_l2_cm'] for c in cases])),'cases':cases}
 return result

results={'initialization':evaluate(P/'experiments/checkpoints/full_multiscale_v2/gnn_11_best.pt')}
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for nodes in (128,1024):
 name=f'query_density_{nodes}_v3'
 metric=json.loads((P/f'experiments/metrics/{name}/gnn_11.json').read_text())
 if not metric['complete']:raise RuntimeError(f'Incomplete run: {name}')
 results[str(nodes)]=evaluate(P/f'experiments/checkpoints/{name}/gnn_11_best.pt')
 results[str(nodes)]['train_seconds']=metric['train_seconds']
 hist=[r for r in metric['history'] if 'validation_component_mse_cm2' in r]
 axes[0].plot([h['epoch']+1 for h in hist],np.sqrt([h['validation_component_mse_cm2'] for h in hist]),marker='.',label=f'{nodes} queries')
axes[0].set(xlabel='Continuation epochs (0 = initialization)',ylabel='Validation component RMSE (cm)',title='Same initialization and update budget')
axes[0].legend()
labels=list(results);x=np.arange(len(labels));w=.35
for j,split in enumerate(selections):axes[1].bar(x+(j-.5)*w,[results[k][split]['component_rmse_cm'] for k in labels],w,label=f'{split} (16 geometries)')
axes[1].set(xticks=x,xticklabels=['Initial','128 queries','1,024 queries'],ylabel='Component RMSE (cm)',title='Selected checkpoints; single matched seed')
axes[1].legend()
out={'task':'validation-only query density ablation','seed':11,'training_geometries':2250,'training_diagnostic_ids':train_ids,'selection_validation_ids':list(range(16)),'calibration_and_test_read':False,'selection_rule':'minimum validation component MSE including initialization','inference_scope':'one matched seed and reused selection set; descriptive, no population significance claim','results':results}
initial=results['initialization']['validation']['component_rmse_cm']
a=results['128']['validation']['component_rmse_cm'];b=results['1024']['validation']['component_rmse_cm']
out['dense_relative_rmse_reduction_vs_sparse']=1-b/a
out['dense_relative_rmse_reduction_vs_initial']=1-b/initial
out['preferred_query_count']=1024 if b<a*(1-1e-6) else 128
out['selection_tie_tolerance_relative']=1e-6
out['improves_initialization']=min(a,b)<initial*(1-1e-6)
out['decision']='Expand only an improvement beyond numerical tolerance; otherwise retain existing ensemble and change the next hypothesis.'
(P/'experiments/metrics/query_density.json').write_text(json.dumps(out,indent=2))
fig.savefig(P/'reports/figures/query_density.png',dpi=180);plt.close(fig)
print(json.dumps({k:v for k,v in out.items() if k not in ('results','training_diagnostic_ids')},indent=2))
print({k:round(v['validation']['component_rmse_cm'],7) for k,v in results.items()})
