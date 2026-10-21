"""Actual frozen GRU/graph parameter gradients through P2 mechanics and causal histories."""
import hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.dynamic_graph import DynamicGraph,newmark_operators
from lv_surrogate.physics.quadratic_residual import CardiacQuadratic
torch.set_num_threads(2);torch.manual_seed(20261021)
manifest=json.loads((P/'experiments/metrics/quadratic_network_geometry.json').read_text());path=P/manifest['path']
assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256'];arrays=dict(np.load(path))
root=P/'data/processed/cardiac_dynamic_ml_v2_full';stats=dict(np.load(root/'normalizers.npz'));train=dict(np.load(root/'train.npz'))
assert str(train['case_ids'][0])==manifest['training_case']['id']
checkpoint=P/'experiments/checkpoints/cardiac_dynamic_graph_v2_full/data_seed11.pt'
freeze=json.loads((P/'experiments/manifests/dynamic_final_freeze.json').read_text());digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert freeze['checkpoint_sha256'][str(checkpoint.relative_to(P))]==digest
net=DynamicGraph(stats).double();net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);net.eval()
basis=net.basis.detach().numpy().reshape(len(net.basis),-1,3);assert basis.shape[1]==len(arrays['coords'])
lift=np.empty((len(basis),int(arrays['dof_count'])),dtype=np.float64)
nodes=arrays['cells'];edges=list(itertools.combinations(range(4),2))
canonical=np.concatenate((basis[:,nodes],np.stack([.5*(basis[:,nodes[:,i]]+basis[:,nodes[:,j]]) for i,j in edges],axis=2)),axis=2)
for c in range(3):lift[:,arrays['canonical_cell_dofs'][:,:,c]]=canonical[:,:,:,c]
lift=torch.tensor(lift,dtype=torch.float64)
features=[torch.tensor(train[k][:1],dtype=torch.float64) for k in ['coarse','globals','edges']]
top=torch.tensor(np.load(root/'topology.npy'));pressure=torch.tensor(train['pressure'][:1,:11],dtype=torch.float64)
vm,am=[torch.tensor(x,dtype=torch.float64) for x in newmark_operators(11,.005)]
mechanics=CardiacQuadratic(arrays,dtype=torch.float64,dt=.005)
parameters=list(net.parameters());original=[p.detach().clone() for p in parameters]
def fields(frame):
 q=net(*features,top,pressure)[0];v=vm@q;a=am@q
 decode=lambda x:(x*net.coefficient_scale)@lift
 zero=torch.zeros_like(q[0])
 return dict(u=decode(q[frame]),u_old=decode(q[frame-1] if frame else zero),
  v_old=decode(v[frame-1] if frame else zero),a_old=decode(a[frame-1] if frame else zero))
all_fields=[];case_pressure=[];spec=[];records=[]
epsilons=[1e-3,3e-4,1e-4]
for direction_id,frame in enumerate([0,5,10]):
 direction=[torch.randn_like(p) for p in parameters];size=torch.sqrt(sum(d.square().sum() for d in direction));direction=[d/size for d in direction]
 f=fields(frame);load=float(pressure[0,frame])*16070.954
 R=mechanics(f['u'],load,create_graph=True,history={k:v for k,v in f.items() if k!='u'});loss=.5*R.square().sum()
 grad=torch.autograd.grad(loss,parameters,allow_unused=True)
 derivative=float(sum((g*d).sum() for g,d in zip(grad,direction) if g is not None))
 assert np.isfinite(derivative) and abs(derivative)>1e-14
 start=len(all_fields);all_fields.append({k:v.detach().numpy() for k,v in f.items()});case_pressure.append(load)
 spec.append(dict(direction=direction_id,frame=frame,epsilon=0.,sign=0))
 for eps in epsilons:
  for sign in [-1,1]:
   with torch.no_grad():
    for p,o,d in zip(parameters,original,direction):p.copy_(o+sign*eps*d)
    trial=fields(frame)
   all_fields.append({k:v.numpy() for k,v in trial.items()});case_pressure.append(load)
   spec.append(dict(direction=direction_id,frame=frame,epsilon=eps,sign=sign))
 with torch.no_grad():
  for p,o in zip(parameters,original):p.copy_(o)
 records.append(dict(direction=direction_id,frame=frame,autograd_directional_derivative=derivative,
  torch_baseline_loss=float(loss.detach()),field_rows=list(range(start,len(all_fields)))))
 print('P2 NETWORK AUTOGRAD',frame,derivative,flush=True)
out=P/'experiments/logs/quadratic_network_gradient_fields.npz';assert not out.exists()
np.savez_compressed(out,**{k:np.stack([f[k] for f in all_fields]) for k in all_fields[0]},pressure=np.array(case_pressure))
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
result=dict(complete=True,checkpoint=str(checkpoint.relative_to(P)),checkpoint_sha256=digest,
 geometry_manifest_sha256=hashlib.sha256((P/'experiments/metrics/quadratic_network_geometry.json').read_bytes()).hexdigest(),
 fields_path=str(out.relative_to(P)),fields_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),cases=spec,records=records,epsilons=epsilons,
 data_access=['original_train_inputs','original_train_only_preprocessing','frozen_checkpoint'],
 model_source_sha256=hashlib.sha256((P/'src/lv_surrogate/models/dynamic_graph.py').read_bytes()).hexdigest(),
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 scope='Actual frozen trained graph/GRU weights, causal predicted histories and exact Newmark chain through a P1-to-P2 lifted train-only POD decoder into the P2 weak residual. Temporary in-memory parameter perturbations only; checkpoint unchanged. This verifies the architecture/backend chain, not a newly trained full-P2 basis/model; final follow-up trained checkpoints must be rechecked. No test/calibration data or training updates.')
(P/'experiments/metrics/quadratic_network_gradient_export.json').write_text(json.dumps(result,indent=2)+'\n')
