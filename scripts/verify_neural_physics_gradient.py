"""Physics-only trained-network gradient, checked against exact native FE VJP."""
from pathlib import Path
import sys,json
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.geometry_pod import GeometryPOD
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.physics.fem_bridge import FEMBridge
torch.set_num_threads(2);D=P/'data/processed/cardiac_geometry_ml';n=np.load(D/'normalizers.npz');a=np.load(D/'train.npz');g=np.load(P/'data/processed/cardiac_geometry_v1/geometry_000.npz');m=json.loads((P/'experiments/manifests/cardiac_geometry_models_v2.json').read_text());lam=m['lambda_selection']['lambda'];net=GeometryPOD(*[n[k] for k in ['field_mean','basis','coefficient_scale']]).float().to('mps');ck=torch.load(P/f'experiments/checkpoints/cardiac_geometry_models_v2/physics_lambda{lam:g}_seed11_best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);own=CardiacFirstStep(g,dtype=torch.float32,analytic_piola=True).to('mps');b=[torch.tensor(a[k][3:4],dtype=torch.float32,device='mps') for k in ['coarse','globals','edges']];top=torch.tensor(np.load(D/'topology.npy'),device='mps');u=net(*b,top)[0];p=float(a['pressures'][3]);R=own(u,p,create_graph=True);loss=(R/float(g['force_scale_N'])).square().mean();params=list(net.parameters());grad=torch.autograd.grad(loss,params,retain_graph=True);flat=torch.cat([z.reshape(-1) for z in grad]).detach().cpu().numpy();bridge=FEMBridge(P,threads=2)
try:
 ref=bridge.evaluate(u.detach().cpu().numpy(),p,geometry_path=P/'data/processed/cardiac_geometry_v1/geometry_000.npz');other=torch.autograd.grad(u,params,grad_outputs=torch.tensor(ref['gradient'],dtype=torch.float32,device='mps'));exact=torch.cat([z.reshape(-1) for z in other]).detach().cpu().numpy();relative=float(np.linalg.norm(flat-exact)/np.linalg.norm(exact));norm=float(np.linalg.norm(flat));assert norm>0 and np.isfinite(norm)
 result={'physics_only_parameter_gradient_norm':norm,'relative_difference_vs_native_VJP_through_same_network_graph':relative,'below_one_percent_gate':relative<.01,'trained_state':'seed11, geometry0, maximum startup pressure','data_loss_used':False,'scope':'actual trained graph/POD model; float32 residual compared with native float64 tangent gradient at exactly the same predicted field'};(P/'experiments/metrics/neural_physics_gradient.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
 if relative>=.01:raise RuntimeError('Trained-state GPU gradient accuracy gate failed')
finally:bridge.close()
