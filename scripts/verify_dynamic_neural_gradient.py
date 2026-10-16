"""Native finite-difference verification through the full recurrent history chain."""
from pathlib import Path
import json,sys,numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.dynamic_graph import DynamicGraph,newmark_operators
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.physics.fem_bridge import FEMBridge
torch.set_num_threads(2);cfg=json.loads((P/'configs/physics/dynamic_training.json').read_text());m=json.loads((P/f'experiments/manifests/{cfg["name"]}_full.json').read_text());label=m['selection']['selected'];D=P/f'data/processed/{cfg["data_tag"]}_full';stats=dict(np.load(D/'normalizers.npz'));a=np.load(D/'train.npz');top=torch.tensor(np.load(D/'topology.npy'));net=DynamicGraph(stats).double();net.load_state_dict(torch.load(P/f'experiments/checkpoints/{cfg["name"]}_full/{label}_seed11.pt',map_location='cpu',weights_only=False)['model']);inputs=[torch.tensor(a[k][:1],dtype=torch.float64) for k in ['coarse','globals','edges']];pressure=torch.tensor(a['pressure'][:1],dtype=torch.float64);vm,am=[torch.tensor(x) for x in newmark_operators(200,.005)];t=99;root=P/f'data/processed/cardiac_dynamic_v1/{a["case_ids"][0]}';geo=np.load(root/'geometry.npz');own=CardiacFirstStep(geo,dt=.005,analytic_piola=True);p=float(pressure[0,t])*16070.954

def fields():
 q=net(*inputs,top,pressure)[0];v=vm@q;acc=am@q;return net.decode(q[t]).reshape(-1,3),{k:net.decode(x[t-1]).reshape(-1,3) for k,x in zip(['u_old','v_old','a_old'],[q,v,acc])}
u,h=fields();r=own(u,p,create_graph=True,history=h);loss=(r/float(geo['force_scale_N'])).square().mean();gradient=torch.autograd.grad(loss,list(net.parameters()));norm=float(sum(g.square().sum() for g in gradient).sqrt());assert norm>0;direction=[g/norm for g in gradient];original=[p.detach().clone() for p in net.parameters()];bridge=FEMBridge(P,threads=2);values=[];epsilon=1e-5
try:
 for sign in [1,-1]:
  with torch.no_grad():
   for p0,base,d in zip(net.parameters(),original,direction):p0.copy_(base+sign*epsilon*d)
   u,h=fields();values.append(float(bridge.evaluate(u.numpy(),p,False,geometry_path=root/'geometry.npz',history={k:x.numpy() for k,x in h.items()},dt=.005)['loss']))
 fd=(values[0]-values[1])/(2*epsilon);error=abs(fd-norm)/max(norm,1e-20);assert error<1e-3
 result={'passed':True,'seed':11,'case':str(a['case_ids'][0]),'frame':t,'read_split':'training only','physics_only_network_parameter_gradient_norm':norm,'native_full_recurrence_directional_finite_difference':fd,'relative_error':error,'epsilon':epsilon,'scope':'CPUfloat64 full200-step predicted recurrence, derivative includes displacement and both velocity/acceleration histories, compared with independent native FE residual'};(P/'experiments/metrics/dynamic_neural_gradient.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
finally:bridge.close()
