"""Independent native checks for 5ms full-cycle residual/history gradients."""
from pathlib import Path
import sys,json,numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.physics.fem_bridge import FEMBridge
from lv_surrogate.models.dynamic_graph import newmark_operators
torch.set_num_threads(2);root=P/'data/processed/cardiac_dynamic_v1/g000_l0';a=np.load(root/'trajectory.npz');geom=np.load(root/'geometry.npz');bridge=FEMBridge(P,threads=2);records=[]
try:
 for t in [39,99]:
  h={k:a[k][t] for k in ['u_old','v_old','a_old']};p=float(a['pressure'][t]);f=1e-5*np.sin(50*geom['coords']);u=a['u'][t]+f;ref=bridge.evaluate(u,p,True,geometry_path=root/'geometry.npz',history=h,dt=.005)
  own=CardiacFirstStep(geom,dt=.005,analytic_piola=True);z=torch.tensor(u,requires_grad=True);r=own(z,p,create_graph=True,history=h);loss=(r/float(geom['force_scale_N'])).square().mean();g=torch.autograd.grad(loss,z)[0].numpy();force_error=np.linalg.norm(r.detach().numpy()-ref['residual'])/np.linalg.norm(ref['residual']);gradient_error=np.linalg.norm(g-ref['gradient'])/np.linalg.norm(ref['gradient'])
  gpu=CardiacFirstStep(geom,dtype=torch.float32,dt=.005,analytic_piola=True).to('mps');z=torch.tensor(u,dtype=torch.float32,device='mps',requires_grad=True);rr=gpu(z,p,create_graph=True,history=h);gg=torch.autograd.grad((rr/float(geom['force_scale_N'])).square().mean(),z)[0];gpu_force=np.linalg.norm(rr.detach().cpu().numpy()-ref['residual'])/np.linalg.norm(ref['residual']);gpu_grad=np.linalg.norm(gg.cpu().numpy()-ref['gradient'])/np.linalg.norm(ref['gradient'])
  eta=torch.tensor(.1,dtype=torch.float64,requires_grad=True);ff=torch.tensor(f);hh={k:torch.tensor(h[k])+eta*mult*ff for k,mult in [('u_old',.9),('v_old',.1/.005),('a_old',.05/.005**2)]};rr=own(torch.tensor(a['u'][t])+eta*ff,p,create_graph=True,history=hh);value=(rr/float(geom['force_scale_N'])).square().mean();ad=float(torch.autograd.grad(value,eta)[0]);eps=1e-4;fd=[]
  for v in [.1+eps,.1-eps]:
   hh={k:h[k]+v*mult*f for k,mult in [('u_old',.9),('v_old',.1/.005),('a_old',.05/.005**2)]};fd.append(float(bridge.evaluate(a['u'][t]+v*f,p,False,geometry_path=root/'geometry.npz',history=hh,dt=.005)['loss']))
  fd=(fd[0]-fd[1])/(2*eps);history_error=abs(fd-ad)/max(abs(ad),1e-20);record={'frame':t,'cpu64_force_relative_error':float(force_error),'cpu64_u_gradient_relative_error':float(gradient_error),'gpu32_force_relative_error':float(gpu_force),'gpu32_u_gradient_relative_error':float(gpu_grad),'history_chain_gradient_relative_error_vs_native_finite_difference':history_error};assert max(force_error,gradient_error)<1e-7 and max(gpu_force,gpu_grad)<.01 and history_error<1e-4;records.append(record)
 # Verify causal linear maps against all stored Newmark histories, independently of POD.
 vmap,amap=newmark_operators(len(a['u']),.005);projected=a['u'].reshape(len(a['u']),-1)[:,::400];v=vmap@projected;acc=amap@projected;ve=np.linalg.norm(v[:-1]-a['v_old'][1:].reshape(len(a['u'])-1,-1)[:,::400])/np.linalg.norm(v[:-1]);ae=np.linalg.norm(acc[:-1]-a['a_old'][1:].reshape(len(a['u'])-1,-1)[:,::400])/np.linalg.norm(acc[:-1]);assert max(ve,ae)<1e-8
 result={'passed':True,'dt_s':.005,'records':records,'newmark_velocity_relative_error':float(ve),'newmark_acceleration_relative_error':float(ae),'source':'actual converged full-cycle training geometry; native FE forces, tangent and coupled-history finite differences'};(P/'experiments/metrics/dynamic_backend.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
finally:bridge.close()
