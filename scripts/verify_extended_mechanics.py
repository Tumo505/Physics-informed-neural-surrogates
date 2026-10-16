"""Release gates for nonzero-history and geometry-dependent exact FE gradients."""
from pathlib import Path
import sys,json
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.fem_bridge import FEMBridge
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
torch.set_num_threads(2);bridge=FEMBridge(P,threads=2);rng=np.random.default_rng(2027);records=[]
base=np.load(P/'data/processed/cardiac/cohort.npz')
tasks=[('startup',base,base['displacements'][15],float(base['pressures'][15]),{},None)]
for g in [0,1]:
 path=P/f'data/processed/cardiac_geometry_v1/geometry_{g:03d}.npz';a=np.load(path);tasks.append((f'geometry{g}',a,a['displacements'][2],float(a['pressures'][2]),{},path))
for step in [10,100,400]:
 path=P/f'data/processed/cardiac_passive_trajectory_v1/step_{step:04d}.npz';a=np.load(path);h={k:a[k] for k in ['u_old','v_old','a_old']};tasks.append((f'temporal_step{step}',base,a['u'],float(a['pressure']),h,None))
tasks.append(('startup_after_history_switch',*tasks[0][1:]))
try:
 for name,a,u,p,h,path in tasks:
  own=CardiacFirstStep(a);kwargs={'geometry_path':path,'history':h};pert=u+1e-7*rng.normal(size=u.shape)
  ref=bridge.evaluate(pert,p,**kwargs);ind=own(torch.tensor(pert),p,history=h).detach().numpy();relative=float(np.linalg.norm(ind-ref['residual'])/np.linalg.norm(ref['residual']));assert relative<1e-6,(name,relative)
  direction=rng.normal(size=u.shape);direction/=np.linalg.norm(direction);eps=1e-8
  fd=(float(bridge.evaluate(pert+eps*direction,p,False,**kwargs)['loss'])-float(bridge.evaluate(pert-eps*direction,p,False,**kwargs)['loss']))/(2*eps)
  ad=float(np.sum(ref['gradient']*direction));error=abs(fd-ad)/max(abs(fd),abs(ad),1e-14);assert error<1e-4,(name,error)
  eq=float(bridge.evaluate(u,p,False,**kwargs)['loss'])
  parts=[float(bridge.evaluate(pert,p,False,region=region,**kwargs)['loss']) for region in ['interior','boundary']]
  np.testing.assert_allclose(sum(parts),float(ref['loss']),rtol=1e-12)
  # Independent PyTorch residual differentiated all the way to displacement.
  value=torch.tensor(pert,requires_grad=True);R=own(value,p,create_graph=True,history=h);loss=(R/float(a['force_scale_N'])).square().mean();grad=torch.autograd.grad(loss,value)[0].detach().numpy();gerr=float(np.linalg.norm(grad-ref['gradient'])/np.linalg.norm(ref['gradient']));assert gerr<1e-6,(name,gerr)
  rec={'task':name,'relative_residual_error':relative,'VJP_fd_relative_error':error,'independent_autodiff_relative_gradient_error':gerr,'converged_reference_normalized_residual_loss':eq,'boundary_interior_partition_additivity':True};records.append(rec);print(rec,flush=True)
 (P/'experiments/metrics/extended_mechanics_verification.json').write_text(json.dumps({'passed':True,'cases':records,'scope':'two numerical geometries and nonzero dynamic histories; independent assembly, full autodiff, exact tangent FD and natural-boundary partition checks'},indent=2))
finally:bridge.close()
