"""Compare GPU float32 residual/gradient to exact trusted float64 FE assembly."""
from pathlib import Path
import sys,json,time
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.physics.fem_bridge import FEMBridge
torch.set_num_threads(2);device='mps'
if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
bridge=FEMBridge(P,threads=2);records=[]
try:
 for g in [0,1]:
  path=P/f'data/processed/cardiac_geometry_v1/geometry_{g:03d}.npz';a=np.load(path);own=CardiacFirstStep(a,dtype=torch.float32,analytic_piola=True).to(device);p=float(a['pressures'][2]);scale=float(a['force_scale_N']);u=a['displacements'][2]+np.random.default_rng(2027).normal(0,1e-7,a['displacements'][2].shape);ref=bridge.evaluate(u,p,geometry_path=path)
  oracle=CardiacFirstStep(a,analytic_piola=True);value64=torch.tensor(u,requires_grad=True);rr=oracle(value64,p,create_graph=True);ll=(rr/scale).square().mean();gg64=torch.autograd.grad(ll,value64)[0].detach().numpy()
  rel64=float(np.linalg.norm(gg64-ref['gradient'])/np.linalg.norm(ref['gradient']));assert rel64<1e-6,rel64
  elapsed=[]
  for j in range(5):
   value=torch.tensor(u,dtype=torch.float32,device=device,requires_grad=True);torch.mps.synchronize();start=time.perf_counter();R=own(value,p,create_graph=True);loss=(R/scale).square().mean();grad=torch.autograd.grad(loss,value)[0];torch.mps.synchronize();elapsed.append(time.perf_counter()-start)
  r=R.detach().cpu().numpy();gg=grad.detach().cpu().numpy();rerr=float(np.linalg.norm(r-ref['residual'])/np.linalg.norm(ref['residual']));gerr=float(np.linalg.norm(gg-ref['gradient'])/np.linalg.norm(ref['gradient']));rec={'geometry':g,'analytic_piola_float64_gradient_relative_error':rel64,'relative_force_error_at_1e_minus7_m_perturbation':rerr,'relative_loss_gradient_error':gerr,'forward_backward_median_s':float(np.median(elapsed[1:]))};records.append(rec);print(rec,flush=True)
 passed=all(r['relative_force_error_at_1e_minus7_m_perturbation']<.01 and r['relative_loss_gradient_error']<.01 for r in records)
 result={'passed':passed,'device':'MPS float32, analytic exact Piola with autodiff tangent','reference':'trusted FEniCS float64 exact nonlinear weak residual and VJP','gate':'both residual and full gradient relative L2 error below1% at controlled perturbed startup states on two numerical geometries','cases':records,'scope':'training backend only; final mechanics evaluation uses float64 trusted solver; no near-machine-tolerance residual claim from float32'}
 (P/'experiments/metrics/physics_backend.json').write_text(json.dumps(result,indent=2))
 if not passed:raise RuntimeError('GPU precision gate failed; retain float64 backend')
finally:bridge.close()
