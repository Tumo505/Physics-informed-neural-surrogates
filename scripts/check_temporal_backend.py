"""Verify exact float64 analytic-stress residual/tangent at a deformed dynamic state."""
from pathlib import Path
import sys,json,time
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.physics.fem_bridge import FEMBridge
torch.set_num_threads(2);a=np.load(P/'data/processed/cardiac/cohort.npz');state=np.load(P/'data/processed/cardiac_passive_trajectory_v1/step_0400.npz');h={k:state[k] for k in ['u_old','v_old','a_old']};u=state['u']+np.random.default_rng(2027).normal(0,1e-7,state['u'].shape);p=float(state['pressure']);own=CardiacFirstStep(a,analytic_piola=True);bridge=FEMBridge(P,threads=2)
try:
 ref=bridge.evaluate(u,p,history=h);times=[]
 for _ in range(4):
  value=torch.tensor(u,requires_grad=True);start=time.perf_counter();R=own(value,p,create_graph=True,history=h);loss=(R/float(a['force_scale_N'])).square().mean();g=torch.autograd.grad(loss,value)[0];times.append(time.perf_counter()-start)
 rerr=float(np.linalg.norm(R.detach().numpy()-ref['residual'])/np.linalg.norm(ref['residual']));gerr=float(np.linalg.norm(g.detach().numpy()-ref['gradient'])/np.linalg.norm(ref['gradient']));assert max(rerr,gerr)<1e-6,(rerr,gerr)
 result={'passed':True,'backend':'CPU float64 analytic Piola with full autodiff','state':'time0.401s, source passive trajectory, nonzero histories','relative_force_error':rerr,'relative_gradient_error':gerr,'median_forward_backward_s':float(np.median(times[1:]))};(P/'experiments/metrics/temporal_backend.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
finally:bridge.close()
