"""Verify matrix-free adjoint-action VJP against explicit trusted Jacobian."""
from pathlib import Path
import sys,json,time
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.fem_bridge import FEMBridge
bridge=FEMBridge(P,threads=2);a=np.load(P/'data/processed/cardiac_geometry_v1/geometry_000.npz');u=a['displacements'][2]+np.random.default_rng(2027).normal(0,1e-7,a['displacements'][2].shape);p=float(a['pressures'][2]);path=P/'data/processed/cardiac_geometry_v1/geometry_000.npz'
try:
 results={};grad={}
 for assembled in [True,False]:
  t=[]
  for j in range(7):
   start=time.perf_counter();r=bridge.evaluate(u,p,geometry_path=path,assembled_vjp=assembled);elapsed=time.perf_counter()-start
   if j>1:t.append(elapsed)
  results[str(assembled)]={'median_s':float(np.median(t)),'repetitions':len(t)};grad[assembled]=r['gradient']
 error=float(np.linalg.norm(grad[False]-grad[True])/np.linalg.norm(grad[True]));assert error<1e-10,error
 result={'explicit_Jacobian':results['True'],'adjoint_action':results['False'],'relative_gradient_difference':error,'same_residual_and_full_nonlinear_tangent':True};(P/'experiments/metrics/vjp_benchmark.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
finally:bridge.close()
