"""Independent, chunked native/P2 audit of one converged loaded capacity query."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import sys
import numpy as np
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import UFCTetrahedron
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,residual,df
from lv_surrogate.physics.quadratic_solver import make_quadratic_problem
ap=argparse.ArgumentParser();ap.add_argument('--stem',required=True);args=ap.parse_args()
assert Path(args.stem).name==args.stem
rp=P/f'experiments/metrics/{args.stem}.json';r=json.loads(rp.read_text())
assert r['complete'] and r['worker']['solve_accepted'] and r['process_exit_code']==0
assert r['displacement_degree']==2 and r['inexact_newton']
hp=P/f'experiments/logs/{args.stem}_fixed_histories.npz'
up=P/f'experiments/logs/{args.stem}_UNACCEPTED_iterate.npy'
paths=[rp,hp,up]
hashes={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for name,expected in r['worker']['history_probe'].get('resume',{}).get('input_sha256',{}).items():
 assert hashlib.sha256((P/name).read_bytes()).hexdigest()==expected
geo,material,base=setup(refinement_level=r['refinement_level'],dt=r['dt_s'],geometry_scales=[1,1])
par=dict(base.parameters);par['function_space']='P_2'
problem=make_quadratic_problem(geo,material,par,'gamg')
saved=np.load(hp);current=np.load(up);old=saved['u_old']
assert current.shape==(problem.u_space.dim(),)
assert float(saved['dt_s'])==r['dt_s']
assert float(saved['pressure_Pa'])==r['worker']['pressure_Pa']
for name,value in [('u',current)]+[(n,saved[n]) for n in ['u_old','v_old','a_old']]:
 assert np.isfinite(value).all()
 getattr(problem,name).vector().set_local(value);getattr(problem,name).vector().apply('insert')
problem.parameters['p'].assign(float(saved['pressure_Pa']))
force=float(residual(problem).norm('l2'));assert np.isfinite(force) and force<1e-8
np.testing.assert_allclose(force,r['worker']['residual_l2_N'],rtol=1e-7,atol=1e-12)
print('NATIVE FORCE VERIFIED',force,flush=True)
edges=list(itertools.combinations(range(4),2))
canonical=np.concatenate((np.eye(4),np.array([(np.eye(4)[i]+np.eye(4)[j])/2 for i,j in edges])))
multi=np.array([a for a in itertools.product(range(4),repeat=4) if sum(a)==3])
mult=np.array([math.factorial(3)/math.prod(math.factorial(int(i)) for i in a) for a in multi])
def basis(b):return mult[None,:]*np.prod(np.asarray(b)[:,None,:]**multi[None,:,:],axis=-1)
lattice=multi/3;transform=np.linalg.inv(basis(lattice));check=np.random.default_rng(2027).dirichlet(np.ones(4),size=7)
q=create_quadrature(UFCTetrahedron(),4);bary=np.c_[1-q.get_points().sum(1),q.get_points()]
shape=np.concatenate((bary*(2*bary-1),np.stack([4*bary[:,i]*bary[:,j] for i,j in edges],axis=1)),axis=1)
coords=problem.u_space.tabulate_dof_coordinates().reshape(-1,3)
cells=geo.mesh.cells();vertices=geo.mesh.coordinates();minimum={'endpoint':float('inf'),'stress':float('inf')};unresolved={k:0 for k in minimum}
independent=0.;mapping_error=0.
for start in range(0,len(cells),4096):
 end=min(start+4096,len(cells));X=vertices[cells[start:end]]
 D=(X[:,1:]-X[:,:1]).transpose(0,2,1);inverse=np.linalg.inv(D);measure=abs(np.linalg.det(D))
 gl=np.concatenate((-inverse.sum(1,keepdims=True),inverse),axis=1);maps=[]
 for component in range(3):
  local=np.stack([problem.u_space.sub(component).dofmap().cell_dofs(i) for i in range(start,end)])
  xyz=np.einsum('ekj,eij->eki',coords[local]-X[:,:1],inverse)
  bc=np.concatenate((1-xyz.sum(-1,keepdims=True),xyz),axis=-1)
  distance=abs(bc[:,:,None,:]-canonical[None,None,:,:]).sum(-1);order=distance.argmin(axis=1)
  error=float(np.take_along_axis(distance,order[:,None,:],axis=1).max());assert error<1e-9
  mapping_error=max(mapping_error,error);maps.append(np.take_along_axis(local,order,axis=1))
 maps=np.stack(maps,axis=-1)
 gradients=np.stack([np.concatenate(((4*b[None,:,None]-1)*gl,np.stack([4*(b[i]*gl[:,j]+b[j]*gl[:,i]) for i,j in edges],axis=1)),axis=1) for b in np.eye(4)],axis=1)
 for state,field in [('endpoint',current),('stress',.4*old+.6*current)]:
  vertex_F=np.eye(3)+np.einsum('eki,evkj->evij',field[maps],gradients)
  coeff=np.linalg.det(np.einsum('qv,evij->eqij',lattice,vertex_F))@transform.T
  assert np.isfinite(coeff).all();lower=coeff.min(axis=1)
  minimum[state]=min(minimum[state],float(lower.min()));unresolved[state]+=int((lower<=1e-9).sum())
  np.testing.assert_allclose(np.linalg.det(np.einsum('qv,evij->eqij',check,vertex_F)),coeff@basis(check).T,rtol=1e-11,atol=1e-11)
 values=np.einsum('qk,ekc->eqc',shape,current[maps]);independent+=float(np.einsum('e,q,eqc,eqc->',measure,q.get_weights(),values,values))
 if start%102400==0:print('AUDITED ELEMENTS',end,'/',len(cells),flush=True)
native=float(df.assemble(df.inner(problem.u,problem.u)*df.dx(metadata={'quadrature_degree':4})))
np.testing.assert_allclose(independent,native,rtol=1e-11,atol=1e-24)
for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(P))]
out=dict(complete=True,passed=not any(unresolved.values()),full_cycle=False,engineering_only=True,
 input_sha256=hashes,executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 native_residual_N=force,minimum_Bernstein_bound=minimum,unresolved_elements=unresolved,
 displacement_dofs=problem.u_space.dim(),tetrahedra=len(cells),canonical_map_maximum_error=mapping_error,
 physical_L2_squared_m5=native,independent_norm_relative_error=abs(independent-native)/max(native,1e-30),
 scope='One loaded algebraic query with inherited interpolated histories. Independently reassembled native force, every-element endpoint/stress cubic Bernstein bounds with 1e-9 float64 margin, polynomial identities and independent canonical P2/UFL norm. Not interval arithmetic, a trajectory or numerical convergence evidence.')
(P/f'experiments/metrics/{args.stem}_verification.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print('LOADED QUERY AUDIT PASSED',out['passed'],flush=True)
sys.exit(0 if out['passed'] else 2)
