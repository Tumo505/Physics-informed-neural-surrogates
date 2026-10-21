"""One known-equation MPI solve, canonical transfer and serial-target equivalence."""
import argparse,fcntl,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import df,LVProblem,residual
from mpi4py import MPI
from cardiac_benchmark.geometry import LVGeometry
from cardiac_benchmark.material import HolzapfelOgden
from lv_surrogate.physics.quadratic_solver import make_quadratic_problem
comm=MPI.COMM_WORLD;rank=comm.rank
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
 pp=P/args.protocol;p=json.loads(pp.read_text());assert comm.size==p['parallel_ranks']
 assert p['steps_after_checkpoint']==1 and p.get('timestep_divisor',1)==1
 root=P/'data/processed'/p['reference'];m=json.loads((root/'manifest.json').read_text());cfg=m['config']
 cache=P/'data/processed/parallel_mesh_cache'/p['reference'];meta=json.loads((cache/'manifest.json').read_text())
 assert meta['config_sha256']==m['config_sha256']
 payload=None
 if rank==0:
  lock=(P/'experiments/logs'/f'{Path(p["output"]).stem}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (P/p['output']).exists()
  for key in ['required_qualification','required_audit']:
   if key in p:
    proof=json.loads((P/p[key]).read_text());assert proof['complete'] and proof['passed'], key
  for n,h in meta['output_sha256'].items():assert sha(cache/n)==h
  assert sha(root/'inputs.npz')==meta['inputs_sha256']
  data=np.load(root/'inputs.npz');saved=np.load(root/'restart.npz')
  lineage={};initial_expected=None
  if 'qualification_step' in p:
   assert m['complete'];step=p['qualification_step'];all_fields=np.load(root/'displacement_coefficients.npy',mmap_mode='r')
   old=np.zeros_like(saved['u_old']);v=old.copy();a=old.copy();dt=cfg['dt_s']
   for u in all_fields[:step]:
    anew=(u-old-dt*v-(.5-.36)*dt**2*a)/(.36*dt**2)
    v=v+dt*(.3*a+.7*anew);old=u.copy();a=anew
   expected=all_fields[step].copy();previous=all_fields[step-2].copy()
  else:
   source=json.loads((P/p['serial_target_protocol']).read_text());sr=json.loads((P/source['output']).read_text())
   audit=json.loads((P/p['required_serial_target_audit']).read_text());assert audit['passed'] and audit['complete']
   assert source['resume_probe_protocol']==p['resume_probe_protocol'] and source['reference']==p['reference']
   assert audit['fields_sha256']==sha(P/source['fields']) and audit['protocol_sha256']==sha(P/p['serial_target_protocol'])
   assert sr['all_requested_steps_accepted'];step=sr['start_step']
   parent=json.loads((P/p['resume_probe_protocol']).read_text());pr=json.loads((P/parent['output']).read_text())
   assert sha(P/parent['checkpoint'])==sr['resumed_checkpoint_sha256']==pr['probe_checkpoint_sha256']
   s=np.load(P/parent['checkpoint']);old=s['u_old'];v=s['v_old'];a=s['a_old']
   assert step==pr['start_step']+int(s['completed_probe_steps'])
   predecessor=json.loads((P/source['stress_stage_predictor']['previous_displacement_protocol']).read_text())
   previous=np.load(P/predecessor['checkpoint'])['u_old']
   assert sha(P/predecessor['checkpoint'])==pr['resumed_checkpoint_sha256']
   expected=np.load(P/source['fields'],mmap_mode='r')[0].copy()
   initial_expected=sr['records'][0]['iteration_history'][0]['residual_N']
   lineage=dict(resumed_checkpoint_sha256=sha(P/parent['checkpoint']),resumed_accepted_probe_steps=int(s['completed_probe_steps']))
  payload=dict(step=step,time_s=float(data['time'][step]),pressure=float(data['pressure'][step]),
   old=old,v=v,a=a,previous=previous,expected=expected,lineage=lineage,initial_expected=initial_expected,checkpoint_sha256=sha(root/'restart.npz'))
 payload=comm.bcast(payload,root=0)
 result=dict(complete=False,finished=False,engineering_only=True,protocol=p,protocol_sha256=sha(pp),
  reference_config_sha256=m['config_sha256'],checkpoint_sha256=payload['checkpoint_sha256'],
  start_step=payload['step'],dt_s=cfg['dt_s'],full_cycle=False,changed_time_discretization=False,
  qualification_only='qualification_step' in p,records=[],parallel_ranks=comm.size,source_sha256=sha(Path(__file__)),**payload['lineage'])
 def persist():
  if rank==0:
   temp=(P/p['output']).with_suffix('.pending.json');temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(P/p['output'])
 persist();setup_tick=time.perf_counter()
 df.set_log_level(30);df.parameters['form_compiler']['quadrature_degree']=cfg['quadrature_degree']
 df.parameters['form_compiler']['cpp_optimize']=True
 mesh=df.Mesh()
 with df.HDF5File(mesh.mpi_comm(),str(cache/'mesh.h5'),'r') as h:
  h.read(mesh,'mesh',False);ffun=df.MeshFunction('size_t',mesh,2);h.read(ffun,'ffun')
  V1=df.VectorFunctionSpace(mesh,'CG',1);fibers=[]
  for name in ['f0','s0','n0']:
   f=df.Function(V1);h.read(f,name);fibers.append(f)
 geo=LVGeometry(mesh=mesh,ffun=ffun,markers=meta['markers'],f0=fibers[0],s0=fibers[1],n0=fibers[2])
 material=HolzapfelOgden(fibers[0],fibers[1],tau=df.Constant(0.),parameters={k:df.Constant(v) for k,v in meta['material_parameters'].items()})
 parameters=LVProblem.default_parameters();parameters.update(function_space='P_2',dt=cfg['dt_s'])
 problem=make_quadratic_problem(geo,material,parameters,'gamg',linear_maximum_iterations=3000,gmres_restart=200)
 settings=problem.solver._solver.parameters
 settings.update(dict(absolute_tolerance=1e-8,relative_tolerance=0.,maximum_iterations=100,line_search='bt'))
 problem.solver._snes.setUseEW(True,version=2,rtol_0=.01,rtol_max=.1)
 V=problem.u_space;coords=V.tabulate_dof_coordinates();owned=problem.u.vector().local_size()
 lo,hi=V.dofmap().ownership_range();assert len(coords)==owned==hi-lo
 components=np.full(owned,-1,dtype=np.int8)
 for c in range(3):
  ids=np.asarray(V.sub(c).dofmap().dofs(),dtype=np.int64);ids=ids[(ids>=lo)&(ids<hi)]-lo;components[ids]=c
 assert (components>=0).all()
 serial=np.load(cache/'serial_dof_map.npz');mapping=np.empty(owned,dtype=np.int64);distance_max=0.
 for c in range(3):
  original=np.flatnonzero(serial['components']==c);local=np.flatnonzero(components==c)
  distances,near=cKDTree(serial['coordinates'][original]).query(coords[local],workers=1)
  distance_max=max(distance_max,float(distances.max(initial=0.)));mapping[local]=original[near]
 assert distance_max<1e-12,'MPI space differs from exact serial mesh/space'
 parts=comm.gather(mapping,root=0)
 if rank==0:
  assert len(np.concatenate(parts))==len(serial['components'])
  np.testing.assert_array_equal(np.bincount(np.concatenate(parts),minlength=len(serial['components'])),1)
 result['canonical_mapping_max_coordinate_error_m']=comm.allreduce(distance_max,op=MPI.MAX)
 def assign(name,global_values):
  f=getattr(problem,name);f.vector().set_local(global_values[mapping]);f.vector().apply('insert')
 for name,key in [('u_old','old'),('v_old','v'),('a_old','a')]:assign(name,payload[key])
 problem.parameters['p'].assign(payload['pressure'])
 dt=cfg['dt_s'];alpha=float(problem.parameters['alpha_f']);old=payload['old']
 stage=(alpha*payload['previous']+(1-alpha)*old-alpha*old)/(1-alpha)
 guesses=[old,old+dt*payload['v'],old+dt*payload['v']+.5*dt**2*payload['a'],stage]
 best=float('inf');chosen=None;initial=[]
 for i,guess in enumerate(guesses):
  assign('u',guess);r=float(residual(problem).norm('l2'));initial.append(r if np.isfinite(r) else None)
  if np.isfinite(r) and r<best:best=r;chosen=i
 assert chosen is not None;assign('u',guesses[chosen])
 if payload['initial_expected'] is not None:
  np.testing.assert_allclose(best,payload['initial_expected'],rtol=1e-6,atol=1e-10)
 result.update(initial_predictor_residuals_N=initial,selected_predictor=chosen,setup_seconds=time.perf_counter()-setup_tick)
 persist();history=[];started=time.perf_counter();timings=dict(F=0.,J=0.);calls=dict(F=0,J=0)
 for name in ['F','J']:
  original=getattr(problem._problem,name)
  def timed(*a,_name=name,_original=original):
   tick=time.perf_counter();r=_original(*a);timings[_name]+=time.perf_counter()-tick;calls[_name]+=1;return r
  setattr(problem._problem,name,timed)
 def monitor(snes,iteration,norm):
  history.append(dict(iteration=int(iteration),residual_N=float(norm)))
  if rank==0:print('MPI NATIVE',comm.size,iteration,norm,flush=True)
  if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>p['wall_budget_seconds']:
   raise RuntimeError('Declared MPI solve budget exceeded')
 problem.solver._snes.setMonitor(monitor);problem.solver.solve()
 seconds=comm.allreduce(time.perf_counter()-started,op=MPI.MAX)
 force=float(residual(problem).norm('l2'));snes=problem.solver._snes
 assert snes.getConvergedReason()>0 and np.isfinite(force) and force<1e-8
 expected=df.Function(V);expected.vector().set_local(payload['expected'][mapping]);expected.vector().apply('insert')
 error=float(df.assemble(df.inner(problem.u-expected,problem.u-expected)*df.dx(metadata={'quadrature_degree':4})))
 norm=float(df.assemble(df.inner(expected,expected)*df.dx(metadata={'quadrature_degree':4})))
 relative=float(np.sqrt(error/max(norm,1e-300)));assert relative<1e-6
 maximum=comm.allreduce(float(np.max(abs(problem.u.vector().get_local()-payload['expected'][mapping]),initial=0.)),op=MPI.MAX)
 assert maximum<1e-9
 record=dict(step=payload['step'],time_s=payload['time_s'],accepted=True,SNES_reason=int(snes.getConvergedReason()),
  residual_N=force,nonlinear_seconds=seconds,nonlinear_iterations=int(snes.getIterationNumber()),
  iteration_history=history,effective_KSP=snes.getKSP().getType(),effective_PC=snes.getKSP().getPC().getType(),
  EW_effective=bool(snes.getUseEW()),mechanical_audit_pending=True)
 result['records']=[record];result['serial_target_equivalence']=dict(relative_volume_L2=relative,maximum_absolute_coefficient_difference_m=maximum,passed=True)
 result['assembly_profile_max_rank_seconds']={k:comm.allreduce(v,op=MPI.MAX) for k,v in timings.items()};result['assembly_calls']=calls
 problem._update_fields();canonical={}
 for name in ['u','u_old','v_old','a_old']:
  arrays=comm.gather(problem.__getattribute__(name).vector().get_local(),root=0)
  if rank==0:
   a=np.empty(len(payload['old']))
   for ids,values in zip(parts,arrays):a[ids]=values
   canonical[name]=a
 if rank==0:
  np.save(P/p['fields'],canonical['u'][None,:]);np.savez_compressed(P/p['checkpoint'],completed_probe_steps=1,**canonical)
  result['probe_checkpoint_sha256']=sha(P/p['checkpoint'])
 result.update(complete=True,finished=True,all_requested_steps_accepted=True,wall_seconds=seconds)
 persist()
 if rank==0:print('MPI NATIVE SOLVE COMPLETE',seconds,relative,force,flush=True)
try:main()
except BaseException:
 traceback.print_exc();comm.Abort(1)
