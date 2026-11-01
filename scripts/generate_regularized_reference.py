"""Restartable MPI P2 trajectory with whole-element positivity before history updates."""
import argparse,fcntl,json,mmap,os,sys,time,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.parallel_fem import ParallelP2,sha,df,MPI
from lv_surrogate.physics.reference_fem import residual
from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit as BernsteinAudit
from lv_surrogate.physics.admissible_newton_regularized_forcing import solve as regularized_solve
from lv_surrogate.physics.durable_reference import recover,atomic_json,sha
comm=MPI.COMM_WORLD;rank=comm.rank
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--qualification-stop',type=int);args=ap.parse_args()
 p=json.loads((P/args.protocol).read_text());cfg=json.loads((P/p['config']).read_text())
 assert comm.size==p['parallel_ranks'];root=P/'data/processed'/cfg['name'];mp=root/'manifest.json'
 m=json.loads(mp.read_text());assert m['config_sha256']==sha(P/p['config']) and m['prefix_positivity_passed'] and m['prefix_independent_native_audit_passed']
 if rank==0:
  lock=(root/'generation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (root/'failure.json').exists(),'Preserve a failed run; fix a separately declared follow-up protocol'
  for relative in p.get('required_audits',[]):
   proof=json.loads((P/relative).read_text());assert proof['complete'] and proof['passed']
  for path,digest in p['frozen_source_sha256'].items():assert sha(P/path)==digest
  if p.get('source_freeze'):
   freeze=json.loads((P/p['source_freeze']).read_text())
   for relative,digest in freeze.get('qualification_audit_sha256',{}).items():
    proof=json.loads((P/relative).read_text());assert proof['complete'] and proof['passed'] and sha(P/relative)==digest
   for relative,digest in freeze['sha256'].items():assert sha(P/relative)==digest,'Frozen numerical implementation changed: '+relative
   for relative,digest in freeze.get('cache_manifest_sha256',{}).items():assert sha(P/relative)==digest,'Qualified mesh/fiber cache changed'
 state=None
 if rank==0:
  c=np.load(root/'displacement_coefficients.npy',mmap_mode='r+')
  recover(root,m,c)
  saved=np.load(root/'restart.npz');count=int(saved['completed_steps'])
  assert count==m['completed_steps']
  state={k:saved[k] for k in ['u','u_old','v_old','a_old']}
  previous=c[count-2].copy() if count>=2 else np.zeros(c.shape[1])
  payload=dict(state=state,previous=previous,count=count)
 else:payload=None
 payload=comm.bcast(payload,root=0);start=payload['count'];stop=min(p.get('stop_after_steps',m['n_steps']),m['n_steps'])
 if args.qualification_stop is not None:
  assert args.qualification_stop in p.get('allowed_qualification_stops',[]);stop=args.qualification_stop
 adapter=ParallelP2(P,p['source_reference'],cfg);problem=adapter.problem
 for name,values in payload['state'].items():adapter.assign(name,values)
 previous_local=payload['previous'][adapter.mapping].copy()
 if rank==0:
  data=np.load(root/'inputs.npz');audit=BernsteinAudit(data)
  positivity=json.loads((root/'positivity.json').read_text());assert len(positivity)==start
  node_map=np.full((len(data['coords']),3),-1,dtype=np.int64)
  for component in range(3):
   for j in range(4):node_map[data['cells'][:,j],component]=data['canonical_cell_dofs'][:,j,component]
  assert (node_map>=0).all();nodes=np.load(root/'displacements.npy',mmap_mode='r+')
  known=np.load(P/'data/processed'/p['known_reference']/'displacement_coefficients.npy',mmap_mode='r') if p.get('known_reference') else None
  times=data['time'];pressures=data['pressure'];(root/'step_journal').mkdir(exist_ok=True)
  accepted_source=dict(generator_sha256=sha(Path(__file__)),parallel_adapter_sha256=sha(P/'src/lv_surrogate/physics/parallel_fem.py'),
   positivity_operator_sha256=sha(P/'src/lv_surrogate/physics/bernstein.py'),ranks=comm.size,first_step=start)
  if p.get('source_freeze'):accepted_source['source_freeze_sha256']=sha(P/p['source_freeze'])
  m.setdefault('source_segments',[]).append(accepted_source)
 else:times=pressures=None;audit=None
 times=comm.bcast(times,root=0);pressures=comm.bcast(pressures,root=0)
 problem.solver._solver.parameters.update(dict(method='newtonls',maximum_iterations=100,absolute_tolerance=1e-8,relative_tolerance=0.))
 problem.solver._snes.setUseEW(False);overall=time.perf_counter();prior_native_wall=m.get('cumulative_native_wall_seconds',0)
 current={k:payload['state'][k] for k in ['u_old','v_old','a_old']}
 for i in range(start,stop):
  assert prior_native_wall+time.perf_counter()-overall<p['resource_protocol']['native_hours']*3600
  old=problem.u_old.vector().get_local().copy();v=problem.v_old.vector().get_local();a=problem.a_old.vector().get_local();dt=cfg['dt_s'];alpha=float(problem.parameters['alpha_f']);assert alpha==.4
  stage=(alpha*previous_local+(1-alpha)*old-alpha*old)/(1-alpha)
  guesses=[old,old+dt*v,old+dt*v+.5*dt**2*a,stage];initial=[]
  problem.parameters['p'].assign(float(pressures[i]))
  for values in guesses:
   problem.u.vector().set_local(values);problem.u.vector().apply('insert');R=float(residual(problem).norm('l2'));initial.append(R if np.isfinite(R) else None)
  chosen=None
  for j in sorted([j for j,x in enumerate(initial) if x is not None],key=lambda j:initial[j]):
   problem.u.vector().set_local(guesses[j]);problem.u.vector().apply('insert');candidate=adapter.gather('u');error=None
   if rank==0:
    try:audit.pair(candidate,current['u_old'])
    except Exception as exc:error=str(exc)
   if comm.bcast(error,root=0) is None:chosen=j;break
  assert chosen is not None,'No whole-element-admissible initializer'
  trace=dict(step=i,time_s=float(times[i]),custom_newton_steps=[]);trace_path=root/f'custom_trace_{i:06d}.json'
  def persist_trace():
   if rank==0:atomic_json(trace_path,trace)
  persist_trace();solve_started=time.perf_counter()
  try:
   remaining=p['resource_protocol']['native_hours']*3600-prior_native_wall-(time.perf_counter()-overall)
   regularized_solve(problem,adapter,current,audit,dict(p['admissible_newton'],wall_budget_seconds=min(p['per_step_wall_seconds'],remaining)),trace,persist_trace)
   before=adapter.gather('u');problem.solver.solve();after=adapter.gather('u')
   assert problem.solver._snes.getIterationNumber()==0
   if rank==0:np.testing.assert_array_equal(before,after)
  except Exception:
   trial=adapter.gather('u')
   if rank==0:np.savez_compressed(root/'UNACCEPTED_iterate.npz',accepted=False,step=i,time_s=times[i],u=trial,**current)
   raise
  seconds=comm.allreduce(time.perf_counter()-solve_started,op=MPI.MAX);R=float(residual(problem).norm('l2'));snes=problem.solver._snes
  assert snes.getConvergedReason()>0 and np.isfinite(R) and R<1e-8
  assert not snes.getUseEW() and snes.getKSP().getType()=='gmres' and snes.getKSP().getPC().getType()=='gamg'
  u=adapter.gather('u');proof=None;error=None;proof_seconds=0.
  if rank==0:
   try:
    tick=time.perf_counter();proof=audit.pair(u,c[i-1] if i else np.zeros(len(u)),alpha_f=alpha);proof_seconds=time.perf_counter()-tick
    if known is not None:
     np.testing.assert_allclose(u,known[i],rtol=1e-6,atol=1e-9)
     assert np.max(abs(u-known[i]))<1e-9
   except Exception as exc:
    error=str(exc);cp=np.load(root/'restart.npz')
    np.savez_compressed(root/'UNACCEPTED_iterate.npz',accepted=False,step=i,time_s=times[i],u=u,**{k:cp[k] for k in ['u_old','v_old','a_old']})
  error=comm.bcast(error,root=0)
  if error is not None:raise RuntimeError('Whole-element/known-field gate: '+error)
  problem._update_fields();previous_local=old
  checkpoint={name:adapter.gather(name) for name in ['u','u_old','v_old','a_old']}
  current=comm.bcast({k:checkpoint[k] for k in current} if rank==0 else None,root=0)
  if rank==0:
   record=dict(step=i,time_s=float(times[i]),SNES_reason=int(snes.getConvergedReason()),residual_l2_N=R,
    nonlinear_iterations=int(snes.getIterationNumber()),EW_effective=False,effective_KSP='gmres',effective_PC='gamg',
    nonlinear_solve_seconds=seconds,initial_predictor_residuals_N=initial,selected_predictor=chosen,
    custom_iterations=trace['custom_iterations'],custom_trace_sha256=sha(trace_path),whole_element_positivity_passed=True,whole_element_check_seconds=proof_seconds)
   c[i]=u;nodes[i]=u[node_map];c.flush();nodes.flush()
   if hasattr(c._mmap,'madvise'):
    c._mmap.madvise(mmap.MADV_DONTNEED);nodes._mmap.madvise(mmap.MADV_DONTNEED)
   positivity.append(dict(step=i,time_s=float(times[i]),**proof))
   journal=root/'step_journal'/f'{i:06d}.json';assert not journal.exists(),'Never overwrite durable acceptance'
   temp=root/'restart.pending.npz';np.savez_compressed(temp,completed_steps=i+1,**checkpoint)
   with temp.open('rb') as f:os.fsync(f.fileno())
   import hashlib
   atomic_json(journal,dict(config_sha256=m['config_sha256'],record=record,positivity=positivity[-1],coefficient_sha256=hashlib.sha256(np.asarray(u).tobytes()).hexdigest(),input_checkpoint_sha256=sha(root/'restart.npz'),accepted_checkpoint_sha256=sha(temp)))
   (root/'restart.npz').replace(root/'restart.previous.npz');temp.replace(root/'restart.npz')
   m['steps'].append(record);m['completed_steps']=i+1;m['measured_wall_seconds']=time.perf_counter()-overall;m['cumulative_native_wall_seconds']=prior_native_wall+time.perf_counter()-overall
   m['positivity_scope']='Whole-element cubic Bernstein positivity at endpoint and stress stage, with same-margin deterministic subdivision; float64,not interval arithmetic.'
   atomic_json(root/'positivity.json',positivity);atomic_json(mp,m)
   print('PARALLEL REFERENCE',i+1,'/',m['n_steps'],'solve_s',seconds,'whole_element_s',proof_seconds,flush=True)
 comm.Barrier()
 if rank==0:
  m['cumulative_native_wall_seconds']=prior_native_wall+time.perf_counter()-overall
  m['pilot_complete']=m['completed_steps']==stop;m['complete']=m['completed_steps']==m['n_steps']
  if m['complete']:
   m['output_sha256']={name:sha(root/name) for name in ['inputs.npz','displacement_coefficients.npy','displacements.npy','positivity.json','restart.npz']}
  temp=mp.with_suffix('.pending.json');temp.write_text(json.dumps(m,indent=2));temp.replace(mp)
  print('PARALLEL GENERATION END',m['complete'],m['completed_steps'],flush=True)
try:main()
except BaseException as exc:
 traceback.print_exc()
 if rank==0:
  try:
   a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);a.add_argument('--qualification-stop',type=int);arg=a.parse_args();p=json.loads((P/arg.protocol).read_text());cfg=json.loads((P/p['config']).read_text())
   root=P/'data/processed'/cfg['name'];(root/'failure.json').write_text(json.dumps(dict(failed=True,error=str(exc),source_sha256=sha(Path(__file__)),scope='No unaccepted iterate is a physical label; last durable accepted checkpoint and journal retained.'),indent=2))
  except Exception:pass
 comm.Abort(1)
