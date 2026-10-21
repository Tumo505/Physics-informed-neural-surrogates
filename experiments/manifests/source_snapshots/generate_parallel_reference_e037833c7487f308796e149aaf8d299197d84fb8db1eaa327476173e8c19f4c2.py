"""Restartable MPI P2 trajectory with whole-element positivity before history updates."""
import argparse,fcntl,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.parallel_fem import ParallelP2,sha,df,MPI
from lv_surrogate.physics.reference_fem import residual
from lv_surrogate.physics.bernstein import BernsteinAudit
comm=MPI.COMM_WORLD;rank=comm.rank
def atomic_json(path,value):
 temp=path.with_suffix('.pending.json')
 with temp.open('w') as f:
  json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 temp.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
 p=json.loads((P/args.protocol).read_text());cfg=json.loads((P/p['config']).read_text())
 assert comm.size==p['parallel_ranks'];root=P/'data/processed'/cfg['name'];mp=root/'manifest.json'
 m=json.loads(mp.read_text());assert m['config_sha256']==sha(P/p['config']) and m['prefix_positivity_passed']
 if rank==0:
  lock=(root/'generation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (root/'failure.json').exists(),'Preserve a failed run; fix a separately declared follow-up protocol'
  for relative in p.get('required_audits',[]):
   proof=json.loads((P/relative).read_text());assert proof['complete'] and proof['passed']
  if p.get('source_freeze'):
   freeze=json.loads((P/p['source_freeze']).read_text())
   for relative,digest in freeze['sha256'].items():assert sha(P/relative)==digest,'Frozen numerical implementation changed: '+relative
 state=None
 if rank==0:
  if not (root/'restart.npz').exists():
   assert (root/'restart.previous.npz').exists();(root/'restart.previous.npz').replace(root/'restart.npz')
  saved=np.load(root/'restart.npz');count=int(saved['completed_steps'])
  c=np.load(root/'displacement_coefficients.npy',mmap_mode='r+')
  next_journal=root/'step_journal'/f'{m["completed_steps"]:06d}.json'
  if count!=m['completed_steps'] or next_journal.exists():
   assert count in [m['completed_steps'],m['completed_steps']+1] and next_journal.exists()
   j=json.loads(next_journal.read_text());index=m['completed_steps'];record=j['record']
   assert j['config_sha256']==m['config_sha256'] and record['step']==index and record['SNES_reason']>0 and record['residual_l2_N']<1e-8
   import hashlib
   assert hashlib.sha256(np.asarray(c[index]).tobytes()).hexdigest()==j['coefficient_sha256']
   assert all(j['positivity'][s]['passed'] and j['positivity'][s]['minimum_Bernstein_bound']>1e-9 for s in ['endpoint','stress'])
   prior_path=root/'restart.previous.npz' if count==index+1 else root/'restart.npz'
   prior=np.load(prior_path);assert int(prior['completed_steps'])==index and sha(prior_path)==j['input_checkpoint_sha256']
   dt=cfg['dt_s'];anew=(c[index]-prior['u_old']-dt*prior['v_old']-(.5-.36)*dt**2*prior['a_old'])/(.36*dt**2)
   vnew=prior['v_old']+dt*(.3*prior['a_old']+.7*anew)
   if count==index+1:
    np.testing.assert_array_equal(saved['u_old'],c[index])
    np.testing.assert_allclose(saved['v_old'],vnew,rtol=1e-7,atol=1e-10)
    np.testing.assert_allclose(saved['a_old'],anew,rtol=1e-7,atol=1e-10)
   else:
    temp=root/'restart.recovered.npz';np.savez_compressed(temp,completed_steps=index+1,u=c[index],u_old=c[index],v_old=vnew,a_old=anew);temp.replace(root/'restart.npz')
   pos=json.loads((root/'positivity.json').read_text())
   if len(pos)==index:pos.append(j['positivity'])
   else:assert len(pos)==index+1 and pos[-1]==j['positivity']
   atomic_json(root/'positivity.json',pos);m['steps'].append(record);m['completed_steps']=index+1
   m.setdefault('journal_recoveries',[]).append(dict(step=index,positive_SNES_record_reused=True,physical_step_recomputed=False))
   atomic_json(mp,m);saved=np.load(root/'restart.npz');count=int(saved['completed_steps'])
   print('DURABLE ACCEPTANCE JOURNAL RECOVERED',index,flush=True)
  assert count==m['completed_steps']
  state={k:saved[k] for k in ['u','u_old','v_old','a_old']}
  previous=c[count-2].copy() if count>=2 else np.zeros(c.shape[1])
  payload=dict(state=state,previous=previous,count=count)
 else:payload=None
 payload=comm.bcast(payload,root=0);start=payload['count'];stop=min(p.get('stop_after_steps',m['n_steps']),m['n_steps'])
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
  m.setdefault('source_segments',[]).append(accepted_source)
 else:times=pressures=None
 times=comm.bcast(times,root=0);pressures=comm.bcast(pressures,root=0)
 history=[];solve_started=0.
 def monitor(snes,iteration,norm):
  history.append(dict(iteration=int(iteration),residual_N=float(norm) if np.isfinite(norm) else None))
  if rank==0:print('STREAM NEWTON',i,iteration,norm,flush=True)
  if comm.allreduce(time.perf_counter()-solve_started,op=MPI.MAX)>p.get('per_step_wall_seconds',1800):
   raise RuntimeError('Declared native per-step wall budget exceeded')
 problem.solver._snes.setMonitor(monitor);overall=time.perf_counter()
 for i in range(start,stop):
  old=problem.u_old.vector().get_local().copy();v=problem.v_old.vector().get_local();a=problem.a_old.vector().get_local();dt=cfg['dt_s']
  alpha=float(problem.parameters['alpha_f']);assert alpha==.4
  stage=(alpha*previous_local+(1-alpha)*old-alpha*old)/(1-alpha)
  guesses=[old,old+dt*v,old+dt*v+.5*dt**2*a,stage];best=float('inf');chosen=None
  problem.parameters['p'].assign(float(pressures[i]));initial=[]
  for j,values in enumerate(guesses):
   problem.u.vector().set_local(values);problem.u.vector().apply('insert');R=float(residual(problem).norm('l2'))
   initial.append(R if np.isfinite(R) else None)
   if np.isfinite(R) and R<best:chosen=j;best=R
  assert chosen is not None
  problem.u.vector().set_local(guesses[chosen]);problem.u.vector().apply('insert')
  history.clear();solve_started=time.perf_counter()
  try:problem.solver.solve()
  except Exception:
   trial=adapter.gather('u')
   if rank==0:
    cp=np.load(root/'restart.npz');np.savez_compressed(root/'UNACCEPTED_iterate.npz',accepted=False,step=i,time_s=times[i],u=trial,**{k:cp[k] for k in ['u_old','v_old','a_old']})
   raise
  seconds=comm.allreduce(time.perf_counter()-solve_started,op=MPI.MAX);R=float(residual(problem).norm('l2'));snes=problem.solver._snes
  assert snes.getConvergedReason()>0 and np.isfinite(R) and R<1e-8
  assert snes.getUseEW() and snes.getKSP().getType()=='gmres' and snes.getKSP().getPC().getType()=='gamg'
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
  if rank==0:
   record=dict(step=i,time_s=float(times[i]),SNES_reason=int(snes.getConvergedReason()),residual_l2_N=R,
    nonlinear_iterations=int(snes.getIterationNumber()),EW_effective=True,effective_KSP='gmres',effective_PC='gamg',
    nonlinear_solve_seconds=seconds,initial_predictor_residuals_N=initial,selected_predictor=chosen,
    iteration_history=list(history),whole_element_positivity_passed=True,whole_element_check_seconds=proof_seconds)
   c[i]=u;nodes[i]=u[node_map];c.flush();nodes.flush()
   positivity.append(dict(step=i,time_s=float(times[i]),**proof))
   journal=root/'step_journal'/f'{i:06d}.json'
   if journal.exists():journal.rename(journal.with_suffix('.previous_'+sha(journal)[:12]+'.json'))
   import hashlib
   atomic_json(journal,dict(config_sha256=m['config_sha256'],record=record,positivity=positivity[-1],
    coefficient_sha256=hashlib.sha256(np.asarray(u).tobytes()).hexdigest(),input_checkpoint_sha256=sha(root/'restart.npz')))
   temp=root/'restart.pending.npz';np.savez_compressed(temp,completed_steps=i+1,**checkpoint)
   with temp.open('rb') as f:os.fsync(f.fileno())
   (root/'restart.npz').replace(root/'restart.previous.npz');temp.replace(root/'restart.npz')
   m['steps'].append(record);m['completed_steps']=i+1;m['measured_wall_seconds']=time.perf_counter()-overall
   m['positivity_scope']='Whole-element cubic Bernstein positivity at endpoint and stress stage, with same-margin deterministic subdivision; float64,not interval arithmetic.'
   atomic_json(root/'positivity.json',positivity);atomic_json(mp,m)
   print('PARALLEL REFERENCE',i+1,'/',m['n_steps'],'solve_s',seconds,'whole_element_s',proof_seconds,flush=True)
 comm.Barrier()
 if rank==0:
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
   a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);arg=a.parse_args();p=json.loads((P/arg.protocol).read_text());cfg=json.loads((P/p['config']).read_text())
   root=P/'data/processed'/cfg['name'];(root/'failure.json').write_text(json.dumps(dict(failed=True,error=str(exc),source_sha256=sha(Path(__file__)),scope='No unaccepted iterate is a physical label; last durable accepted checkpoint and journal retained.'),indent=2))
  except Exception:pass
 comm.Abort(1)
