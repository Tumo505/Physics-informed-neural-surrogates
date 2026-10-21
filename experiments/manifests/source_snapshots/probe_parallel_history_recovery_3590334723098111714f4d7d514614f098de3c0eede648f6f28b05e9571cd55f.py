"""MPI history/load homotopy; only exact lambda=1 endpoints are physical states."""
import argparse,fcntl,json,sys,time,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.parallel_fem import ParallelP2,sha
from lv_surrogate.physics.reference_fem import residual
from lv_surrogate.physics.bernstein import BernsteinAudit
from mpi4py import MPI
comm=MPI.COMM_WORLD;rank=comm.rank

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
 pp=P/args.protocol;p=json.loads(pp.read_text());assert comm.size==p['parallel_ranks']
 assert p['absolute_tolerance_N']==1e-8 and p['relative_tolerance']==0. and p.get('timestep_divisor',1)==1
 root=P/'data/processed'/p['reference'];m=json.loads((root/'manifest.json').read_text());cfg=m['config'];payload=None
 result=dict(complete=False,finished=False,all_requested_steps_accepted=False,full_cycle=False,
  changed_time_discretization=False,protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),
  checkpoint_sha256=sha(root/'restart.npz'),reference_config_sha256=m['config_sha256'],
  dt_s=cfg['dt_s'],parallel_ranks=comm.size,records=[],engineering_roots=[],attempts=[],
  scope='Adaptive algebraic history/load continuation; no intermediate roots are physical timesteps or labels. Final equation, histories, dt, material, load, Q6 and strict1e-8N acceptance unchanged. Local bounded qualification/recovery only.')
 def persist():
  if rank==0:
   out=P/p['output'];tmp=out.with_suffix('.pending.json');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(out)
 if rank==0:
  lock=(P/'experiments/logs'/f'{Path(p["output"]).stem}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (P/p['output']).exists(),'Never retry an existing output'
  for audit_path in p.get('required_audits',[]):
   proof=json.loads((P/audit_path).read_text());assert proof['complete'] and proof['passed'],audit_path
  for path,digest in p['frozen_source_sha256'].items():assert sha(P/path)==digest,'Recovery implementation changed'
  data=np.load(root/'inputs.npz');coef=np.load(root/'displacement_coefficients.npy',mmap_mode='r');expected=None
  if 'qualification_step' in p:
   assert m['complete'];step=p['qualification_step'];dt=cfg['dt_s'];old=np.zeros(coef.shape[1]);v=old.copy();a=old.copy();previous=None
   for u in coef[:step]:
    previous=dict(u_old=old.copy(),v_old=v.copy(),a_old=a.copy())
    anew=(u-old-dt*v-(.5-.36)*dt**2*a)/(.36*dt**2)
    v=v+dt*(.3*a+.7*anew);a=anew;old=u.copy()
   current=dict(u_old=old,v_old=v,a_old=a);expected=np.array(coef[step:step+p['steps_after_checkpoint']])
  else:
   parent=json.loads((P/p['resume_probe_protocol']).read_text());pr=json.loads((P/parent['output']).read_text())
   predecessor=json.loads((P/p['previous_history_protocol']).read_text());qr=json.loads((P/predecessor['output']).read_text())
   assert parent['reference']==predecessor['reference']==p['reference']
   assert pr['finished'] and pr['all_requested_steps_accepted']
   assert sha(P/parent['checkpoint'])==pr['probe_checkpoint_sha256']
   assert sha(P/predecessor['checkpoint'])==pr['resumed_checkpoint_sha256']==qr['probe_checkpoint_sha256']
   cp=np.load(P/parent['checkpoint']);prior=np.load(P/predecessor['checkpoint']);step=pr['start_step']+int(cp['completed_probe_steps'])
   current={k:cp[k].copy() for k in ['u_old','v_old','a_old']};previous={k:prior[k].copy() for k in current}
   result['resumed_checkpoint_sha256']=sha(P/parent['checkpoint'])
   result['previous_history_checkpoint_sha256']=sha(P/predecessor['checkpoint'])
   if p.get('serial_target_protocol'):
    target=json.loads((P/p['serial_target_protocol']).read_text());tr=json.loads((P/target['output']).read_text())
    assert tr['start_step']==step and tr['all_requested_steps_accepted'];expected=np.load(P/target['fields']).copy()
  assert step>=2 and previous is not None
  result['start_step']=step;payload=dict(step=step,current=current,previous=previous,expected=expected,
   times=np.array(data['time']),pressures=np.array(data['pressure']))
 payload=comm.bcast(payload,root=0);result['start_step']=payload['step'];persist()
 adapter=ParallelP2(P,p['reference'],cfg);problem=adapter.problem
 assert float(problem._beta)==.36 and abs(float(problem._gamma)-.7)<1e-14 and float(problem.parameters['alpha_f'])==.4
 settings=problem.solver._solver.parameters;options=p['history_continuation'];settings['maximum_iterations']=options['maximum_newton_per_increment']
 if rank==0:audit=BernsteinAudit(np.load(root/'inputs.npz'))
 else:audit=None
 current=payload['current'];previous=payload['previous'];dt=cfg['dt_s'];started=time.perf_counter();iteration_history=[]
 def local_assign(name,values):adapter.assign(name,values)
 def assign(hist,pressure,u):
  for k,x in hist.items():local_assign(k,x)
  local_assign('u',u);problem.parameters['p'].assign(float(pressure))
 def monitor(snes,it,norm):
  record=dict(iteration=int(it),residual_N=float(norm) if np.isfinite(norm) else None)
  iteration_history.append(record)
  if rank==0:
   print('MPI HISTORY NEWTON',target_step,target_fraction,it,norm,flush=True)
   result['last_iteration_history']=list(iteration_history);persist()
  if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>p['wall_budget_seconds']:
   raise RuntimeError('Declared total native recovery wall budget exhausted')
 problem.solver._snes.setMonitor(monitor);fields=[]
 for offset in range(p['steps_after_checkpoint']):
  target_step=payload['step']+offset;target_fraction=0.;root_u=current['u_old'].copy()
  assign(previous,payload['pressures'][target_step-1],root_u)
  anchor_force=float(residual(problem).norm('l2'));assert np.isfinite(anchor_force) and anchor_force<1e-8,'Strict preceding equation replay failed'
  if rank==0:
   result.setdefault('anchor_replays',[]).append(dict(step=target_step-1,native_residual_N=anchor_force));persist()
  fraction=0.;increment=options['initial_increment'];recovered=False
  for attempt in range(options['maximum_attempts']):
   target_fraction=min(1.,fraction+increment)
   hist={k:(1-target_fraction)*previous[k]+target_fraction*current[k] for k in current}
   if target_fraction==1.:hist={k:x.copy() for k,x in current.items()}
   pressure=float((1-target_fraction)*payload['pressures'][target_step-1]+target_fraction*payload['pressures'][target_step])
   assign(hist,pressure,root_u);iteration_history.clear();tick=time.perf_counter();error=None
   try:problem.solver.solve()
   except Exception as exc:error=str(exc)
   force=float(residual(problem).norm('l2'));snes=problem.solver._snes;candidate=adapter.gather('u');proof=None;proof_error=None
   valid=bool(error is None and snes.getConvergedReason()>0 and np.isfinite(force) and force<1e-8)
   if rank==0:
    valid=valid and bool(np.isfinite(candidate).all())
    if valid:
     try:proof=audit.pair(candidate,hist['u_old'])
     except Exception as exc:proof_error=str(exc);valid=False
    record=dict(step=target_step,attempt=attempt,fraction=target_fraction,increment=increment,accepted_engineering_root=valid,
     physical_endpoint=bool(valid and target_fraction==1.),SNES_reason=int(snes.getConvergedReason()),native_residual_N=force if np.isfinite(force) else None,
     newton_iterations=int(snes.getIterationNumber()),seconds=time.perf_counter()-tick,failure=error,positivity_failure=proof_error,iteration_history=list(iteration_history))
    result['attempts'].append(record)
    np.savez_compressed(P/p['iterate_checkpoint'],accepted=False,engineering_root=valid,fraction=target_fraction,step=target_step,
     u=candidate,**hist,protocol_sha256=result['protocol_sha256'])
    if valid:
     result['engineering_roots'].append(dict(step=target_step,fraction=target_fraction,proof=proof))
     np.savez_compressed(P/p['engineering_checkpoint'],physical_label=False,fraction=target_fraction,step=target_step,u=candidate,**hist)
    persist();print('MPI HISTORY ROOT',target_step,target_fraction,valid,force,flush=True)
   valid=comm.bcast(valid if rank==0 else None,root=0)
   if valid:
    root_u=comm.bcast(candidate,root=0);fraction=target_fraction;increment=min(options['maximum_increment'],increment*options['growth_factor'])
   else:increment*=.5
   if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>p['wall_budget_seconds']:
    result['failure']='Declared total native recovery wall budget exhausted';break
   if fraction==1.:recovered=True;break
   if increment<options['minimum_increment']:
    result['failure']='Adaptive homotopy exhausted declared minimum increment';break
  if not recovered:
   result.setdefault('failure','Adaptive homotopy exhausted declared attempt count');break
  # No interpolated histories survive the final admission: replay exact physical equation.
  assign(current,payload['pressures'][target_step],root_u);final_force=float(residual(problem).norm('l2'))
  assert final_force<1e-8 and snes.getConvergedReason()>0
  if payload['expected'] is not None:
   expected=payload['expected'][offset];difference=root_u-expected
   assert np.max(abs(difference))<1e-9
   if rank==0:
    result.setdefault('known_field_checks',[]).append(dict(step=target_step,maximum_absolute_difference_m=float(np.max(abs(difference)))))
  prior_current={k:x.copy() for k,x in current.items()}
  problem._update_fields();new_current={k:adapter.gather(k) for k in current}
  if rank==0:
   fields.append(root_u.copy());np.save(P/p['fields'],np.array(fields));np.savez_compressed(P/p['checkpoint'],completed_probe_steps=len(fields),u=root_u,**new_current)
   result['probe_checkpoint_sha256']=sha(P/p['checkpoint'])
   result['records'].append(dict(step=target_step,time_s=float(payload['times'][target_step]),accepted=True,SNES_reason=int(snes.getConvergedReason()),residual_N=final_force))
  current=comm.bcast(new_current,root=0);previous=prior_current;persist()
 result.update(complete=True,finished=True,all_requested_steps_accepted=len(fields)==p['steps_after_checkpoint'] if rank==0 else False,
  wall_seconds=comm.allreduce(time.perf_counter()-started,op=MPI.MAX))
 persist();comm.Barrier()

if __name__=='__main__':
 try:main()
 except BaseException:
  traceback.print_exc();comm.Abort(1)
