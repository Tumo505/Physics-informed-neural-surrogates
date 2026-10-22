"""Bounded whole-element-admissible Armijo Newton, followed by native stopping confirmation."""
import argparse,fcntl,json,sys,time,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.parallel_fem import ParallelP2,sha
from lv_surrogate.physics.reference_fem import residual
from lv_surrogate.physics.bernstein import BernsteinAudit
from lv_surrogate.physics.admissible_newton import solve as admissible_solve
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
  scope=p['scope'])
 def persist():
  if rank==0:
   out=P/p['output'];tmp=out.with_suffix('.pending.json');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(out)
 if rank==0:
  lock=(P/'experiments/logs'/f'{Path(p["output"]).stem}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (P/p['output']).exists(),'Never retry an existing output'
  diagnostic=json.loads((P/p['diagnostic_gate']).read_text());assert diagnostic['complete'] and diagnostic['tangent_checks_passed']
  assert all(z['linear_solve']['KSP_reason']>0 and z['linear_solve']['true_relative_residual']<1e-7 for z in diagnostic['records'] if 'linear_solve' in z)
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
 settings=problem.solver._solver.parameters
 settings.update(dict(method='newtonls',maximum_iterations=100,absolute_tolerance=1e-8,relative_tolerance=0.))
 settings['krylov_solver'].update(dict(relative_tolerance=1e-8,absolute_tolerance=1e-12,maximum_iterations=3000))
 problem.solver._snes.setUseEW(False)
 if rank==0:audit=BernsteinAudit(np.load(root/'inputs.npz'))
 else:audit=None
 current=payload['current'];previous=payload['previous'];started=time.perf_counter();history=[];fields=[]
 def persist_iterate():
  state=adapter.gather('u')
  if rank==0:
   np.savez_compressed(P/p['iterate_checkpoint'],accepted=False,step=step,u=state,**current,protocol_sha256=result['protocol_sha256'])
 def monitor(snes,it,norm):
  history.append(dict(iteration=int(it),native_residual_N=float(norm) if np.isfinite(norm) else None))
  if rank==0:
   result['last_iteration_history']=list(history);persist();print('TRUST REGION',step,it,norm,flush=True)
  if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>p['wall_budget_seconds']:raise RuntimeError('Declared native trust-region wall budget exhausted')
 problem.solver._snes.setMonitor(monitor)
 for offset in range(p['steps_after_checkpoint']):
  step=payload['step']+offset;old=current['u_old'];dt=cfg['dt_s'];alpha=float(problem.parameters['alpha_f'])
  for k,x in current.items():adapter.assign(k,x)
  problem.parameters['p'].assign(float(payload['pressures'][step]))
  stage=(alpha*previous['u_old']+(1-alpha)*old-alpha*old)/(1-alpha)
  guesses=[old,old+dt*current['v_old'],old+dt*current['v_old']+.5*dt**2*current['a_old'],stage]
  initial=[];best=float('inf');chosen=None
  for j,u in enumerate(guesses):
   adapter.assign('u',u);force=float(residual(problem).norm('l2'));initial.append(force if np.isfinite(force) else None)
   if np.isfinite(force) and force<best:best=force;chosen=j
  assert chosen is not None;adapter.assign('u',guesses[chosen]);snes=problem.solver._snes
  # Select the minimum finite-force initial guess that also passes whole-element admissibility.
  admissible_choices=[]
  for j in sorted([j for j,x in enumerate(initial) if x is not None],key=lambda j:initial[j]):
   error=None
   if rank==0:
    try:audit.pair(guesses[j],old)
    except Exception as exc:error=str(exc)
   error=comm.bcast(error,root=0)
   if error is None:chosen=j;break
  else:raise RuntimeError('No admissible original initializer')
  adapter.assign('u',guesses[chosen]);snes.setType('newtonls')
  result.setdefault('initializations',[]).append(dict(step=step,predictor_residuals_N=initial,selected_predictor=chosen))
  persist();history.clear();error=None;t=time.perf_counter()
  custom_record=dict(step=step)
  result.setdefault('admissible_solves',[]).append(custom_record)
  try:
   remaining=p['wall_budget_seconds']-(time.perf_counter()-started);options=dict(p['admissible_newton'],wall_budget_seconds=remaining)
   admissible_solve(problem,adapter,current,audit,options,custom_record,persist)
   # A real fresh native SNES confirms stopping at this unchanged equation. It must take zero updates.
   before=adapter.gather('u');problem.solver.solve();after=adapter.gather('u')
   assert snes.getIterationNumber()==0,'Native confirmation unexpectedly changed the custom endpoint'
   if rank==0:np.testing.assert_allclose(after,before,rtol=0,atol=1e-14)
  except Exception as exc:error=str(exc)
  force=float(residual(problem).norm('l2'));candidate=adapter.gather('u');valid=bool(error is None and snes.getConvergedReason()>0 and np.isfinite(force) and force<1e-8)
  if rank==0:
   proof=None;positivity_error=None
   if valid:
    try:proof=audit.pair(candidate,old)
    except Exception as exc:positivity_error=str(exc);valid=False
   known_error=None
   if valid and payload['expected'] is not None:
    known_error=float(np.max(abs(candidate-payload['expected'][offset])));valid=known_error<1e-9
   record=dict(step=step,time_s=float(payload['times'][step]),accepted=valid,SNES_reason=int(snes.getConvergedReason()),residual_N=force if np.isfinite(force) else None,
    nonlinear_iterations=int(snes.getIterationNumber()),seconds=time.perf_counter()-t,SNES_type=snes.getType(),KSP_type=snes.getKSP().getType(),PC_type=snes.getKSP().getPC().getType(),KSP_tolerances=list(snes.getKSP().getTolerances()),EW_effective=bool(snes.getUseEW()),
    failure=error,positivity_failure=positivity_error,known_field_max_difference_m=known_error,whole_element_proof=proof,iteration_history=list(history))
   result['attempts'].append(record);persist()
  valid=comm.bcast(valid if rank==0 else None,root=0)
  assert snes.getType()=='newtonls' and not snes.getUseEW()
  if not valid:
   persist_iterate();result['failure']='Strict admissible-Newton endpoint admission failed';break
  previous={k:x.copy() for k,x in current.items()};problem._update_fields();updated={k:adapter.gather(k) for k in current};current=comm.bcast(updated,root=0)
  if rank==0:
   fields.append(candidate);np.save(P/p['fields'],np.array(fields));np.savez_compressed(P/p['checkpoint'],completed_probe_steps=len(fields),u=candidate,**current)
   result['probe_checkpoint_sha256']=sha(P/p['checkpoint']);result['records'].append({k:v for k,v in record.items() if k not in ['whole_element_proof','iteration_history']})
  persist()
 result.update(complete=True,finished=True,all_requested_steps_accepted=len(fields)==p['steps_after_checkpoint'] if rank==0 else False,wall_seconds=comm.allreduce(time.perf_counter()-started,op=MPI.MAX))
 persist();comm.Barrier()
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc();comm.Abort(1)
