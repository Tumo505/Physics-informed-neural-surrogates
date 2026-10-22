"""Read-only native residual/tangent/linear diagnostics at preserved failed-step states."""
import argparse,fcntl,json,sys,time,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.parallel_fem import ParallelP2,sha
from lv_surrogate.physics.reference_fem import df,residual
from mpi4py import MPI
from petsc4py import PETSc
comm=MPI.COMM_WORLD;rank=comm.rank

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args();pp=P/args.protocol;p=json.loads(pp.read_text())
 assert comm.size==p['parallel_ranks'];out=P/p['output'];result=dict(complete=False,records=[],protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),scope=p['scope'])
 def persist():
  if rank==0:
   tmp=out.with_suffix('.pending.json');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(out)
 payload=None
 if rank==0:
  lock=(P/'experiments/logs'/f'{out.stem}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not out.exists()
  for name,h in p['source_sha256'].items():assert sha(P/name)==h
  recovery=json.loads((P/p['recovery_protocol']).read_text());rr=json.loads((P/recovery['output']).read_text());assert rr['finished'] and not rr['all_requested_steps_accepted']
  parent=json.loads((P/recovery['resume_probe_protocol']).read_text());prior=json.loads((P/recovery['previous_history_protocol']).read_text())
  saved=np.load(P/parent['checkpoint']);old=np.load(P/prior['checkpoint']);engineering=np.load(P/recovery['engineering_checkpoint']);trial=np.load(P/recovery['iterate_checkpoint'])
  histories=lambda x:{k:x[k].copy() for k in ['u_old','v_old','a_old']}
  root=P/'data/processed'/recovery['reference'];m=json.loads((root/'manifest.json').read_text());data=np.load(root/'inputs.npz');step=rr['start_step'];pressures=data['pressure']
  states=[dict(name='accepted_physical_anchor',u=saved['u'].copy(),history=histories(old),pressure=float(pressures[step-1]),fraction=0.,recorded_force=rr['anchor_replays'][0]['native_residual_N'],linear_check=False)]
  for name,x,linear in [('converged_engineering_root',engineering,False),('stalled_engineering_iterate',trial,True)]:
   f=float(x['fraction']);hist=histories(x)
   for k in hist:np.testing.assert_allclose(hist[k],(1-f)*old[k]+f*saved[k],rtol=1e-13,atol=1e-14)
   record=[a for a in rr['attempts'] if a['fraction']==f][-1]
   states.append(dict(name=name,u=x['u'].copy(),history=hist,pressure=float((1-f)*pressures[step-1]+f*pressures[step]),fraction=f,recorded_force=record['native_residual_N'],linear_check=linear))
  states.append(dict(name='exact_failed_equation_at_stalled_initialization',u=trial['u'].copy(),history=histories(saved),pressure=float(pressures[step]),fraction=1.,recorded_force=None,linear_check=True))
  direction=trial['u']-engineering['u'];direction/=max(float(np.max(abs(direction))),1e-30)
  rng=np.random.default_rng(p['direction_seed']);random=rng.standard_normal(len(direction));random/=np.max(abs(random))
  payload=dict(reference=recovery['reference'],cfg=m['config'],states=states,directions=[direction,random])
  result['input_sha256']={path:sha(P/path) for path in [p['recovery_protocol'],recovery['output'],parent['checkpoint'],prior['checkpoint'],recovery['engineering_checkpoint'],recovery['iterate_checkpoint']]}
 payload=comm.bcast(payload,root=0);persist();adapter=ParallelP2(P,payload['reference'],payload['cfg']);problem=adapter.problem;matrix=df.PETScMatrix();started=time.perf_counter()
 for state in payload['states']:
  for k,x in state['history'].items():adapter.assign(k,x)
  adapter.assign('u',state['u']);problem.parameters['p'].assign(state['pressure']);R=residual(problem);norm=float(R.norm('l2'))
  record=dict(name=state['name'],fraction=state['fraction'],native_force_N=norm,recorded_force_N=state['recorded_force'],directions=[])
  if state['recorded_force'] is not None:
   np.testing.assert_allclose(norm,state['recorded_force'],rtol=1e-5,atol=1e-12);record['force_replay_passed']=True
  t=time.perf_counter();problem._problem.J(matrix,problem.u.vector());record['J_assembly_seconds']=comm.allreduce(time.perf_counter()-t,op=MPI.MAX)
  for j,direction in enumerate(payload['directions']):
   d=problem.u.vector().copy();d.set_local(direction[adapter.mapping]);d.apply('insert');jv=R.copy();matrix.mult(d,jv);jn=float(jv.norm('l2'));checks=[]
   for eps in p['finite_difference_steps_m']:
    adapter.assign('u',state['u']+eps*direction);plus=residual(problem)
    adapter.assign('u',state['u']-eps*direction);minus=residual(problem)
    plus.axpy(-1.,minus);plus*=1/(2*eps);plus.axpy(-1.,jv);error=float(plus.norm('l2'))/max(jn,1e-30)
    checks.append(dict(epsilon_m=eps,relative_error=error))
   adapter.assign('u',state['u']);record['directions'].append(dict(direction=j,tangent_norm_N_per_m=jn,finite_differences=checks,passed=max(x['relative_error'] for x in checks[-2:])<p['tangent_relative_tolerance']))
  if state['linear_check']:
   A=df.as_backend_type(matrix).mat();b=df.as_backend_type(R).vec().copy();b.scale(-1);solution=b.duplicate();solution.set(0.)
   ksp=PETSc.KSP().create(A.comm);ksp.setOperators(A);ksp.setType('gmres');ksp.setGMRESRestart(200);ksp.getPC().setType('gamg')
   ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED);ksp.setTolerances(rtol=1e-8,atol=1e-12,max_it=3000);history=[]
   def monitor(k,it,value):
    if rank==0:print('DIAGNOSTIC KSP',state['name'],it,value,flush=True)
    history.append(dict(iteration=int(it),reported_residual_N=float(value)))
    if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>p['native_wall_budget_seconds']:raise RuntimeError('Bounded diagnostic native wall budget exhausted')
   ksp.setMonitor(monitor);t=time.perf_counter();ksp.solve(b,solution);true=b.duplicate();A.mult(solution,true);true.axpy(-1.,b)
   record['linear_solve']=dict(KSP_reason=int(ksp.getConvergedReason()),iterations=int(ksp.getIterationNumber()),seconds=comm.allreduce(time.perf_counter()-t,op=MPI.MAX),true_residual_N=float(true.norm()),true_relative_residual=float(true.norm()/max(b.norm(),1e-30)),rhs_norm_N=float(b.norm()),requested_relative_tolerance=1e-8,requested_absolute_tolerance_N=1e-12,history=history)
   # Linear solve evidence is diagnostic. It is not a nonlinear step or a physical label.
   ksp.destroy();true.destroy();solution.destroy();b.destroy()
  result['records'].append(record);persist()
  if rank==0:print('STATE DIAGNOSTIC COMPLETE',state['name'],norm,flush=True)
  if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>p['native_wall_budget_seconds']:raise RuntimeError('Bounded diagnostic native wall budget exhausted')
 result.update(complete=True,tangent_checks_passed=all(d['passed'] for r in result['records'] for d in r['directions']),wall_seconds=comm.allreduce(time.perf_counter()-started,op=MPI.MAX),no_physical_states_generated=True);persist();comm.Barrier()
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc();comm.Abort(1)
