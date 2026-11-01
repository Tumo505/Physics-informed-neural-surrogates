"""Read-only failed-step Newton-direction and Cauchy-model descent diagnostic."""
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.parallel_fem import ParallelP2,sha
from lv_surrogate.physics.reference_fem import df,residual
from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit
from mpi4py import MPI
from petsc4py import PETSc
comm=MPI.COMM_WORLD;rank=comm.rank

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args();pp=P/args.protocol;p=json.loads(pp.read_text());assert comm.size==p['parallel_ranks'];out=P/p['output'];payload=None
 if rank==0:
  assert not out.exists();proof=json.loads((P/p['required_prefix_audit']).read_text());assert proof['passed'] and proof['accepted_prefix_only']
  source=json.loads((P/p['failed_protocol']).read_text());result=json.loads((P/source['output']).read_text());assert result['finished'] and not result['all_requested_steps_accepted']
  for f,h in p['source_sha256'].items():assert sha(P/f)==h
  trial_path=P/source['iterate_checkpoint'];x=np.load(trial_path);assert not bool(x['accepted']) and str(x['protocol_sha256'])==sha(P/p['failed_protocol'])
  root=P/'data/processed'/source['reference'];m=json.loads((root/'manifest.json').read_text());data=np.load(root/'inputs.npz');step=int(x['step']);last=result['attempts'][-1];assert last['step']==step and not last['accepted']
  payload=dict(reference=source['reference'],cfg=m['config'],state={k:x[k].copy() for k in ['u','u_old','v_old','a_old']},pressure=float(data['pressure'][step]),step=step,time_s=float(data['time'][step]),recorded_force_N=last['residual_N'],input_sha256={str(trial_path.relative_to(P)):sha(trial_path),p['failed_protocol']:sha(P/p['failed_protocol']),p['required_prefix_audit']:sha(P/p['required_prefix_audit'])})
 payload=comm.bcast(payload,root=0);adapter=ParallelP2(P,payload['reference'],payload['cfg']);problem=adapter.problem
 for k,v in payload['state'].items():adapter.assign(k,v)
 problem.parameters['p'].assign(payload['pressure']);R=residual(problem);norm=float(R.norm('l2'));np.testing.assert_allclose(norm,payload['recorded_force_N'],rtol=1e-5,atol=1e-12)
 A0=df.PETScMatrix();problem._problem.J(A0,problem.u.vector());A=df.as_backend_type(A0).mat();b=df.as_backend_type(R).vec().copy();b.scale(-1);delta=b.duplicate();delta.set(0.);ksp=PETSc.KSP().create(A.comm);ksp.setOperators(A);ksp.setType('gmres');ksp.setGMRESRestart(200);ksp.getPC().setType('gamg');ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED);ksp.setTolerances(rtol=1e-8,atol=1e-12,max_it=3000);started=time.perf_counter()
 def monitor(k,it,r):
  if rank==0:print('BACKTRACK DIAGNOSTIC KSP',it,r,flush=True)
  if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>p['native_wall_budget_seconds']:raise RuntimeError('Declared diagnostic wall budget exhausted')
 ksp.setMonitor(monitor);ksp.solve(b,delta);true=b.duplicate();A.mult(delta,true);true.axpy(-1,b)
 assert ksp.getConvergedReason()>0 and true.norm()<=max(1.1e-12,1e-7*norm)
 def canonical(vector):
  arrays=comm.gather(np.array(vector.getArray(readonly=True)),root=0);x=None
  if rank==0:
   x=np.empty(adapter.ndof)
   for ids,values in zip(adapter.parts,arrays):x[ids]=values
  return comm.bcast(x,root=0)
 newton=canonical(delta);g=b.duplicate();A.multTranspose(df.as_backend_type(R).vec(),g);jg=g.duplicate();A.mult(g,jg);cauchy_scale=(g.norm()/max(jg.norm(),1e-300))**2;cauchy=g.copy();cauchy.scale(-cauchy_scale);gradient_step=canonical(cauchy)
 report=dict(complete=False,physical_step_recovered=False,no_physical_state_generated=True,step=payload['step'],time_s=payload['time_s'],native_force_N=norm,KSP_reason=int(ksp.getConvergedReason()),KSP_iterations=int(ksp.getIterationNumber()),true_linear_residual_N=float(true.norm()),true_linear_relative_residual=float(true.norm()/norm),newton_direction_max_abs_m=float(np.max(abs(newton))),newton_direction_L2_m=float(delta.norm()),gradient_norm=float(g.norm()),cauchy_scale=cauchy_scale,directions=[],input_sha256=payload['input_sha256'],source_sha256=sha(Path(__file__)),scope=p['scope'])
 audit=ThreadedBernsteinAudit(np.load(P/'data/processed'/payload['reference']/'inputs.npz')) if rank==0 else None
 for name,direction in [('tight_Newton',newton),('Cauchy_linearized_residual_model',gradient_step)]:
  tests=[];first=None
  for power in range(p['maximum_trial_halvings']+1):
   if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>p['native_wall_budget_seconds']:raise RuntimeError('Declared diagnostic wall budget exhausted')
   alpha=2.**(-power);candidate=payload['state']['u']+alpha*direction;adapter.assign('u',candidate);force=float(residual(problem).norm('l2'));merit=bool(np.isfinite(force) and force<=(1-1e-4*alpha)*norm);error=None;proof=None
   if merit and rank==0:
    try:proof=audit.pair(candidate,payload['state']['u_old'])
    except Exception as exc:error=str(exc)
   error=comm.bcast(error,root=0);valid=merit and error is None;tests.append(dict(alpha=alpha,native_force_N=force if np.isfinite(force) else None,armijo_passed=merit,whole_element_passed=valid,positivity_failure=error))
   if valid:
    first=dict(alpha=alpha,native_force_N=force,proof=proof);break
  if rank==0:report['directions'].append(dict(name=name,maximum_absolute_direction_m=float(np.max(abs(direction))),trials=tests,first_admissible_descent=first))
  adapter.assign('u',payload['state']['u'])
 if rank==0:
  report.update(complete=True,wall_seconds=time.perf_counter()-started);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');audit.close()
 ksp.destroy();comm.Barrier()
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc();comm.Abort(1)
