"""True-residual Armijo Newton with whole-element admissibility and rollback."""
import time
import numpy as np
from .reference_fem import df,residual
from mpi4py import MPI
from petsc4py import PETSc

def solve(problem,adapter,hist,audit,options,record,persist):
 comm=adapter.comm;rank=adapter.rank;started=time.perf_counter();matrix=df.PETScMatrix()
 def budget():
  if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>options['wall_budget_seconds']:raise RuntimeError('Admissible Newton wall budget exhausted')
 def admissible(candidate):
  proof=None;error=None
  if rank==0:
   try:proof=audit.pair(candidate,hist['u_old'])
   except Exception as exc:error=str(exc)
  return comm.bcast(error,root=0),proof
 initial=adapter.gather('u');error,_=admissible(initial);assert error is None,'Starting field fails whole-element admissibility: '+str(error)
 record['custom_newton_steps']=[]
 for iteration in range(options['maximum_custom_newton_iterations']+1):
  budget();R=residual(problem);norm=float(R.norm('l2'))
  if norm<options['custom_absolute_tolerance_N']:
   record['custom_force_N']=norm;record['custom_iterations']=iteration;return
  if iteration==options['maximum_custom_newton_iterations']:raise RuntimeError('Admissible Newton iteration ceiling reached')
  old=adapter.gather('u');old=comm.bcast(old,root=0);problem._problem.J(matrix,problem.u.vector());A=df.as_backend_type(matrix).mat()
  b=df.as_backend_type(R).vec().copy();b.scale(-1);direction=b.duplicate();direction.set(0.)
  ksp=PETSc.KSP().create(A.comm);ksp.setOperators(A);ksp.setType('gmres');ksp.setGMRESRestart(200);ksp.getPC().setType('gamg');ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED);ksp.setTolerances(rtol=1e-8,atol=1e-12,max_it=3000)
  ksp.setMonitor(lambda k,it,r:budget());ksp.solve(b,direction);true=b.duplicate();A.mult(direction,true);true.axpy(-1,b)
  step_record=dict(iteration=iteration,initial_force_N=norm,KSP_reason=int(ksp.getConvergedReason()),KSP_iterations=int(ksp.getIterationNumber()),linear_true_residual_N=float(true.norm()),linear_true_relative_residual=float(true.norm()/norm),trials=[])
  assert ksp.getConvergedReason()>0 and true.norm()<=max(1.1e-12,1e-7*norm),'Tight linear solve not qualified'
  local=problem.u.vector().copy();local.set_local(direction.getArray(readonly=True));local.apply('insert')
  arrays=comm.gather(local.get_local(),root=0);delta=None
  if rank==0:
   delta=np.empty(adapter.ndof)
   for ids,values in zip(adapter.parts,arrays):delta[ids]=values
  delta=comm.bcast(delta,root=0);accepted=False
  for halving in range(options['maximum_backtracks']+1):
   budget();alpha=2.**(-halving);candidate=old+alpha*delta;adapter.assign('u',candidate);trial=float(residual(problem).norm('l2'));proof=None;error=None
   armijo=bool(np.isfinite(trial) and trial<=(1-options['armijo_constant']*alpha)*norm)
   if armijo:error,proof=admissible(candidate)
   valid=armijo and error is None
   step_record['trials'].append(dict(alpha=alpha,native_force_N=trial if np.isfinite(trial) else None,armijo_passed=armijo,whole_element_admissible=valid,positivity_failure=error))
   if valid:
    accepted=True
    if rank==0:step_record['accepted_whole_element_proof']=proof
    break
  if not accepted:adapter.assign('u',old)
  if rank==0:record['custom_newton_steps'].append(step_record);persist();print('ADMISSIBLE NEWTON',iteration,norm,trial,accepted,alpha,flush=True)
  ksp.destroy();direction.destroy();b.destroy();true.destroy()
  if not accepted:raise RuntimeError('No admissible force-decreasing trial within declared backtracking ceiling')
