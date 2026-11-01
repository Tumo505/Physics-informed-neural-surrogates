"""Diagonal-scaled regularized Newton; exact physical residual admission unchanged."""
import time
import numpy as np
from .reference_fem import df,residual
from mpi4py import MPI
from petsc4py import PETSc

def solve(problem,adapter,hist,audit,options,record,persist):
 comm=adapter.comm;rank=adapter.rank;started=time.perf_counter();matrix=df.PETScMatrix();mu=options['initial_regularization'];ksp=None;reference_force=None
 def budget():
  if comm.allreduce(time.perf_counter()-started,op=MPI.MAX)>options['wall_budget_seconds']:raise RuntimeError('Regularized Newton wall budget exhausted')
 def canonical(v):
  arrays=comm.gather(np.array(v.getArray(readonly=True)),root=0);x=None
  if rank==0:
   x=np.empty(adapter.ndof)
   for ids,values in zip(adapter.parts,arrays):x[ids]=values
  return comm.bcast(x,root=0)
 def positive(candidate):
  error=None;proof=None
  if rank==0:
   try:proof=audit.pair(candidate,hist['u_old'])
   except Exception as exc:error=str(exc)
  return comm.bcast(error,root=0),proof
 old=comm.bcast(adapter.gather('u'),root=0);error,_=positive(old);assert error is None,error
 record['custom_newton_steps']=[]
 for iteration in range(options['maximum_custom_newton_iterations']+1):
  budget();R=residual(problem);norm=float(R.norm('l2'));assert np.isfinite(norm)
  if norm<options['custom_absolute_tolerance_N']:
   record.update(custom_force_N=norm,custom_iterations=iteration)
   if ksp is not None:ksp.destroy()
   return
  if iteration==options['maximum_custom_newton_iterations']:raise RuntimeError('Regularized Newton iteration ceiling reached')
  if reference_force is None:reference_force=norm
  eta=min(1e-4,max(1e-8,1e-4*np.sqrt(norm/reference_force)))
  old=comm.bcast(adapter.gather('u'),root=0);problem._problem.J(matrix,problem.u.vector());A=df.as_backend_type(matrix).mat()
  diag=A.getDiagonal();scale=diag.copy();scale.abs();values=scale.getArray();values[:]=np.maximum(values,scale.norm(PETSc.NormType.NORM_INFINITY)*1e-12)
  shifted=diag.copy();shifted.axpy(mu,scale);A.setDiagonal(shifted);A.assemble()
  b=df.as_backend_type(R).vec().copy();b.scale(-1);delta=b.duplicate();delta.set(0.)
  if ksp is None:
   ksp=PETSc.KSP().create(A.comm);ksp.setType('gmres');ksp.setGMRESRestart(200);ksp.getPC().setType('gamg');ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED);ksp.setMonitor(lambda k,it,r:budget())
  ksp.setOperators(A);ksp.setTolerances(rtol=eta,atol=1e-12,max_it=3000);ksp.getPC().setReusePreconditioner(False)
  tick=time.perf_counter();ksp.solve(b,delta);true=b.duplicate();A.mult(delta,true);true.axpy(-1,b)
  fallback=None
  limit=max(1.1e-12,1.1*eta*norm)
  if ksp.getConvergedReason()<=0 or true.norm()>limit:
   fallback=dict(first_KSP_reason=int(ksp.getConvergedReason()),first_iterations=int(ksp.getIterationNumber()),first_true_residual_N=float(true.norm()))
   delta.set(0.);ksp.getPC().setReusePreconditioner(False);ksp.setUp();ksp.solve(b,delta);A.mult(delta,true);true.axpy(-1,b)
  sr=dict(iteration=iteration,linear_forcing=eta,reuse_requested=False,rebuild_fallback=fallback,initial_force_N=norm,regularization=mu,KSP_reason=int(ksp.getConvergedReason()),KSP_iterations=int(ksp.getIterationNumber()),true_shifted_linear_residual_N=float(true.norm()),true_shifted_linear_relative_residual=float(true.norm()/norm),linear_seconds=time.perf_counter()-tick,trials=[])
  qualified=ksp.getConvergedReason()>0 and true.norm()<=limit
  if not qualified:
   if rank==0:record['custom_newton_steps'].append(sr);persist()
   raise RuntimeError('Shifted linear direction failed true residual qualification')
  direction=canonical(delta);A.setDiagonal(diag);A.assemble();jdelta=b.duplicate();A.mult(delta,jdelta);model=b.duplicate();accepted=False
  for power in range(options['maximum_backtracks']+1):
   budget();alpha=2.**(-power);candidate=old+alpha*direction;adapter.assign('u',candidate);force=float(residual(problem).norm('l2'));error=None;proof=None
   armijo=bool(np.isfinite(force) and force<=(1-options['armijo_constant']*alpha)*norm)
   if armijo:error,proof=positive(candidate)
   adapter.assign('u',old)  # Keep a safe accepted iterate even if a later budget check interrupts.
   valid=armijo and error is None;model.set(0.);model.axpy(alpha,jdelta);model.axpy(-1,b);predicted=.5*(norm**2-model.norm()**2)
   ratio=(.5*(norm**2-force**2)/predicted) if np.isfinite(force) and predicted>0 else None
   sr['trials'].append(dict(alpha=alpha,native_force_N=force if np.isfinite(force) else None,armijo_passed=armijo,whole_element_admissible=valid,positivity_failure=error,actual_to_predicted_merit_ratio=ratio))
   if valid:
    adapter.assign('u',candidate);accepted=True
    if rank==0:sr['accepted_whole_element_proof']=proof
    if ratio is not None and ratio>.75 and power==0:mu=max(options['minimum_regularization'],mu*.25)
    elif ratio is None or ratio<.25 or power>0:mu=min(options['maximum_regularization'],mu*4)
    break
  if not accepted:mu=min(options['maximum_regularization'],mu*10)
  sr.update(accepted_custom_update=accepted,next_regularization=mu)
  if rank==0:record['custom_newton_steps'].append(sr);persist();print('REGULARIZED NEWTON',iteration,norm,force,accepted,mu,flush=True)
  for v in [diag,scale,shifted,b,delta,true,jdelta,model]:v.destroy()
  if not accepted and mu>=options['maximum_regularization']:raise RuntimeError('Regularization ceiling without admissible descent')
