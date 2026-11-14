"""State-551 experiment with a gentler reduction after a successful full step."""
import time
import numpy as np
from .reference_fem import df,residual
from mpi4py import MPI
from petsc4py import PETSc
from .regularization_recovery_policy import update_shift

def solve(problem,adapter,hist,audit,options,record,persist,*,policy='band_half',timings=None,reuse_jacobian=False,stop_after=None,reuse_gamg=False,log_phases=False):
 comm=adapter.comm;rank=adapter.rank;started=time.perf_counter();matrix=df.PETScMatrix();mu=options['initial_regularization'];ksp=None;reference_force=None
 good_full_steps = 0
 iteration = 0
 def measured(name, function, *args):
  if log_phases and rank == 0:
   print(f'NEWTON {iteration} shift={mu:g} begin {name}', flush=True)
  start = time.perf_counter()
  try:
   return function(*args)
  finally:
   if timings is not None:
    timings[name] = timings.get(name, 0.) + time.perf_counter() - start
   if log_phases and rank == 0:
    print(f'NEWTON {iteration} shift={mu:g} end {name} seconds={time.perf_counter()-start:.3f}', flush=True)
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
   try:proof=measured('positivity',audit.pair,candidate,hist['u_old'])
   except Exception as exc:error=str(exc)
  return comm.bcast(error,root=0),proof
 old=comm.bcast(adapter.gather('u'),root=0);error,_=positive(old);assert error is None,error
 record['custom_newton_steps']=[]
 reuse_next = False
 previous_fresh_shift = None
 while True:
  reused = reuse_next
  reuse_next = False
  budget();R=measured('iteration_residual',residual,problem);norm=float(R.norm('l2'));assert np.isfinite(norm)
  if norm<options['custom_absolute_tolerance_N']:
   record.update(custom_force_N=norm,custom_iterations=iteration)
   if ksp is not None:ksp.destroy()
   return
  if iteration==options['maximum_custom_newton_iterations']:raise RuntimeError('Regularized Newton iteration ceiling reached')
  if reference_force is None:reference_force=norm
  eta=min(1e-4,max(1e-8,1e-4*np.sqrt(norm/reference_force)))
  old=comm.bcast(adapter.gather('u'),root=0)
  if not reused:
   measured('jacobian_assembly',problem._problem.J,matrix,problem.u.vector())
  A=df.as_backend_type(matrix).mat()
  diag=A.getDiagonal();scale=diag.copy();scale.abs();values=scale.getArray();values[:]=np.maximum(values,scale.norm(PETSc.NormType.NORM_INFINITY)*1e-12)
  shifted=diag.copy();shifted.axpy(mu,scale);A.setDiagonal(shifted);A.assemble()
  b=df.as_backend_type(R).vec().copy();b.scale(-1);delta=b.duplicate();delta.set(0.)
  if ksp is None:
   ksp=PETSc.KSP().create(A.comm);ksp.setType('gmres');ksp.setGMRESRestart(200);ksp.getPC().setType('gamg');ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED);ksp.setMonitor(lambda k,it,r:budget())
  reuse_pc = reuse_gamg and reused and previous_fresh_shift == mu
  ksp.setOperators(A)
  ksp.setTolerances(rtol=eta,atol=1e-12,max_it=3000)
  true = b.duplicate()
  limit = max(1.1e-12,1.1*eta*norm)

  def linear_attempt(reuse):
   if log_phases and rank == 0:
    print(f'NEWTON {iteration} linear attempt reuse_gamg={reuse}', flush=True)
   budget()
   ksp.getPC().setReusePreconditioner(reuse)
   delta.set(0.)
   setup_started = time.perf_counter()
   measured('ksp_setup',ksp.setUp)
   setup_seconds = time.perf_counter()-setup_started
   budget()
   solve_started = time.perf_counter()
   measured('ksp_solve',ksp.solve,b,delta)
   solve_seconds = time.perf_counter()-solve_started
   A.mult(delta,true)
   true.axpy(-1,b)
   residual_norm = float(true.norm())
   reason = int(ksp.getConvergedReason())
   return dict(reuse=reuse,setup_seconds=setup_seconds,solve_seconds=solve_seconds,
               reason=reason,iterations=int(ksp.getIterationNumber()),
               true_residual_N=residual_norm if np.isfinite(residual_norm) else None,
               qualified=reason>0 and residual_norm<=limit)

  tick = time.perf_counter()
  first = linear_attempt(reuse_pc)
  attempts = [first]
  fallback = None
  if not first['qualified'] and (not reused or reuse_pc):
   fallback = dict(first_KSP_reason=first['reason'],first_iterations=first['iterations'],
                   first_true_residual_N=first['true_residual_N'])
   attempts.append(linear_attempt(False))
  last = attempts[-1]
  previous_fresh_shift = mu if not reused and last['qualified'] else None
  true_force = last['true_residual_N']
  sr=dict(iteration=iteration,linear_forcing=eta,reuse_requested=reuse_pc,
          jacobian_reused=reused,rebuild_fallback=fallback,initial_force_N=norm,
          regularization=mu,KSP_reason=last['reason'],KSP_iterations=last['iterations'],
          true_shifted_linear_residual_N=true_force,
          true_shifted_linear_relative_residual=true_force/norm if true_force is not None else None,
          linear_seconds=time.perf_counter()-tick,linear_attempts=attempts,trials=[])
  qualified=last['qualified']
  if not qualified and reused:
   A.setDiagonal(diag);A.assemble()
   for v in (diag,scale,shifted,b,delta,true):v.destroy()
   if rank==0:
    record.setdefault('jacobian_reuse_rejections',[]).append(dict(sr,reason='linear residual'))
    persist()
   continue
  if not qualified:
   if rank==0:record['custom_newton_steps'].append(sr);persist()
   if stop_after == 1:
    A.setDiagonal(diag);A.assemble()
    for v in (diag,scale,shifted,b,delta,true):v.destroy()
    if ksp is not None:ksp.destroy()
    return
   raise RuntimeError('Shifted linear direction failed true residual qualification')
  direction=canonical(delta);A.setDiagonal(diag);A.assemble();jdelta=b.duplicate();A.mult(delta,jdelta);model=b.duplicate();accepted=False
  for power in range(1 if reused else options['maximum_backtracks']+1):
   budget();alpha=2.**(-power);candidate=old+alpha*direction;adapter.assign('u',candidate);force=float(measured('trial_residual',residual,problem).norm('l2'));error=None;proof=None
   armijo=bool(np.isfinite(force) and force<=(1-options['armijo_constant']*alpha)*norm)
   if armijo:error,proof=positive(candidate)
   adapter.assign('u',old)  # Keep a safe accepted iterate even if a later budget check interrupts.
   valid=armijo and error is None;model.set(0.);model.axpy(alpha,jdelta);model.axpy(-1,b);predicted=.5*(norm**2-model.norm()**2)
   ratio=(.5*(norm**2-force**2)/predicted) if np.isfinite(force) and predicted>0 else None
   sr['trials'].append(dict(alpha=alpha,native_force_N=force if np.isfinite(force) else None,armijo_passed=armijo,whole_element_admissible=valid,positivity_failure=error,actual_to_predicted_merit_ratio=ratio))
   if valid:
    adapter.assign('u',candidate);accepted=True
    reuse_next = reuse_jacobian and not reused and power == 0 and ratio is not None and ratio > .75
    if rank==0:sr['accepted_whole_element_proof']=proof
    if policy == 'fixed_shift':
     pass
    elif policy == 'two_good_steps':
     mu, good_full_steps = update_shift(
      mu, good_full_steps, accepted=True, full_step=power == 0, ratio=ratio,
      minimum=options['minimum_regularization'], maximum=options['maximum_regularization'])
    elif ratio is not None and ratio > .75 and power == 0:
     # The failed step alternated between these bounds. Try the intermediate shift.
     shrink = .5 if .00390625 <= mu <= .015625 else .25
     mu = max(options['minimum_regularization'], mu * shrink)
    elif ratio is None or ratio<.25 or power>0:mu=min(options['maximum_regularization'],mu*4)
    break
  if reused and not accepted:
   for v in (diag,scale,shifted,b,delta,true,jdelta,model):v.destroy()
   if rank==0:
    record.setdefault('jacobian_reuse_rejections',[]).append(dict(sr,reason='full trial rejected'))
    persist()
   continue
  if not accepted and policy == 'fixed_shift':
   if rank==0:
    record['custom_newton_steps'].append(dict(sr,accepted_custom_update=False))
    persist()
   raise RuntimeError('Fixed shift produced no admissible descent')
  if not accepted:
   mu=min(options['maximum_regularization'],mu*10)
   good_full_steps=0
  sr.update(accepted_custom_update=accepted,next_regularization=mu,good_full_steps_at_shift=good_full_steps)
  if rank==0:record['custom_newton_steps'].append(sr);persist();print('REGULARIZED NEWTON',iteration,norm,force,accepted,mu,flush=True)
  for v in [diag,scale,shifted,b,delta,true,jdelta,model]:v.destroy()
  if stop_after != 1 and not accepted and mu>=options['maximum_regularization']:raise RuntimeError('Regularization ceiling without admissible descent')
  iteration += 1
  if stop_after is not None and iteration == stop_after:
   if ksp is not None:ksp.destroy()
   return
