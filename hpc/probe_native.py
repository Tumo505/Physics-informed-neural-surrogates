"""Two known R2 equations on allocated MPI ranks; no untouched test response fields."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'src'),str(P/'hpc')]
from lv_surrogate.physics.parallel_fem import ParallelP2,MPI
from lv_surrogate.physics.reference_fem import residual
from lv_surrogate.physics.admissible_newton_regularized_forcing import solve
from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit
from registry import atomic,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--output',required=True);ap.add_argument('--fields',required=True);ap.add_argument('--checkpoint',required=True);a=ap.parse_args();s=json.loads((P/a.spec).read_text());snapshot=json.loads((P/s['snapshot']).read_text());root=P/'data/processed'/snapshot['reference'];m=json.loads((root/'manifest.json').read_text());data=np.load(root/'inputs.npz');f=np.load(root/'displacement_coefficients.npy',mmap_mode='r');cp=np.load(P/s['known_start_checkpoint']);start=int(cp['completed_steps']);comm=MPI.COMM_WORLD;rank=comm.rank;assert comm.size==s['ranks']
 adapter=ParallelP2(P,snapshot['source_reference'],m['config']);problem=adapter.problem;problem.solver._solver.parameters.update(dict(method='newtonls',maximum_iterations=100,absolute_tolerance=1e-8,relative_tolerance=0.));problem.solver._snes.setUseEW(False);current={k:cp[k].copy() for k in ['u_old','v_old','a_old']};previous=np.array(f[start-2]);audit=ThreadedBernsteinAudit(data) if rank==0 else None;out=P/a.output
 if rank==0:assert not out.exists()
 records=[];fields=[];report=dict(complete=False,passed=False,start_step=start,records=records,physical_new_state_generated=False,known_case_only=True)
 for i in range(start,start+2):
  for k,v in current.items():adapter.assign(k,v)
  dt=m['config']['dt_s'];old=current['u_old'];alpha=.4;guesses=[old,old+dt*current['v_old'],old+dt*current['v_old']+.5*dt**2*current['a_old'],(alpha*previous+(1-alpha)*old-alpha*old)/(1-alpha)];problem.parameters['p'].assign(float(data['pressure'][i]));initial=[]
  for u in guesses:adapter.assign('u',u);value=float(residual(problem).norm('l2'));initial.append(value if np.isfinite(value) else None)
  chosen=None
  for j in sorted([j for j,x in enumerate(initial) if x is not None],key=lambda j:initial[j]):
   error=None
   if rank==0:
    try:audit.pair(guesses[j],old)
    except Exception as exc:error=str(exc)
   if comm.bcast(error,root=0) is None:chosen=j;break
  assert chosen is not None;adapter.assign('u',guesses[chosen]);trace={};report['active_trace']=trace;tick=time.perf_counter()
  def persist():
   if rank==0:atomic(out,report)
  solve(problem,adapter,current,audit,dict(s['admissible_newton'],wall_budget_seconds=1800),trace,persist);problem.solver.solve();force=float(residual(problem).norm('l2'));assert problem.solver._snes.getConvergedReason()>0 and problem.solver._snes.getIterationNumber()==0 and force<1e-8
  u=adapter.gather('u');difference=None
  if rank==0:difference=float(np.max(abs(u-f[i])));assert difference<1e-9;fields.append(u);proof=audit.pair(u,old);records.append(dict(step=i,native_force_N=force,SNES_reason=int(problem.solver._snes.getConvergedReason()),native_updates=0,known_field_max_difference_m=difference,solve_seconds=time.perf_counter()-tick,whole_element_proof=proof,custom_trace=trace));persist()
  previous=current['u_old'].copy();problem._update_fields()
  # All ranks must participate in each canonical gather.
  gathered={k:adapter.gather(k) for k in current};current=comm.bcast(gathered,root=0)
 if rank==0:
  np.save(P/a.fields,np.array(fields));np.savez_compressed(P/a.checkpoint,completed_probe_steps=2,**current,u=fields[-1]);report.update(complete=True,passed=True,fields_sha256=sha(P/a.fields),checkpoint_sha256=sha(P/a.checkpoint));atomic(out,report);audit.close()
 comm.Barrier()

if __name__=='__main__':
 try:main()
 except SystemExit as exc:
  if exc.code in (None,0):raise
  MPI.COMM_WORLD.Abort(1)
  raise
 except BaseException:
  import traceback
  traceback.print_exc();MPI.COMM_WORLD.Abort(1)
