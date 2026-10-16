"""Engineering-only thread probe: identical frozen FE history and pressure query."""
from pathlib import Path
import sys,time,json,os,numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,export_geometry,assign_nodes,assign_history,residual,pressure_function,default_parameters_benchmark1,df
s=np.load(P/'experiments/logs/fine_thread_probe_state.npz');dt=.0025;step=int(s['step'])+1;g,mat,problem=setup(dt=dt,refinement_level=1,geometry_scales=[1,1]);problem.solver._solver.parameters['line_search']='bt';assign_history(problem,s['last_u'],s['last_v'],s['last_a']);p=float(pressure_function((0,1),default_parameters_benchmark1(),t_eval=(np.arange(1,401)*dt-.4*dt))[step]);problem.parameters['p'].assign(p)
best=float('inf');guess=s['last_u']
for u in [s['last_u'],s['last_u']+dt*s['last_v'],s['last_u']+dt*s['last_v']+.5*dt**2*s['last_a']]:
 assign_nodes(problem,u);r=residual(problem).norm('l2')
 if np.isfinite(r) and r<best:best=r;guess=u
assign_nodes(problem,guess);t=time.perf_counter();iterations,ok=problem.solver.solve();seconds=time.perf_counter()-t;r=residual(problem).norm('l2');assert int(problem.solver._snes.getConvergedReason())>0 and r<1e-6
result={'engineering_probe_not_surrogate_speedup':True,'threads':int(os.environ['OMP_NUM_THREADS']),'query_step':step,'seconds':seconds,'residual_l2_N':r,'iterations':int(iterations)};np.savez(P/f'experiments/logs/fine_thread_probe_{result["threads"]}.npz',u=problem.u.compute_vertex_values(g.mesh).reshape(3,-1).T);(P/f'experiments/logs/fine_thread_probe_{result["threads"]}.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
