"""History-consistent passive benchmark trajectory, with resumable solver state."""
from pathlib import Path
import argparse,json,sys,time,hashlib
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,assign_nodes,assign_history,residual,vertex_map,pressure_function,default_parameters_benchmark1,df
ap=argparse.ArgumentParser();ap.add_argument('--dt',type=float);ap.add_argument('--end',type=float);ap.add_argument('--name');args=ap.parse_args()
cfg=json.loads((P/'configs/physics/trajectory.json').read_text());dt=args.dt or cfg['dt_s'];end=args.end or cfg['end_s'];name=args.name or cfg['name']
out=P/'data/processed'/name;out.mkdir(exist_ok=True);geo,mat,problem=setup(dt=dt);v2d=vertex_map(problem)
times=np.arange(dt,end,dt);pressures=np.asarray(pressure_function((0,1),parameters=default_parameters_benchmark1(),t_eval=times-.4*dt))
records=[];start_step=0;prior_seconds=0.;saved=out/'state.npz';fields=[];old_u=[];old_v=[];old_a=[]
if saved.exists():
 with np.load(saved) as a:
  start_step=int(a['step'])+1;assign_nodes(problem,a['u']);assign_history(problem,a['u'],a['v'],a['a'])
 d=json.loads((P/f'experiments/metrics/{name}.json').read_text());records=d['steps'];prior_seconds=d['seconds']
 print('RESUME',start_step,flush=True)
start=time.perf_counter()
try:
 for i in range(start_step,len(times)):
  histories={k:getattr(problem,k).vector().get_local()[v2d].reshape(-1,3).copy() for k in ['u_old','v_old','a_old']}
  problem.parameters['p'].assign(float(pressures[i]));t=time.perf_counter();iterations,ok=problem.solver.solve()
  if not ok:raise RuntimeError(f'Trajectory solve failed at step {i}')
  r=residual(problem).norm('l2');u=problem.u.vector().get_local()[v2d].reshape(-1,3).copy()
  np.savez_compressed(out/f'step_{i:04d}.npz',u=u,**histories,pressure=pressures[i],time=times[i],dt=dt)
  problem._update_fields();records.append({'step':i,'time_s':float(times[i]),'pressure_Pa':float(pressures[i]),'iterations':int(iterations),'residual_l2_N':float(r),'seconds':time.perf_counter()-t,'max_nodal_displacement_m':float(np.linalg.norm(u,axis=-1).max())})
  if i%20==0:print(json.dumps(records[-1]),flush=True)
  if (i+1)%cfg['checkpoint_every']==0 or i==len(times)-1:
   # Persist the state first; metadata then matches the durable restart point.
   tmp=out/'state.tmp.npz';np.savez(tmp,step=i,u=u,v=problem.v_old.vector().get_local()[v2d].reshape(-1,3),a=problem.a_old.vector().get_local()[v2d].reshape(-1,3));tmp.replace(saved)
   result={'config':cfg,'name':name,'dt_s':dt,'end_s':end,'n_steps':len(times),'completed_steps':i+1,'complete':i==len(times)-1,'independent_geometries':1,'pressure_timing':'t_n+(1-alpha_f)*dt; pinned source convention','initial_state':'u=v=a=0','activation':'zero throughout; passive variant, not full active benchmark','seconds':prior_seconds+time.perf_counter()-start,'steps':records,'source_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'src/lv_surrogate/physics/reference_fem.py']}}
   (P/f'experiments/metrics/{name}.json').write_text(json.dumps(result,indent=2))
except Exception as exc:
 (out/'failure.json').write_text(json.dumps({'step':i,'error':str(exc),'time_s':float(times[i])},indent=2));raise
print('TRAJECTORY COMPLETE',name,flush=True)
