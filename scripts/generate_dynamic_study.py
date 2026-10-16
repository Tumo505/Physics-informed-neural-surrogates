"""Run one preregistered passive trajectory; exact source mechanics and histories."""
from pathlib import Path
import argparse,json,sys,time,hashlib,fcntl
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,export_geometry,vertex_map,residual,assign_nodes,assign_history,pressure_function,default_parameters_benchmark1,df
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--output-suffix',default='');args=ap.parse_args();cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());case=next(c for c in cfg['cases'] if c['id']==args.case)
out=P/'data/processed/cardiac_dynamic_v1'/(case['id']+args.output_suffix);out.mkdir(parents=True,exist_ok=True);lock=(out/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX);meta=out/'manifest.json';digest=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
if meta.exists():
 old=json.loads(meta.read_text())
 if old['case_sha256']!=digest:raise RuntimeError('Case changed; new run required')
 if old['complete']:
  assert hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest()==old['trajectory_sha256'];print('REUSED',case['id'],flush=True);sys.exit(0)
dt=case['dt'];geo,mat,problem=setup(dt=dt,geometry_scales=case['scales'],refinement_level=case.get('refinement',0));geometry=export_geometry(geo,mat,problem);area=float(df.assemble(df.Constant(1.)*problem.ds(1)));geometry.update(geometry_scales=case['scales'],force_scale_N=16070.954*area/np.sqrt(problem.u_space.dim()));np.savez_compressed(out/'geometry.npz',**geometry);v2d=vertex_map(problem)
n=int(round(1/dt));times=np.arange(1,n+1)*dt;par=default_parameters_benchmark1()
for k in ['t_sys_pre','t_dias_pre','gamma']:par[k]*=case['timing_scale']
pressures=case['amplitude']*pressure_function((0,1),parameters=par,t_eval=times-.4*dt)
values={k:[] for k in ['u','u_old','v_old','a_old']};records=[];start_step=0;prior=0.;saved=out/'restart.npz'
if saved.exists() and meta.exists():
 old=json.loads(meta.read_text());r=np.load(saved);start_step=int(r['step'])+1;assign_nodes(problem,r['last_u']);assign_history(problem,r['last_u'],r['last_v'],r['last_a']);values={k:list(r[k]) for k in values};records=old['steps'];prior=old['seconds']
start=time.perf_counter()
for i in range(start_step,n):
 h={k:getattr(problem,k).vector().get_local()[v2d].reshape(-1,3).copy() for k in ['u_old','v_old','a_old']};problem.parameters['p'].assign(float(pressures[i]));t=time.perf_counter();iterations,ok=problem.solver.solve()
 reason=int(problem.solver._snes.getConvergedReason())
 if reason<=0:raise RuntimeError(f'Nonconvergence {case["id"]} step{i}: SNES reason {reason}')
 u=problem.u.vector().get_local()[v2d].reshape(-1,3).copy();R=residual(problem).norm('l2')
 if not np.isfinite(u).all() or not np.isfinite(R) or R>1e-6:raise RuntimeError(f'Invalid FE residual {case["id"]} step{i}: {R}')
 C=geometry['cells'];X=geometry['coords'];F=np.eye(3)+np.einsum('eij,ejk->eik',(u[C[:,1:]]-u[C[:,:1]]).transpose(0,2,1),np.linalg.inv((X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1)))
 if np.linalg.det(F).min()<=0:raise RuntimeError(f'Inverted FE endpoint {case["id"]} step{i}')
 problem._update_fields();values['u'].append(u)
 for k in h:values[k].append(h[k])
 records.append({'step':i,'time':float(times[i]),'iterations':int(iterations),'snes_converged_reason':reason,'minimum_detF':float(np.linalg.det(F).min()),'residual_l2_N':float(R),'solve_seconds':time.perf_counter()-t})
 if (i+1)%10==0 or i==n-1:
  np.savez_compressed(saved,step=i,last_u=u,last_v=problem.v_old.vector().get_local()[v2d].reshape(-1,3),last_a=problem.a_old.vector().get_local()[v2d].reshape(-1,3),**values)
  result={'case':case,'case_sha256':digest,'complete':False,'completed_steps':i+1,'n_steps':n,'seconds':prior+time.perf_counter()-start,'steps':records,'scope':cfg['scope'],'source_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'src/lv_surrogate/physics/reference_fem.py']}}
  meta.write_text(json.dumps(result,indent=2));print(case['id'],i+1,'/',n,flush=True)
np.savez_compressed(out/'trajectory.npz',**values,pressure=pressures,time=times,dt=dt);result.update(complete=True,trajectory_sha256=hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest(),geometry_sha256=hashlib.sha256((out/'geometry.npz').read_bytes()).hexdigest());meta.write_text(json.dumps(result,indent=2));print('COMPLETE',case['id'],flush=True)
