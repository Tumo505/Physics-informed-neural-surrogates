"""Independent native finite differences for the actual network parameter directions."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,df,residual
from lv_surrogate.physics.quadratic_solver import make_quadratic_problem
e=json.loads((P/'experiments/metrics/quadratic_network_gradient_export.json').read_text())
g=json.loads((P/'experiments/metrics/quadratic_network_geometry.json').read_text())
path=P/e['fields_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==e['fields_sha256'];a=np.load(path)
geo,mat,base=setup(dt=.005,geometry_scales=g['training_case']['scales']);params=dict(base.parameters);params['function_space']='P_2'
problem=make_quadratic_problem(geo,mat,params,'mumps');losses=[]
for i,load in enumerate(a['pressure']):
 for name in ['u','u_old','v_old','a_old']:
  f=getattr(problem,name);f.vector().set_local(a[name][i]);f.vector().apply('insert')
 problem.parameters['p'].assign(float(load));R=residual(problem);losses.append(.5*R.norm('l2')**2)
print('NATIVE NETWORK CASES',len(losses),flush=True)
records=[]
for r in e['records']:
 rows=[i for i,c in enumerate(e['cases']) if c['direction']==r['direction']];base=next(i for i in rows if e['cases'][i]['sign']==0)
 derivatives=[]
 for eps in e['epsilons']:
  plus=next(i for i in rows if e['cases'][i]['epsilon']==eps and e['cases'][i]['sign']==1)
  minus=next(i for i in rows if e['cases'][i]['epsilon']==eps and e['cases'][i]['sign']==-1)
  fd=(losses[plus]-losses[minus])/(2*eps);ad=r['autograd_directional_derivative']
  derivatives.append(dict(epsilon=eps,native_FD=fd,relative_error=abs(fd-ad)/max(abs(ad),1e-30)))
 convergence=abs(derivatives[-1]['native_FD']-derivatives[-2]['native_FD'])/max(abs(derivatives[-1]['native_FD']),1e-30)
 baseline=abs(losses[base]-r['torch_baseline_loss'])/max(abs(losses[base]),1e-30)
 passed=max(d['relative_error'] for d in derivatives[-2:])<.01 and convergence<.005 and baseline<1e-7
 records.append(dict(frame=r['frame'],autograd_directional_derivative=ad,finite_differences=derivatives,
  baseline_loss_relative_error=baseline,two_smallest_epsilon_consistency=convergence,passed=passed))
result=dict(complete=True,passed=all(r['passed'] for r in records),records=records,
 declared_tolerances=dict(native_parameter_FD_relative_error=.01,FD_epsilon_consistency=.005,baseline_loss=1e-7),
 network_export_sha256=hashlib.sha256((P/'experiments/metrics/quadratic_network_gradient_export.json').read_bytes()).hexdigest(),
 fields_sha256=e['fields_sha256'],checkpoint_sha256=e['checkpoint_sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 pretraining_architecture_chain_verified=all(r['passed'] for r in records),new_trained_full_P2_checkpoint_verified=False,
 scope=e['scope']+' Independent native assembly at all centered parameter perturbations verifies force-loss gradients through current and predicted histories. CPU64 only; future GPU/full-P2 trained checkpoint checks remain required.')
out=P/'experiments/metrics/quadratic_network_parameter_gradient.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True);assert result['passed']
