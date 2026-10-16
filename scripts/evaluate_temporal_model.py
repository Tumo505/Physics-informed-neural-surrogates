"""Teacher-forced future-time evaluation plus recursive rollout and failure audit."""
from pathlib import Path
import json,sys,argparse
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.temporal_pod import TemporalPOD
from lv_surrogate.physics.fem_bridge import FEMBridge
from lv_surrogate.physics.kinematics import deformation_gradient
ap=argparse.ArgumentParser();ap.add_argument('--run',default='cardiac_temporal_pilot_v1');args=ap.parse_args()
torch.set_num_threads(2);selection=json.loads((P/f'experiments/manifests/{args.run}_selection.json').read_text());cfg=selection['config'];D=P/'data/processed/cardiac_temporal_ml';stats=dict(np.load(D/'normalizers.npz'));a=dict(np.load(D/'test.npz'));dt=float(stats['dt']);bridge=FEMBridge(P,threads=2);geom=np.load(P/'data/processed/cardiac/cohort.npz');X=torch.tensor(geom['coords']);C=torch.tensor(geom['cells'],dtype=torch.long);results={};predictions={}
def det(u):return float(torch.linalg.det(deformation_gradient(X,torch.tensor(u),C)).min())
try:
 for name in ['data',selection['selected']]:
  net=TemporalPOD(stats).double();net.load_state_dict(torch.load(P/f'experiments/checkpoints/{cfg["name"]}/{name}.pt',weights_only=False)['model']);net.eval()
  with torch.inference_mode():delta=net(torch.tensor(a['features'],dtype=torch.float64)).numpy().reshape(a['u'].shape)
  teacher=a['u_old']+dt*a['v_old']+.5*dt**2*a['a_old']+.36*dt**2*delta;res=[]
  for i,u in enumerate(teacher):res.append(float(bridge.evaluate(u,float(a['pressure'][i]),False,history={k:a[k][i] for k in ['u_old','v_old','a_old']})['loss'])**.5)
  old={k:a[k][0].copy() for k in ['u_old','v_old','a_old']};rollout=[];jac=[];failure=None
  for i,p in enumerate(a['pressure']):
   features=np.r_[*[old[k].reshape(-1)@stats['basis'].T for k in ['u_old','v_old','a_old']],p]
   with torch.inference_mode():da=net(torch.tensor(features[None],dtype=torch.float64)).numpy()[0].reshape(old['u_old'].shape)
   u=old['u_old']+dt*old['v_old']+.5*dt**2*old['a_old']+.36*dt**2*da
   if not np.isfinite(u).all():failure={'step':i,'reason':'nonfinite prediction'};break
   J=det(u)
   if J<=0:failure={'step':i,'time_s':float(a['time'][i]),'reason':'nonpositive element determinant','minimum_detF':J};break
   rollout.append(u);jac.append(J);anew=old['a_old']+da;vnew=old['v_old']+dt*old['a_old']+.7*dt*da;old={'u_old':u,'v_old':vnew,'a_old':anew}
  roll=np.array(rollout);n=len(roll);diff=teacher-a['u'];rerr=np.sqrt(np.mean((roll-a['u'][:n])**2,axis=(1,2))) if n else np.array([])
  results[name]={'teacher_forced_component_rmse_m':float(np.sqrt(np.mean(diff**2))),'teacher_forced_mean_normalized_residual_rms':float(np.mean(res)),'teacher_forced_minimum_detF':min(det(u) for u in teacher),'rollout_completed':n==len(a['u']),'rollout_steps_completed':n,'rollout_steps_required':len(a['u']),'rollout_component_rmse_m_on_valid_prefix':float(np.sqrt(np.mean((roll-a['u'][:n])**2))) if n else None,'rollout_relative_component_rmse_on_valid_prefix':float(np.sqrt(np.mean((roll-a['u'][:n])**2))/np.sqrt(np.mean(a['u'][:n]**2))) if n else None,'rollout_final_valid_frame_component_rmse_m':float(rerr[-1]) if n else None,'rollout_minimum_detF_valid_prefix':min(jac) if jac else None,'failure':failure}
  predictions[name]={'teacher':teacher,'rollout':roll,'rollout_frame_component_rmse_m':rerr}
  np.savez_compressed(P/f'experiments/metrics/{cfg["name"]}_{name}.npz',**predictions[name],time=a['time'],truth=a['u'])
 verdict={'models':results,'n_independent_geometries':1,'n_independent_trajectories':1,'n_test_time_steps':len(a['u']),'test_block_s':[float(a['time'][0]),float(a['time'][-1])],'scope':'single-seed temporal pilot; teacher forcing uses true histories; rollout uses only initial true state and prescribed future loads','limitations':['contiguous time holdout is not independent trajectory or geometry generalization','no conformal coverage or population-level confidence claim for temporal frames','valid-prefix rollout errors do not represent successful full rollouts']}
 (P/f'experiments/metrics/{cfg["name"]}_evaluation.json').write_text(json.dumps(verdict,indent=2));print(json.dumps(verdict,indent=2),flush=True)
finally:bridge.close()
