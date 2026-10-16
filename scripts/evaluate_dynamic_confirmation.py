"""Independent frozen confirmation; reuse primary calibration with no model choice."""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np,torch
from scipy.stats import binomtest,spearmanr
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.dynamic_graph import DynamicGraph,newmark_operators
from lv_surrogate.physics.fem_bridge import FEMBridge
from lv_surrogate.eval.fallback import InputDomainGate
torch.set_num_threads(2);cfg=json.loads((P/'configs/physics/dynamic_confirmation.json').read_text());freeze=json.loads((P/'experiments/manifests/dynamic_confirmation_freeze.json').read_text());assert hashlib.sha256((P/'configs/physics/dynamic_confirmation.json').read_bytes()).hexdigest()==freeze['confirmation_config_sha256']
for path,digest in freeze['checkpoint_sha256'].items():assert hashlib.sha256((P/path).read_bytes()).hexdigest()==digest
while not (P/'experiments/metrics/dynamic_final_evaluation.json').exists():print('WAIT PRIMARY FROZEN EVALUATION',flush=True);time.sleep(30)
primary=json.loads((P/'experiments/metrics/dynamic_final_evaluation.json').read_text());tcfg=json.loads((P/'configs/physics/dynamic_training.json').read_text());nets={};stats={};top=torch.tensor(np.load(P/'data/processed/cardiac_geometry_ml/topology.npy'));cluster=np.load(P/'data/processed/cardiac_geometry_ml/cluster_labels.npy');counts=np.bincount(cluster,minlength=64);vm,am=[torch.tensor(x) for x in newmark_operators(200,.005)];frames=tcfg['validation_residual_frames'];rng=np.random.default_rng(17303);bridge=FEMBridge(P,threads=2);records={key:[] for key in primary['models']};members={key:[] for key in records};gate=InputDomainGate({'radial':(.95,1.05),'longitudinal':(.95,1.05),'amplitude':(.9,1.1),'timing':(.95,1.05)})
for key in records:
 regime,label=key.split('/');stats[regime]=dict(np.load(P/f'data/processed/{tcfg["data_tag"]}_{regime}/normalizers.npz'));nets[key]=[]
 for seed in tcfg['seeds']:
  net=DynamicGraph(stats[regime]).double();net.load_state_dict(torch.load(P/f'experiments/checkpoints/{tcfg["name"]}_{regime}/{label}_seed{seed}.pt',map_location='cpu',weights_only=False)['model']);net.eval();nets[key].append(net)
def jac(u,g):
 X=g['coords'];C=g['cells'];inv=np.linalg.inv((X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1));minimum=1.;inverted=0
 for chunk in np.array_split(u,20):
  F=np.eye(3)+np.einsum('beij,ejk->beik',(chunk[:,C[:,1:]]-chunk[:,C[:,:1]]).transpose(0,1,3,2),inv);J=np.linalg.det(F);minimum=min(minimum,float(J.min()));inverted+=int((J<=0).sum())
 return minimum,inverted/(len(u)*len(C))
def residuals(q,net,case,a):
 v=vm@q;acc=am@q;zero=torch.zeros_like(q[:1]);old=[torch.cat((zero,x[:-1]),0) for x in [q,v,acc]];rr=[]
 for t in frames:
  h={k:net.decode(x[t]).reshape(-1,3).numpy() for k,x in zip(['u_old','v_old','a_old'],old)};r=float(bridge.evaluate(net.decode(q[t]).reshape(-1,3).numpy(),float(a['pressure'][t]),False,history=h,dt=.005,geometry_path=P/f'data/processed/cardiac_dynamic_v1/{case}/geometry.npz')['loss'])
  if not np.isfinite(r):return None
  rr.append(r**.5)
 return float(np.mean(rr))
try:
 for c in cfg['cases']:
  root=P/f'data/processed/cardiac_dynamic_v1/{c["id"]}'
  while True:
   try:
    if json.loads((root/'manifest.json').read_text()).get('complete'):break
   except (FileNotFoundError,json.JSONDecodeError):pass
   print('WAIT CONFIRMATION',c['id'],flush=True);time.sleep(30)
  m=json.loads((root/'manifest.json').read_text());assert m['case']==c;assert hashlib.sha256((root/'trajectory.npz').read_bytes()).hexdigest()==m['trajectory_sha256'];a=dict(np.load(root/'trajectory.npz'));g=dict(np.load(root/'geometry.npz'));x=np.c_[g['coords']/.1,g['fiber'],g['sheet']];coarse=np.zeros((64,9));np.add.at(coarse,cluster,x);coarse/=counts[:,None];delta=coarse[top[:,1],:3]-coarse[top[:,0],:3];inputs=[torch.tensor(v[None],dtype=torch.float64) for v in [coarse,np.r_[(np.array(c['scales'])-1)/.05,0.],np.c_[delta,np.linalg.norm(delta,axis=-1)],a['pressure']/16070.954]]
  for key,models in nets.items():
   net=models[0]
   with torch.inference_mode():
    q=torch.stack([n(*inputs[:3],top,inputs[3])[0] for n in models]);average=q.mean(0);u=net.decode(average).reshape(200,-1,3).numpy();physical=q*net.coefficient_scale;centered=physical-physical.mean(0);gram=net.basis@net.basis.T;variance=float(torch.einsum('mtr,rs,mts->',centered,gram,centered)/(4*200*net.basis.shape[1]));force=residuals(average,net,c['id'],a)
    target=torch.tensor(a['u'].reshape(200,-1));projection=target@net.basis.T;norm2=target.square().sum()
    for k,seed in enumerate(tcfg['seeds']):
     cc=physical[k];mse=(torch.einsum('tr,rs,ts->',cc,gram,cc)-2*(cc*projection).sum()+norm2)/target.numel();members[key].append({'case':c['id'],'geometry':c['geometry'],'seed':seed,'component_rmse_m':float(mse.clamp_min(0).sqrt()),'sampled_normalized_force_rms':residuals(q[k],net,c['id'],a)})
   error=u-a['u'];nodal=np.linalg.norm(error,axis=-1);minimum,inverted=jac(u,g);desc=dict(zip(['radial','longitudinal','amplitude','timing'],[*c['scales'],c['amplitude'],c['timing_scale']]));records[key].append({'case':c['id'],'geometry':c['geometry'],'component_rmse_m':float(np.sqrt(np.mean(error**2))),'relative_component_rmse':float(np.sqrt(np.mean(error**2)/np.mean(a['u']**2))),'mean_nodal_l2_m':float(nodal.mean()),'max_all_nodes_times_l2_m':float(nodal.max()),'sampled_normalized_force_rms':force,'minimum_detF_all200_endpoints':minimum,'inverted_element_time_fraction':inverted,'completed_full_cycle':bool(minimum>0),'mean_coordinate_variance_m2':variance,'routing':gate.decide(desc,minimum_detF=minimum)})
  print('CONFIRMATION EVALUATED',c['id'],flush=True)
 def summary(rr,radii):
  coverage=[]
  for nominal,radius in radii.items():
   covered=sum(r['max_all_nodes_times_l2_m']<=radius for r in rr);ci=binomtest(covered,len(rr)).proportion_ci();coverage.append({'nominal':float(nominal),'covered':covered,'n':len(rr),'coverage':covered/len(rr),'radius_m':radius,'binomial95':[ci.low,ci.high]})
  v=np.array([r['mean_coordinate_variance_m2'] for r in rr]);e=np.array([r['mean_nodal_l2_m'] for r in rr]);return {'n_geometries':len(rr),'mean_component_rmse_m':float(np.mean([r['component_rmse_m'] for r in rr])),'mean_nodal_l2_m':float(e.mean()),'mean_sampled_normalized_force_rms':float(np.mean([r['sampled_normalized_force_rms'] for r in rr])) if all(r['sampled_normalized_force_rms'] is not None for r in rr) else None,'full_cycles_without_inversions':sum(r['completed_full_cycle'] for r in rr),'minimum_detF':min(r['minimum_detF_all200_endpoints'] for r in rr),'coverage':coverage,'spearman_uncertainty_error':float(spearmanr(v,e).statistic),'selective_risk':[{'retained':f,'n':max(1,int(f*len(rr))),'mean_nodal_l2_m':float(e[np.argsort(v)[:max(1,int(f*len(rr)))]].mean())} for f in [.25,.5,.75,1.]],'FE_routes':sum(r['routing']['action']=='FE' for r in rr),'per_case':rr}
 results={}
 for key,rr in records.items():
  combined=primary['models'][key]['splits']['test']['per_case']+rr;results[key]={'confirmation':summary(rr,primary['models'][key]['radii_m']),'pooled_primary_and_confirmation':summary(combined,primary['models'][key]['radii_m']),'per_seed_confirmation':members[key]}
 comparisons={}
 for regime in ['full','limited']:
  keys=[key for key in results if key.startswith(regime+'/')];comparisons[regime]={}
  for split in ['confirmation','pooled_primary_and_confirmation']:
   b=results[keys[0]][split];p=results[keys[1]][split];valid=all(r['sampled_normalized_force_rms'] is not None for r in [*b['per_case'],*p['per_case']]);difference=None;interval=None
   if valid:
    delta=np.array([x['sampled_normalized_force_rms']-y['sampled_normalized_force_rms'] for x,y in zip(p['per_case'],b['per_case'])]);difference=float(delta.mean());interval=np.quantile([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(4000)],[.025,.975]).tolist()
   ratio=p['mean_component_rmse_m']/b['mean_component_rmse_m'];comparisons[regime][split]={'RMSE_ratio_physics_over_data':ratio,'paired_mean_force_RMS_difference_physics_minus_data':difference,'conditional_geometry_bootstrap95':interval,'all_residuals_finite':valid,'H1_descriptive_supported':bool(valid and difference<0 and ratio<=1.1)}
 result={'complete':True,'models':results,'comparisons':comparisons,'preferred_from_primary_validation_only':primary['preferred_from_validation_only'],'calibration':'unchanged20 primary calibration geometries and fixed radii; no confirmation recalibration','bootstrap_scope':'fixed5-member models; independent geometry resampling; excludes retraining and FE discretization uncertainty','limitations':['one affine benchmark family','pooled20 geometry confirmation improves precision but is not anatomical validation','mechanics nine predefined frames, endpoint inversion checks all200 for ensembles','confirmation per-member residuals/accuracy are reported, per-member endpoint-J checks only in primary study']};(P/'experiments/metrics/dynamic_confirmation_evaluation.json').write_text(json.dumps(result,indent=2));print('FROZEN CONFIRMATION COMPLETE',flush=True)
finally:bridge.close()
