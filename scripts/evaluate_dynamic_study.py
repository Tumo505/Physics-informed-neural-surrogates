"""Frozen geometry/trajectory comparison, calibration, mechanics and OOD routing."""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np,torch
from scipy.stats import binomtest,spearmanr
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.dynamic_graph import DynamicGraph,newmark_operators
from lv_surrogate.physics.fem_bridge import FEMBridge
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.eval.calibration import conformal_radius
from lv_surrogate.eval.fallback import InputDomainGate
torch.set_num_threads(2);cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());traincfg=json.loads((P/'configs/physics/dynamic_training.json').read_text());models={};hashes={};stats={};top=torch.tensor(np.load(P/'data/processed/cardiac_geometry_ml/topology.npy'));cluster=np.load(P/'data/processed/cardiac_geometry_ml/cluster_labels.npy');counts=np.bincount(cluster,minlength=64);selections={}
for regime in traincfg['regimes']:
 name=traincfg['name']+'_'+regime;m=json.loads((P/f'experiments/manifests/{name}.json').read_text());assert m['complete'];selections[regime]=m['selection'];stats[regime]=dict(np.load(P/f'data/processed/cardiac_dynamic_ml_{regime}/normalizers.npz'))
 for label in ['data',m['selection']['selected']]:
  key=regime+'/'+label;models[key]=[]
  for seed in traincfg['seeds']:
   path=P/f'experiments/checkpoints/{name}/{label}_seed{seed}.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==m['checkpoints'][path.name];hashes[str(path.relative_to(P))]=digest;net=DynamicGraph(stats[regime]).double();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);net.eval();models[key].append(net)
freeze={'checkpoint_sha256':hashes,'config_sha256':hashlib.sha256((P/'configs/physics/dynamic_study.json').read_bytes()).hexdigest(),'training_selection':selections,'scope':'all model choices frozen before opening calibration/test/shift arrays; both predeclared methods/regimes retained regardless of outcome'};(P/'experiments/manifests/dynamic_final_freeze.json').write_text(json.dumps(freeze,indent=2))
bridge=FEMBridge(P,threads=2);rng=np.random.default_rng(17302);vm,am=[torch.tensor(x) for x in newmark_operators(200,.005)];frames=traincfg['validation_residual_frames'];gate=InputDomainGate({'radial':(.95,1.05),'longitudinal':(.95,1.05),'amplitude':(.9,1.1),'timing':(.95,1.05)})
def inputs(c,geo,a):
 x=np.c_[geo['coords']/.1,geo['fiber'],geo['sheet']];coarse=np.zeros((64,9));np.add.at(coarse,cluster,x);coarse/=counts[:,None];delta=coarse[top[:,1],:3]-coarse[top[:,0],:3]
 return [torch.tensor(v[None],dtype=torch.float64) for v in [coarse,np.r_[(np.array(c['scales'])-1)/.05,0.],np.c_[delta,np.linalg.norm(delta,axis=-1)]]]+[torch.tensor(a['pressure'][None]/16070.954)]
def prediction(nets,b):
 with torch.inference_mode():return torch.stack([net(*b[:3],top,b[3])[0] for net in nets])
def jacobian_stats(u,geo):
 X=geo['coords'];C=geo['cells'];inv=np.linalg.inv((X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1));minimum=1.;inverted=0
 for chunk in np.array_split(u,20):
  du=(chunk[:,C[:,1:]]-chunk[:,C[:,:1]]).transpose(0,1,3,2);F=np.eye(3)+np.einsum('beij,ejk->beik',du,inv);J=np.linalg.det(F);minimum=min(minimum,float(J.min()));inverted+=int((J<=0).sum())
 return minimum,inverted/(len(u)*len(C))
def evaluate_prediction(q,net,c,a,geo,seed=None,full_metrics=True):
 with torch.inference_mode():
  u=net.decode(q).reshape(200,-1,3).numpy();v=vm@q;acc=am@q;z=torch.zeros_like(q[:1]);old=[torch.cat((z,x[:-1]),0) for x in [q,v,acc]]
 error=u-a['u'];minimum,inverted=jacobian_stats(u,geo) if full_metrics else (None,None);R=[];failure=None;strain=[];energy=[];own=CardiacFirstStep(geo,dt=.005,analytic_piola=True) if full_metrics else None
 for t in frames:
  h={k:net.decode(x[t]).reshape(-1,3).detach().numpy() for k,x in zip(['u_old','v_old','a_old'],old)}
  r=bridge.evaluate(u[t],float(a['pressure'][t]),False,history=h,geometry_path=P/f'data/processed/cardiac_dynamic_v1/{c["id"]}/geometry.npz',dt=.005);loss=float(r['loss'])
  if not np.isfinite(loss):failure='nonfinite native residual';break
  R.append(loss**.5)
  if own:
   X=own.X;C=own.cells;F=own.I+torch.einsum('eai,eaj->eij',torch.tensor(u[t])[C],own.grad);Ft=own.I+torch.einsum('eai,eaj->eij',torch.tensor(a['u'][t])[C],own.grad);E=.5*(F.transpose(-1,-2)@F-own.I);Et=.5*(Ft.transpose(-1,-2)@Ft-own.I);strain.append(float(((E-Et).square().sum((-1,-2)).sqrt()*own.vol).sum()/own.vol.sum()))
   if min(float(torch.linalg.det(F).min()),float(torch.linalg.det(Ft).min()))>0:
    W=(own.energy_density(F[:,None].expand(-1,len(own.weights),-1,-1))*own.weights).sum(-1).dot(own.vol);Wt=(own.energy_density(Ft[:,None].expand(-1,len(own.weights),-1,-1))*own.weights).sum(-1).dot(own.vol);energy.append(float(abs(W-Wt)/max(abs(Wt),1e-12)))
 e=np.linalg.norm(error,axis=-1);result={'case':c['id'],'geometry':c['geometry'],'seed':seed,'component_rmse_m':float(np.sqrt(np.mean(error**2))),'mean_nodal_l2_m':float(e.mean()),'p95_nodal_time_l2_m':float(np.quantile(e,.95)),'max_all_nodes_times_l2_m':float(e.max()),'relative_component_rmse':float(np.sqrt(np.mean(error**2)/np.mean(a['u']**2))),'sampled_normalized_force_rms':float(np.mean(R)) if not failure else None,'mechanics_failure':failure,'minimum_detF_all200_endpoints':minimum,'inverted_element_time_fraction':inverted,'completed_full_cycle':bool(minimum>0 and np.isfinite(u).all()) if full_metrics else None,'sampled_volume_mean_strain_error':float(np.mean(strain)) if strain else None,'sampled_relative_elastic_energy_error':float(np.mean(energy)) if energy else None}
 return result,u
records={key:{s:[] for s in ['validation','calibration','test','load_shift','shift']} for key in models};members={key:[] for key in models};latent={key:{} for key in models};preferred=None
try:
 for c in [x for x in cfg['cases'] if x['split'] in ['validation','calibration','test','load_shift','shift']]:
  if c['split']=='calibration' and preferred is None:
   fullkeys=[k for k in models if k.startswith('full/')];base_rmse=np.mean([r['component_rmse_m'] for r in records[fullkeys[0]]['validation']]);eligible=[k for k in fullkeys if np.mean([r['component_rmse_m'] for r in records[k]['validation']])<=1.1*base_rmse and all(r['sampled_normalized_force_rms'] is not None for r in records[k]['validation'])];preferred=min(eligible,key=lambda k:np.mean([r['sampled_normalized_force_rms'] for r in records[k]['validation']]));freeze['preferred_from_validation_only']=preferred;freeze['validation_ensemble_results']={k:records[k]['validation'] for k in fullkeys};(P/'experiments/manifests/dynamic_final_freeze.json').write_text(json.dumps(freeze,indent=2))
  root=P/f'data/processed/cardiac_dynamic_v1/{c["id"]}';meta=json.loads((root/'manifest.json').read_text());assert meta['complete'];assert hashlib.sha256((root/'trajectory.npz').read_bytes()).hexdigest()==meta['trajectory_sha256'];a=dict(np.load(root/'trajectory.npz'));geo=dict(np.load(root/'geometry.npz'));b=inputs(c,geo,a)
  for key,nets in models.items():
   q=prediction(nets,b);latent[key][c['id']]=q.numpy();meanq=q.mean(0);net=nets[0]
   if c['split']=='calibration':
    with torch.inference_mode():u=net.decode(meanq).reshape(200,-1,3).numpy()
    rec={'case':c['id'],'geometry':c['geometry'],'max_all_nodes_times_l2_m':float(np.linalg.norm(u-a['u'],axis=-1).max())}
   else:
    rec,u=evaluate_prediction(meanq,net,c,a,geo)
    with torch.inference_mode():fields=net.decode(q).reshape(5,200,-1,3);rec['mean_coordinate_variance_m2']=float(fields.var(0,unbiased=True).mean())
    if c['split']=='test':
     for k,seed in enumerate(traincfg['seeds']):rr,_=evaluate_prediction(q[k],net,c,a,geo,seed=seed,full_metrics=False);members[key].append(rr)
     if c['id']==next(x['id'] for x in cfg['cases'] if x['split']=='test'):
      np.savez_compressed(P/f'experiments/metrics/dynamic_{key.replace("/","_")}_example.npz',coords=geo['coords'],cells=geo['cells'],facets=geo['facets'],facet_markers=geo['facet_markers'],prediction=u,truth=a['u'],time=a['time'],pressure=a['pressure'])
   if c['split'] in ['test','load_shift','shift']:
    desc=dict(zip(['radial','longitudinal','amplitude','timing'],[*c['scales'],c['amplitude'],c['timing_scale']]));rec['routing']=gate.decide(desc,minimum_detF=rec['minimum_detF_all200_endpoints'])
   records[key][c['split']].append(rec)
  print('FROZEN EVALUATED',c['id'],c['split'],flush=True)
 results={}
 for key,split_records in records.items():
  scores=np.array([r['max_all_nodes_times_l2_m'] for r in split_records['calibration']]);radii={str(1-alpha):conformal_radius(scores,alpha) for alpha in [.1,.05]};result={'calibration_n_geometries':len(scores),'radii_m':radii,'splits':{},'matched_seed_test_cases':members[key]}
  for split,rr in split_records.items():
   if split=='calibration':continue
   coverage=[]
   for nominal,radius in radii.items():
    scores_by_geometry=[max(r['max_all_nodes_times_l2_m'] for r in rr if r['geometry']==g) for g in sorted(set(r['geometry'] for r in rr))];covered=sum(score<=radius for score in scores_by_geometry);ng=len(scores_by_geometry);ci=binomtest(covered,ng).proportion_ci();coverage.append({'nominal':float(nominal),'covered':covered,'n':ng,'coverage':covered/ng,'binomial95':[ci.low,ci.high] if split=='test' else None,'radius_m':radius,'grouping':'all held-out trajectories from a geometry jointly; no exchangeability guarantee under shift'})
   result['splits'][split]={'n_geometries':len(set(r['geometry'] for r in rr)),'n_trajectories':len(rr),'mean_component_rmse_m':float(np.mean([r['component_rmse_m'] for r in rr])),'mean_nodal_l2_m':float(np.mean([r['mean_nodal_l2_m'] for r in rr])),'mean_sampled_normalized_force_rms':float(np.mean([r['sampled_normalized_force_rms'] for r in rr])) if all(r['sampled_normalized_force_rms'] is not None for r in rr) else None,'full_cycles_without_inversions':sum(r['completed_full_cycle'] for r in rr),'minimum_detF':min(r['minimum_detF_all200_endpoints'] for r in rr),'coverage':coverage,'FE_routes':sum(r.get('routing',{}).get('action')=='FE' for r in rr),'per_case':rr}
   if split=='test':
    result['splits'][split]['spearman_uncertainty_error']=float(spearmanr([r['mean_coordinate_variance_m2'] for r in rr],[r['mean_nodal_l2_m'] for r in rr]).statistic)
  results[key]=result
 comparisons={}
 for regime in ['full','limited']:
  keys=[k for k in results if k.startswith(regime+'/')];base=results[keys[0]]['splits']['test']['per_case'];phys=results[keys[1]]['splits']['test']['per_case'];delta=np.array([b['sampled_normalized_force_rms']-a['sampled_normalized_force_rms'] for a,b in zip(base,phys)]);boot=[delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(2000)];comparisons[regime]={'paired_geometry_mean_force_RMS_difference_physics_minus_data':float(delta.mean()),'conditional_geometry_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'RMSE_ratio_physics_over_data':results[keys[1]]['splits']['test']['mean_component_rmse_m']/results[keys[0]]['splits']['test']['mean_component_rmse_m'],'bootstrap_scope':'four independent test geometries, fixed five-member models; excludes retraining and FE discretization uncertainty'}
 # The deployed choice is based only on ensemble validation, never these test outcomes.
 assert preferred is not None
 verdict={'complete':True,'models':results,'comparisons':comparisons,'preferred_from_validation_only':preferred,'coverage_target':'joint maximum nodal L2 error over all nodes and all200 time steps in one new geometry/independently drawn waveform; calibration/test descriptor distributions match','mechanics_frames':frames,'scope':'passive full-cycle dt5ms numerical ellipsoid family, all rollouts start from zero histories; no teacher forcing','limitations':['four independent primary test geometries: very wide confidence intervals','limited regime uses only one training geometry and its own train-only POD','all shapes from one affine ellipsoid family','reference discretization uncertainty reported separately','shift coverage has no exchangeability guarantee','domain gate detects declared input-support violations, not every possible surrogate failure','fallback routing counts are decisions; rejected-reference accuracy is not counted as NN prediction accuracy']}
 (P/'experiments/metrics/dynamic_final_evaluation.json').write_text(json.dumps(verdict,indent=2));np.savez_compressed(P/'experiments/metrics/dynamic_latents.npz',**{key.replace('/','_')+'__'+case:q for key,cases in latent.items() for case,q in cases.items()});print('FINAL DYNAMIC EVALUATION COMPLETE',preferred,flush=True)
finally:bridge.close()
