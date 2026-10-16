"""Frozen-model geometry-level accuracy, mechanics, uncertainty and shift evaluation."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,torch
from scipy.stats import binomtest,spearmanr
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.geometry_pod import GeometryPOD
from lv_surrogate.physics.fem_bridge import FEMBridge
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.physics.kinematics import deformation_gradient,green_lagrange
from lv_surrogate.eval.calibration import conformal_radius
ap=argparse.ArgumentParser();ap.add_argument('--run',default='cardiac_geometry_models_v2');ap.add_argument('--data-root',default='data/processed/cardiac_geometry_ml');ap.add_argument('--raw-root',default='data/processed/cardiac_geometry_v1');args=ap.parse_args()
torch.set_num_threads(2);name=args.run;manifest=json.loads((P/f'experiments/manifests/{name}.json').read_text())
if not manifest['complete']:raise RuntimeError('Study incomplete; evaluation forbidden')
cfg=manifest['config'];lam=manifest['lambda_selection']['lambda'];labels=['data',f'physics_lambda{lam:g}'];D=P/args.data_root;raw=P/args.raw_root;out=P/'experiments/checkpoints'/name;norm=np.load(D/'normalizers.npz');topology=torch.from_numpy(np.load(D/'topology.npy'))
models={};hashes={}
for label in labels:
 models[label]=[]
 for seed in cfg['seeds']:
  rec=json.loads((P/f'experiments/metrics/{name}/{label}_seed{seed}.json').read_text())
  if not rec['complete']:raise RuntimeError('Incomplete model')
  path=out/f'{label}_seed{seed}_best.pt';ck=torch.load(path,weights_only=False,map_location='cpu');net=GeometryPOD(*[norm[k] for k in ['field_mean','basis','coefficient_scale']]).double();net.load_state_dict(ck['model']);net.eval();models[label].append(net);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
freeze={'checkpoint_sha256':hashes,'selection':manifest['lambda_selection'],'data_root':str(D),'raw_root':str(raw),'normalizer_sha256':hashlib.sha256((D/'normalizers.npz').read_bytes()).hexdigest(),'declaration':'All model choices frozen before opening calibration/test/shift target arrays; this is a new cohort with no historical test reuse.'}
(P/f'experiments/manifests/{name}_evaluation_freeze.json').write_text(json.dumps(freeze,indent=2))
sets={s:dict(np.load(D/f'{s}.npz')) for s in ['calibration','test','shift']};results={};predictions={};bridge=FEMBridge(P,threads=2);rng=np.random.default_rng(2027)
def predict(net,a):
 with torch.inference_mode():return net(*[torch.tensor(a[k],dtype=torch.float64) for k in ['coarse','globals','edges']],topology).numpy()
def bootstrap(values):
 a=np.asarray(values);return np.quantile([a[rng.integers(len(a),size=len(a))].mean() for _ in range(2000)],[.025,.975]).tolist()
try:
 for label,nets in models.items():
  prediction={s:np.stack([predict(net,a) for net in nets]) for s,a in sets.items()};predictions[label]=prediction
  cal=prediction['calibration'].mean(0);a=sets['calibration'];cal_scores=np.linalg.norm(cal-a['targets'],axis=-1).reshape(20,4,-1).max((1,2));radii={str(1-alpha):conformal_radius(cal_scores,alpha) for alpha in [.1,.05]}
  result={'radii_m':radii,'calibration_geometry_count':20,'coverage_target':'all nodes and all four prescribed startup pressure loads jointly within a new parameter-drawn geometry','splits':{}}
  for split in ['test','shift']:
   a=sets[split];pred=prediction[split].mean(0);diff=pred-a['targets'];case_errors=np.linalg.norm(diff,axis=-1);variance=prediction[split].var(0,ddof=1).mean((1,2));geometry_records=[];seed_records=[]
   # Individual matched-seed residuals are evaluated, not only the ensemble.
   for g in np.unique(a['geometry_ids']):
    ids=np.where(a['geometry_ids']==g)[0];path=raw/f'geometry_{g:03d}.npz';geom=np.load(path);own=CardiacFirstStep(geom);X=torch.tensor(geom['coords']);C=torch.tensor(geom['cells'],dtype=torch.long)
    rec={'geometry':int(g),'mean_nodal_l2_m':float(case_errors[ids].mean()),'component_rmse_m':float(np.sqrt(np.mean(diff[ids]**2))),'max_nodal_l2_all_loads_m':float(case_errors[ids].max()),'variance_m2':float(variance[ids].mean())}
    residuals=[];strain=[];energies=[];minimum=[];inverted=[]
    for i in ids:
     u=torch.tensor(pred[i]);true=torch.tensor(a['targets'][i]);p=float(a['pressures'][i]);R=bridge.evaluate(pred[i],p,False,geometry_path=path);residuals.append(float(R['loss'])**.5)
     F=deformation_gradient(X,u,C);Ft=deformation_gradient(X,true,C);J=torch.linalg.det(F);minimum.append(float(J.min()));inverted.append(float((J<=0).double().mean()));strain.append(float(torch.linalg.matrix_norm(green_lagrange(F)-green_lagrange(Ft)).mean()))
     if float(J.min())>0:
      E=float((own.energy_density(F[:,None].expand(-1,len(own.weights),-1,-1))*own.weights).sum(-1).dot(own.vol));Et=float((own.energy_density(Ft[:,None].expand(-1,len(own.weights),-1,-1))*own.weights).sum(-1).dot(own.vol));energies.append(abs(E-Et)/max(abs(Et),1e-12))
    rec.update(normalized_residual_rms=float(np.mean(residuals)),mean_strain_frobenius_error=float(np.mean(strain)),mean_relative_elastic_energy_error=float(np.mean(energies)) if energies else None,minimum_detF=min(minimum),inverted_element_fraction=float(np.mean(inverted)))
    geometry_records.append(rec)
    if split=='test':
     for k,seed in enumerate(cfg['seeds']):
      rr=[float(bridge.evaluate(prediction[split][k,i],float(a['pressures'][i]),False,geometry_path=path)['loss'])**.5 for i in ids];dd=prediction[split][k,ids]-a['targets'][ids];seed_records.append({'seed':seed,'geometry':int(g),'component_rmse_m':float(np.sqrt(np.mean(dd**2))),'normalized_residual_rms':float(np.mean(rr))})
   e=np.array([r['mean_nodal_l2_m'] for r in geometry_records]);mx=np.array([r['max_nodal_l2_all_loads_m'] for r in geometry_records]);v=np.array([r['variance_m2'] for r in geometry_records]);coverage=[]
   for nominal,radius in radii.items():
    covered=int((mx<=radius).sum());c={'nominal':float(nominal),'radius_m':radius,'covered':covered,'n_geometries':len(mx),'coverage':covered/len(mx)}
    if split=='test':ci=binomtest(covered,len(mx)).proportion_ci();c['binomial95']=[ci.low,ci.high]
    coverage.append(c)
   result['splits'][split]={'n_geometries':len(geometry_records),'n_load_cases':len(a['targets']),'mean_nodal_l2_m':float(e.mean()),'geometry_bootstrap95_mean_nodal_l2_m':bootstrap(e) if split=='test' else None,'mean_component_rmse_m':float(np.mean([r['component_rmse_m'] for r in geometry_records])),'mean_normalized_residual_rms':float(np.mean([r['normalized_residual_rms'] for r in geometry_records])),'coverage':coverage,'spearman_uncertainty_error':float(spearmanr(v,e).statistic),'selective_risk':[{'retained':f,'mean_nodal_l2_m':float(e[np.argsort(v)[:max(1,int(f*len(e)))]].mean())} for f in [.25,.5,.75,1.]],'per_geometry':geometry_records,'per_seed_geometry':seed_records}
  results[label]=result
  np.savez_compressed(P/f'experiments/metrics/{name}_{label}_predictions.npz',test=prediction['test'],shift=prediction['shift'],calibration=prediction['calibration'])
 ablations={}
 for region in cfg['ablation_regions']:
  stem=f'{region}_lambda{lam:g}_seed11';ck=torch.load(out/f'{stem}_best.pt',map_location='cpu',weights_only=False);net=GeometryPOD(*[norm[k] for k in ['field_mean','basis','coefficient_scale']]).double();net.load_state_dict(ck['model']);net.eval();a=sets['test'];pred=predict(net,a);records=[]
  for g in np.unique(a['geometry_ids']):
   ids=np.where(a['geometry_ids']==g)[0];residual=[float(bridge.evaluate(pred[i],float(a['pressures'][i]),False,geometry_path=raw/f'geometry_{g:03d}.npz')['loss'])**.5 for i in ids];records.append({'geometry':int(g),'component_rmse_m':float(np.sqrt(np.mean((pred[ids]-a['targets'][ids])**2))),'normalized_full_residual_rms':float(np.mean(residual))})
  ablations[region]={'seed':11,'n_geometries':len(records),'per_geometry':records,'mean_component_rmse_m':float(np.mean([r['component_rmse_m'] for r in records])),'mean_normalized_full_residual_rms':float(np.mean([r['normalized_full_residual_rms'] for r in records]))}
 base=results['data']['splits']['test'];phys=results[labels[1]]['splits']['test'];paired=np.array([p['normalized_residual_rms']-b['normalized_residual_rms'] for b,p in zip(base['per_geometry'],phys['per_geometry'])]);seed_paired=[]
 for seed in cfg['seeds']:
  b=[r for r in base['per_seed_geometry'] if r['seed']==seed];p=[r for r in phys['per_seed_geometry'] if r['seed']==seed];seed_paired.append({'seed':seed,'data_residual_rms':float(np.mean([r['normalized_residual_rms'] for r in b])),'physics_residual_rms':float(np.mean([r['normalized_residual_rms'] for r in p])),'displacement_RMSE_ratio':float(np.mean([r['component_rmse_m'] for r in p])/np.mean([r['component_rmse_m'] for r in b]))})
 verdict={'models':results,'natural_boundary_interior_ablations_single_seed':ablations,'inference_precision':'CPU float64 arithmetic with frozen float32-trained weights/buffers cast to float64; native mechanics always float64','matched_seed_comparisons':seed_paired,'paired_ensemble_residual_difference':float(paired.mean()),'paired_geometry_bootstrap95_residual_difference':bootstrap(paired),'ensemble_displacement_RMSE_ratio':phys['mean_component_rmse_m']/base['mean_component_rmse_m'],'H1_descriptive_supported':bool(paired.mean()<0 and phys['mean_component_rmse_m']<=1.1*base['mean_component_rmse_m']),'split_unit':'independent parameter-drawn numerical geometry; all loads clustered','bootstrap_scope':'conditional on the five trained members, excluding retraining uncertainty; seed comparisons reported separately','limitations':['same benchmark ellipsoid family, not independent anatomical sources','four startup loads from rest per geometry, not trajectories','20 calibration geometries and16 test geometries: wide coverage intervals','shift is outside the training parameter support; no exchangeability guarantee','P1 FE discretization uncertainty is additional to surrogate error','physics conclusions specific to this numerical parameter distribution']}
 (P/f'experiments/metrics/{name}_evaluation.json').write_text(json.dumps(verdict,indent=2));print('GEOMETRY EVALUATION COMPLETE',verdict['H1_descriptive_supported'],flush=True)
finally:bridge.close()
