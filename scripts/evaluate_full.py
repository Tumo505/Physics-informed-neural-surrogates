"""Freeze checkpoints before reading calibration and untouched modern-test targets."""
from pathlib import Path
import sys,json,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch,yaml
from scipy.stats import spearmanr,binomtest
from lv_surrogate.models.multiscale import MultiscaleSurrogate
from lv_surrogate.eval.calibration import conformal_radius
P=Path(__file__).resolve().parents[1];D=P/'data/processed/full'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/gnn/full_low_lr.yaml');args=ap.parse_args()
 cfg=yaml.safe_load((P/args.config).read_text());device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');torch.set_num_threads(4);base=P/'experiments/checkpoints'/cfg['run_name'];predir=P/'experiments/predictions'/cfg['run_name'];predir.mkdir(exist_ok=True)
 for kind,seeds in [('gnn',cfg['seeds']),('point',cfg['baseline_seeds'])]:
  for seed in seeds:
   latest=torch.load(base/f'{kind}_{seed}_latest.pt',map_location='cpu',weights_only=False)
   if not latest['complete']:raise RuntimeError(f'{kind} {seed} not complete; evaluation forbidden')
 norm=np.load(D/'normalizers.npz');std=torch.tensor(norm['targets_std'],device=device);mean=torch.tensor(norm['targets_mean'],device=device);top=torch.tensor(np.load(D/'topology.npy'),device=device);parents=torch.tensor(np.load(D/'parents.npy'),device=device)
 timings={};allpred={}
 for kind,seeds in [('gnn',cfg['seeds']),('point',cfg['baseline_seeds'])]:
  for seed in seeds:
   model=MultiscaleSurrogate(width=cfg['width'],steps=cfg['steps'],graph=kind=='gnn').to(device);checkpoint=torch.load(base/f'{kind}_{seed}_best.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model']);model.eval();times=[]
   for split,key,ids in [('validation','calibration',list(range(*cfg['calibration_ids']))),('test','test',list(range(*cfg['test_ids']))),('test','pilot_recheck',list(range(64)))]:
    arrays={n:np.load(D/f'{split}_{n}.npy',mmap_mode='r') for n in ['points','coarse','globals','edges']};pred=[]
    with torch.inference_mode():
     for i in range(0,len(ids),8):
      batchids=ids[i:i+8];b={n:torch.tensor(np.asarray(a[batchids]),device=device) for n,a in arrays.items()};t=time.perf_counter();latent=model.encode(b['coarse'],b['globals'],b['edges'],top);chunks=[]
      for node in range(0,6784,1024):chunks.append(model.decode(b['points'][:,node:node+1024],b['globals'],latent,parents[node:node+1024]))
      result=torch.cat(chunks,1)*std+mean;pred.append(result.cpu().numpy());times.append((time.perf_counter()-t)/len(batchids))
    a=np.concatenate(pred);np.save(predir/f'{kind}_{seed}_{key}.npy',a);allpred[(kind,seed,key)]=a
   timings[f'{kind}_{seed}']={'batched_per_case_ms_median':float(np.median(times)*1000),'note':'batch8 and transfer; not batch-one benchmark'}
 truth={key:np.asarray(np.load(D/f'{sp}_targets.npy',mmap_mode='r')[slice(*cfg[f'{key}_ids'])]) for sp,key in [('validation','calibration'),('test','test')]};gnn=np.stack([allpred[('gnn',s,'test')] for s in cfg['seeds']]);cal=np.mean([allpred[('gnn',s,'calibration')] for s in cfg['seeds']],0);avg=gnn.mean(0);var=gnn.var(0,ddof=1).mean((1,2));err=np.linalg.norm(avg-truth['test'],axis=-1);caseerr=err.mean(-1);maxerr=err.max(-1);rgen=np.random.default_rng(2027)
 def summary(pred):
  diff=pred-truth['test'];e=np.linalg.norm(diff,axis=-1);per=e.mean(-1);boot=np.array([per[rgen.integers(len(per),size=len(per))].mean() for _ in range(2000)])
  return {'mean_nodal_l2_cm':float(per.mean()),'mean_case_component_rmse_cm':float(np.sqrt(np.mean(diff**2,axis=(1,2))).mean()),'pooled_component_rmse_cm':float(np.sqrt(np.mean(diff**2))),'mean_case_p95_cm':float(np.quantile(e,.95,axis=-1).mean()),'mean_case_max_cm':float(e.max(-1).mean()),'geometry_bootstrap95_mean_nodal_l2_cm':np.quantile(boot,[.025,.975]).tolist()}
 models={'full_gnn_ensemble':summary(avg),'full_point_baseline':summary(allpred[('point',11,'test')])}
 for s,pred in zip(cfg['seeds'],gnn):models[f'full_gnn_seed{s}']=summary(pred)
 ref=json.loads((P/'experiments/metrics/reference_varying.json').read_text());refcases=ref['per_case'][cfg['test_ids'][0]:cfg['test_ids'][1]];models['released_gnn_same503']={'mean_nodal_l2_cm':float(np.mean([r['mean_nodal_l2_cm'] for r in refcases])),'mean_case_component_rmse_cm':float(np.mean([r['component_rmse_cm'] for r in refcases]))}
 cs=np.linalg.norm(cal-truth['calibration'],axis=-1).max(-1);coverage=[]
 for alpha in [.1,.05]:
  radius=conformal_radius(cs,alpha);k=int((maxerr<=radius).sum());ci=binomtest(k,len(maxerr)).proportion_ci();coverage.append({'nominal':1-alpha,'radius_cm':radius,'covered':k,'coverage':k/len(maxerr),'ci95':[ci.low,ci.high]})
 shape=np.load(P/'data/raw/lvData/rawData/train/shape-coeffs.npy')[:,0];thresholds=np.quantile(shape,[.1,.9]);testshape=np.load(P/'data/raw/lvData/rawData/test/shape-coeffs.npy')[slice(*cfg['test_ids']),0];extreme=(testshape<thresholds[0])|(testshape>thresholds[1]);shift={}
 for label,mask in [('training_shape10_90_tail',extreme),('central_shape',~extreme)]:
  shift[label]={'n':int(mask.sum()),'mean_nodal_l2_cm':float(caseerr[mask].mean()),'coverage_at_calibrated_radii':[(maxerr[mask]<=c['radius_cm']).mean().item() for c in coverage]}
 baseline_err=np.linalg.norm(allpred[('point',11,'test')]-truth['test'],axis=-1).mean(-1);paired=caseerr-baseline_err;boots=np.array([paired[rgen.integers(len(paired),size=len(paired))].mean() for _ in range(2000)])
 pilot_avg=np.mean([allpred[('gnn',s,'pilot_recheck')] for s in cfg['seeds']],0);pilot_true=np.load(D/'test_targets.npy',mmap_mode='r')[:64];pilot_recheck={'n':64,'mean_nodal_l2_cm':float(np.linalg.norm(pilot_avg-pilot_true,axis=-1).mean()),'old_pilot_mean_nodal_l2_cm':0.47937631607055664,'role':'previously inspected cohort, secondary same-case comparison only'}
 out={'pilot_recheck_same64':pilot_recheck,'run':cfg['run_name'],'n_training':2250,'n_validation':16,'n_calibration':134,'n_test':len(avg),'n_seeds':5,'device':str(device),'models':models,'coverage':coverage,'spearman_uncertainty_error':float(spearmanr(var,caseerr).statistic),'paired_ensemble_minus_point_mean_nodal_l2_cm':float(paired.mean()),'paired_geometry_bootstrap95':np.quantile(boots,[.025,.975]).tolist(),'shape_thresholds_training10_90':thresholds.tolist(),'exploratory_shape_shift':shift,'selective_risk':[{'retained':f,'mean_nodal_l2_cm':float(caseerr[np.argsort(var)[:int(len(var)*f)]].mean())} for f in [.25,.5,.75,1]],'timings':timings,'limitations':['original validation calibration reused after historical pilot; no coverage tuning','one fixed topology; geometry variation within provided cohort','graph reduction and sampled-node training differ from published architecture/budget','shape-tail analysis exploratory, no OOD coverage guarantee','no Dalton FE residual or matched solver timing'],'per_case':[{'id':i,'mean_nodal_l2_cm':float(e),'max_nodal_l2_cm':float(mx),'variance_cm2':float(v),'point_baseline_error_cm':float(bl),'shape_tail':bool(tail)} for i,e,mx,v,bl,tail in zip(range(*cfg['test_ids']),caseerr,maxerr,var,baseline_err,extreme)]}
 (P/'experiments/metrics/full_evaluation.json').write_text(json.dumps(out,indent=2));np.savez_compressed(P/'experiments/metrics/full_plot_data.npz',error=caseerr,variance=var,max_error=maxerr,calibration_scores=cs,ensemble_case0=avg[0],truth_case0=truth['test'][0]);print(json.dumps({k:v for k,v in out.items() if k!='per_case'},indent=2))
if __name__=='__main__':main()
