"""Evaluate frozen ensemble on separately reserved calibration and pilot test geometries."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.stats import spearmanr,binomtest
from lv_surrogate.data.dalton import Dalton
from lv_surrogate.eval.calibration import geometry_scores,conformal_radius
P=Path(__file__).resolve().parents[1];split=json.loads((P/'data/splits/pilot.json').read_text());seeds=[11,22,33,44,55]
truth={s:np.stack([Dalton(P/'data/raw/lvData',src)[i]['real-node-displacement'] for i in split[s]]) for s,src in [('calibration','validation'),('test','test')]}
preds={s:np.stack([np.load(P/f'experiments/predictions/seed{seed}_{s}.npy') for seed in seeds]) for s in truth}
mean={s:p.mean(0) for s,p in preds.items()};err=np.linalg.norm(mean['test']-truth['test'],axis=-1);rmse=np.sqrt(np.mean((mean['test']-truth['test'])**2,axis=(1,2)));unc=preds['test'].var(0,ddof=1).mean((1,2));rng=np.random.default_rng(2027)
boot=np.array([err.mean(-1)[rng.integers(len(err),size=len(err))].mean() for _ in range(2000)])
cal=geometry_scores(mean['calibration'],truth['calibration']);test=geometry_scores(mean['test'],truth['test']);coverage=[]
for alpha in [.1,.05]:
 radius=conformal_radius(cal,alpha);k=int((test<=radius).sum());ci=binomtest(k,len(test)).proportion_ci()
 coverage.append({'nominal':1-alpha,'radius_cm':radius,'coverage':k/len(test),'successes':k,'n':len(test),'exact_binomial_95CI':[ci.low,ci.high]})
# Training-fitted spatially constant mean target baseline; no test fitting.
stats=np.load(P/'experiments/checkpoints/pilot_normalizers.npz');null=np.broadcast_to(stats['umean'],truth['test'].shape);nullerr=np.linalg.norm(null-truth['test'],axis=-1).mean(-1)
out={'label':'small 64-training-case data-only pilot; no physics or final study claim','n_train':len(split['train']),'n_validation':len(split['validation']),'n_calibration':len(cal),'n_test':len(test),'n_seeds':5,'mean_nodal_l2_cm':float(err.mean()),'geometry_bootstrap_95CI_mean_nodal_l2_cm':np.quantile(boot,[.025,.975]).tolist(),'mean_case_component_rmse_cm':float(rmse.mean()),'pooled_component_rmse_cm':float(np.sqrt(np.mean((mean['test']-truth['test'])**2))),'mean_case_p95_nodal_l2_cm':float(np.quantile(err,.95,axis=-1).mean()),'mean_case_max_nodal_l2_cm':float(test.mean()),'training_mean_baseline_mean_nodal_l2_cm':float(nullerr.mean()),'coverage':coverage,'spearman_uncertainty_mean_error':float(spearmanr(unc,err.mean(-1)).statistic),'selective_risk':[{'retained_fraction':f,'mean_nodal_l2_cm':float(err.mean(-1)[np.argsort(unc)[:max(1,int(f*len(err)))]].mean())} for f in [.25,.5,.75,1.]],'seed_accuracy':[{'seed':seed,'mean_nodal_l2_cm':float(np.linalg.norm(p-truth['test'],axis=-1).mean())} for seed,p in zip(seeds,preds['test'])],'per_case':[{'id':f'test:{i}','mean_nodal_l2_cm':float(e.mean()),'component_rmse_cm':float(r),'max_nodal_l2_cm':float(e.max()),'ensemble_mean_variance_cm2':float(v)} for i,e,r,v in zip(split['test'],err,rmse,unc)]}
(P/'experiments/metrics/ensemble_pilot.json').write_text(json.dumps(out,indent=2));np.savez_compressed(P/'experiments/metrics/ensemble_plot_data.npz',error=err.mean(-1),uncertainty=unc,calibration_scores=cal,test_scores=test)
print(json.dumps({k:v for k,v in out.items() if k!='per_case'},indent=2))
