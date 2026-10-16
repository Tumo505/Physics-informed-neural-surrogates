"""Repeated Dalton benchmark evaluation after validation-only architecture freeze."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch,yaml
from scipy.stats import binomtest,spearmanr
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.multiscale import MultiscaleSurrogate
from lv_surrogate.eval.calibration import conformal_radius
cfg=yaml.safe_load((P/'configs/gnn/selected_v4.yaml').read_text());D=P/cfg['data_root'];device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');torch.set_num_threads(4);out=P/'experiments/predictions'/cfg['run_name'];out.mkdir(exist_ok=True);norm=np.load(D/'normalizers.npz');top=torch.tensor(np.load(D/'topology.npy'),device=device);parents=torch.tensor(np.load(D/'parents.npy'),device=device);std=torch.tensor(norm['targets_std'],device=device);mean=torch.tensor(norm['targets_mean'],device=device);models=[];hashes={}
for seed in cfg['seeds']:
 root=P/'experiments/checkpoints'/cfg['run_name'];latest=torch.load(root/f'gnn_{seed}_latest.pt',weights_only=False,map_location='cpu')
 if not latest['complete']:raise RuntimeError('Ensemble incomplete')
 path=root/f'gnn_{seed}_best.pt';ck=torch.load(path,weights_only=False,map_location=device);model=MultiscaleSurrogate(width=cfg['width'],steps=cfg['steps']).to(device);model.load_state_dict(ck['model']);model.eval();models.append(model);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
(P/'experiments/manifests/graph_v4_evaluation_freeze.json').write_text(json.dumps({'config':cfg,'sha256':hashes,'status':'repeated benchmark on already inspected test503; architecture selection used validation only'},indent=2))
predictions={}
for split,key,ids in [('validation','calibration',np.arange(*cfg['calibration_ids'])),('test','test',np.arange(*cfg['test_ids']))]:
 arrays={k:np.load(D/f'{split}_{k}.npy',mmap_mode='r') for k in ['points','coarse','globals','edges']};ensemble=[]
 for seed,model in zip(cfg['seeds'],models):
  pred=[]
  with torch.inference_mode():
   for start in range(0,len(ids),8):
    ii=ids[start:start+8];a={k:torch.tensor(np.array(v[ii],copy=True),device=device) for k,v in arrays.items()};latent=model.encode(a['coarse'],a['globals'],a['edges'],top);chunks=[model.decode(a['points'][:,j:j+1024],a['globals'],latent,parents[j:j+1024]) for j in range(0,6784,1024)];pred.append((torch.cat(chunks,1)*std+mean).cpu().numpy())
  pp=np.concatenate(pred);np.save(out/f'gnn_{seed}_{key}.npy',pp);ensemble.append(pp)
 predictions[key]=np.stack(ensemble)
truth={k:np.asarray(np.load(D/f'{split}_targets.npy',mmap_mode='r')[slice(*cfg[k+'_ids'])]) for split,k in [('validation','calibration'),('test','test')]};avg=predictions['test'].mean(0);cal=predictions['calibration'].mean(0);err=np.linalg.norm(avg-truth['test'],axis=-1);case=err.mean(-1);variance=predictions['test'].var(0,ddof=1).mean((1,2));scores=np.linalg.norm(cal-truth['calibration'],axis=-1).max(-1);coverage=[];rng=np.random.default_rng(2027)
for alpha in [.1,.05]:
 radius=conformal_radius(scores,alpha);k=int((err.max(-1)<=radius).sum());ci=binomtest(k,len(err)).proportion_ci();coverage.append({'nominal':1-alpha,'radius_cm':radius,'coverage':k/len(err),'covered':k,'ci95':[ci.low,ci.high]})
prior=np.mean([np.load(P/f'experiments/predictions/full_multiscale_v2/gnn_{s}_test.npy') for s in cfg['seeds']],0);prior_case=np.linalg.norm(prior-truth['test'],axis=-1).mean(-1);paired=case-prior_case;ci=np.quantile([paired[rng.integers(len(paired),size=len(paired))].mean() for _ in range(2000)],[.025,.975]);result={'run':cfg['run_name'],'n_train':2250,'n_selection_validation':16,'n_calibration':134,'n_test':len(case),'n_seeds':5,'mean_nodal_l2_cm':float(case.mean()),'mean_case_component_rmse_cm':float(np.sqrt(np.mean((avg-truth['test'])**2,axis=(1,2))).mean()),'previous_ensemble_mean_nodal_l2_cm':float(prior_case.mean()),'paired_mean_nodal_l2_change_cm':float(paired.mean()),'paired_geometry_bootstrap95_change_cm':ci.tolist(),'coverage':coverage,'spearman_uncertainty_error':float(spearmanr(variance,case).statistic),'individual_mean_nodal_l2_cm':{str(seed):float(np.linalg.norm(pred-truth['test'],axis=-1).mean()) for seed,pred in zip(cfg['seeds'],predictions['test'])},'limitations':['previously evaluated test503: repeated benchmark, not a fresh confirmatory test','historical calibration134 reused; no coverage-driven selection','architecture screening used one seed and16 selection geometries','no matched Dalton FEM or valid FE residual on graph-only data'],'per_geometry':[{'id':int(g),'mean_nodal_l2_cm':float(e),'previous_mean_nodal_l2_cm':float(p),'max_nodal_l2_cm':float(mx),'variance_cm2':float(v)} for g,e,p,mx,v in zip(range(*cfg['test_ids']),case,prior_case,err.max(-1),variance)]}
(P/'experiments/metrics/graph_v4_evaluation.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='per_geometry'},flush=True)
