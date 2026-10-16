"""Matched graph/POD surrogates with verified exact cardiac residual gradients."""
from pathlib import Path
import sys,json,copy,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch,yaml
from lv_surrogate.models.reduced_graph import ReducedGraphSurrogate
from lv_surrogate.physics.fem_bridge import FEMBridge
from lv_surrogate.physics.kinematics import deformation_gradient,green_lagrange
P=Path(__file__).resolve().parents[1];torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
verification=json.loads((P/'experiments/metrics/cardiac_verification.json').read_text())
if verification['independent_vs_trusted_relative_l2_at_perturbed_state']>=1e-6 or verification['VJP_relative_error']>=1e-4:raise RuntimeError('Cardiac verification gates failed; physics training prohibited')
a=np.load(P/'data/processed/cardiac/cohort.npz');cfg=yaml.safe_load((P/'configs/physics/startup.yaml').read_text());manifest=json.loads((P/'experiments/manifests/cardiac_cohort.json').read_text());split=manifest['splits'];train=np.array(split['train']);valid=np.array(split['validation']);test=np.array(split['test']);fields=a['displacements'];pressures=a['pressures'];pmean=pressures[train].mean();pstd=pressures[train].std();pn=torch.tensor((pressures-pmean)/pstd,dtype=torch.float64)
mean=fields[train].mean(0);_,singular,V=np.linalg.svd((fields[train]-mean).reshape(len(train),-1),full_matrices=False);basis=V[:cfg['pod_rank']].reshape(-1,*mean.shape);coefs=np.einsum('bnc,rnc->br',fields[train]-mean,basis);cscale=np.maximum(coefs.std(0),1e-12);u_scale=float(np.sqrt(np.mean((fields[train]-mean)**2)));target=torch.tensor(fields,dtype=torch.float64)
X=a['coords'];node=np.c_[X,a['fiber'],a['sheet']];node=(node-node.mean(0))/np.maximum(node.std(0),1e-8)
cells=a['cells'];edges=np.concatenate([cells[:,[i,j]] for i in range(4) for j in range(i+1,4)]);edges=np.unique(np.sort(edges,axis=1),axis=0);edges=np.r_[edges,edges[:,::-1]];delta=X[edges[:,1]]-X[edges[:,0]];ef=np.c_[delta,np.linalg.norm(delta,axis=-1)];ef=(ef-ef.mean(0))/np.maximum(ef.std(0),1e-8)
kwargs=[torch.tensor(z,dtype=torch.float64) for z in [mean,basis,cscale,node,ef]]+[torch.tensor(edges,dtype=torch.long)];torch.manual_seed(cfg['seed']);initial=ReducedGraphSurrogate(*kwargs).double().state_dict();bridge=FEMBridge(P);models={};histories={};validation_results={};times={};checkpoints=P/'experiments/checkpoints/cardiac_startup_v1';checkpoints.mkdir(exist_ok=True)

def validation(net):
 with torch.no_grad():pred=net(pn[valid]);mse=float((pred-target[valid]).square().mean())
 eq=float(np.mean([float(bridge.evaluate(u,pressures[i],False)['loss']) for i,u in zip(valid,pred.numpy())]));return mse,eq
try:
 for name,lam in [('data_only',0.)]+[(f'physics_lambda{lam:g}',float(lam)) for lam in cfg['physics_lambdas']]:
  torch.manual_seed(cfg['seed']);model=ReducedGraphSurrogate(*kwargs).double();model.load_state_dict(initial);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['steps'],eta_min=.0001);best=float('inf');history=[];start=time.perf_counter()
  for step in range(cfg['steps']):
   model.train();opt.zero_grad();pred=model(pn[train]);data=(pred-target[train]).square().mean()/u_scale**2;eq=pred.new_zeros(())
   if lam:
    j=step%len(train);eq=bridge.loss(pred[j],pressures[train[j]])
   loss=data+lam*eq
   if not torch.isfinite(loss):raise FloatingPointError(name)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();scheduler.step()
   if step%cfg['validation_every']==0 or step==cfg['steps']-1:
    model.eval();mse,res=validation(model);score=mse/u_scale**2+lam*res;row={'step':step,'training_data_scaled_mse':float(data.detach()),'validation_mse_m2':mse,'validation_residual_loss':res,'selection_score':score};history.append(row)
    if score<best:best=score;beststate=copy.deepcopy(model.state_dict());bestval=(mse,res);beststep=step
    print(name,row,flush=True)
  model.load_state_dict(beststate);model.eval();models[name]=model;validation_results[name]={'component_rmse_m':bestval[0]**.5,'normalized_residual_rms':bestval[1]**.5,'best_step':beststep};times[name]=time.perf_counter()-start;histories[name]=history;torch.save(beststate,checkpoints/f'{name}.pt')
  (P/'experiments/metrics/physics_training.json').write_text(json.dumps({'validation':validation_results,'history':histories,'training_seconds':times},indent=2))
 # Freeze lambda from validation alone; test cases are first used below.
 baseline_val=validation_results['data_only']['component_rmse_m'];eligible=[n for n,r in validation_results.items() if n!='data_only' and r['component_rmse_m']<=1.1*baseline_val]
 selected=min(eligible,key=lambda n:validation_results[n]['normalized_residual_rms']) if eligible else min((n for n in models if n!='data_only'),key=lambda n:validation_results[n]['component_rmse_m'])
 physics_accepted=bool(eligible);results={};XT=torch.tensor(X,dtype=torch.float64);CT=torch.tensor(cells,dtype=torch.long)
 for name in ['data_only',selected]:
  with torch.no_grad():prediction=models[name](pn[test]).numpy()
  records=[]
  for idx,u in zip(test,prediction):
   diff=u-fields[idx];R=bridge.evaluate(u,pressures[idx],False);F=deformation_gradient(XT,torch.tensor(u),CT);Ftrue=deformation_gradient(XT,torch.tensor(fields[idx]),CT);E=green_lagrange(F);Etrue=green_lagrange(Ftrue);J=torch.linalg.det(F);records.append({'case':int(idx),'pressure_Pa':float(pressures[idx]),'component_rmse_m':float(np.sqrt(np.mean(diff**2))),'mean_nodal_l2_m':float(np.linalg.norm(diff,axis=-1).mean()),'normalized_residual_rms':float(R['loss'])**.5,'green_lagrange_mean_frobenius_error':float(torch.linalg.matrix_norm(E-Etrue).mean()),'min_detF':float(J.min()),'inverted_element_fraction':float((J<=0).double().mean())})
  results[name]={'mean_component_rmse_m':float(np.mean([r['component_rmse_m'] for r in records])),'mean_normalized_residual_rms':float(np.mean([r['normalized_residual_rms'] for r in records])),'mean_strain_frobenius_error':float(np.mean([r['green_lagrange_mean_frobenius_error'] for r in records])),'per_case':records}
  np.savez_compressed(P/f'experiments/metrics/physics_{name}_predictions.npz',ids=test,predictions=prediction,truth=fields[test])
 verdict={'config':cfg,'task':manifest['task'],'n_train':20,'n_validation':6,'n_test_load_cases':6,'n_independent_geometries':1,'parameter_count':sum(p.numel() for p in models['data_only'].parameters()),'pod_rank':cfg['pod_rank'],'pod_fit':'training20 cases only','pod_singular_values':singular.tolist(),'u_scale_m':u_scale,'pressure_normalizer':{'mean_Pa':float(pmean),'std_Pa':float(pstd)},'force_scale_N':float(a['force_scale_N']),'lambda_selection':{'selected':selected,'meets_validation_RMSE_10percent_constraint':physics_accepted,'validation':validation_results},'results':results,'training_seconds':times,'BC':'natural epi/base Robin and pressure included in full residual; no imposed Dirichlet DOFs','gradient':'custom autograd using exact trusted tangent VJP; independent assembly and finite-difference verification passed','limitations':['one seed pair, fixed geometry, six held-out load cases','startup passive elastodynamics from rest, not full cycle or Dalton mechanics','POD decoder limits outputs to training-derived subspace','no claim of geometry generalization or calibrated physics ensemble uncertainty']}
 (P/'experiments/metrics/physics_experiment.json').write_text(json.dumps(verdict,indent=2));print(json.dumps(verdict,indent=2))
finally:bridge.close()
