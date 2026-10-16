"""Matched five-seed data/physics graph-POD study; training/selection data ONLY."""
from pathlib import Path
import sys,json,time,copy,hashlib,argparse
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.geometry_pod import GeometryPOD
from lv_surrogate.physics.fem_bridge import FEMBridge
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True);device=torch.device('mps')
if not torch.backends.mps.is_available():raise RuntimeError('MPS required for GPU protocol')
backend=json.loads((P/'experiments/metrics/physics_backend.json').read_text())
if not backend['passed']:raise RuntimeError('GPU residual precision gate failed')
ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/physics/geometry_training_gpu.json');args=ap.parse_args()
cfg=json.loads((P/args.config).read_text())
verify=json.loads((P/'experiments/metrics/extended_mechanics_verification.json').read_text())
if not verify['passed']:raise RuntimeError('Extended mechanics gates failed')
D=P/'data/processed/cardiac_geometry_ml';raw=P/'data/processed/cardiac_geometry_v1';norm=np.load(D/'normalizers.npz');topology=torch.from_numpy(np.load(D/'topology.npy')).to(device)
data={s:dict(np.load(D/f'{s}.npz')) for s in ['train','validation']}
tensors={s:{k:torch.as_tensor(a[k],dtype=torch.float32,device=device) for k in ['coarse','globals','edges','targets']} for s,a in data.items()}
kwargs=[norm[k] for k in ['field_mean','basis','coefficient_scale']];u_scale=float(norm['u_scale']);out=P/'experiments/checkpoints'/cfg['name'];out.mkdir(exist_ok=True);metrics=P/'experiments/metrics'/cfg['name'];metrics.mkdir(exist_ok=True)
bridge=FEMBridge(P,threads=2);operators={}
def force_loss(u,g,p,region='all'):
 if g not in operators:
  a=np.load(raw/f'geometry_{g:03d}.npz');own=CardiacFirstStep(a,dtype=torch.float32,analytic_piola=True).to(device);mask=np.zeros(len(a['coords']),dtype=bool);mask[np.unique(a['facets'])]=True
  operators[g]=(own,float(a['force_scale_N']),torch.tensor(mask,device=device))
 own,scale,boundary=operators[g];R=own(u,p,create_graph=torch.is_grad_enabled())/scale
 if region=='boundary':R=R*boundary[:,None]
 elif region=='interior':R=R*(~boundary)[:,None]
 return R.square().mean()
def model_new(seed):
 torch.manual_seed(seed);net=GeometryPOD(*kwargs).float().to(device)
 if cfg.get('initialize_from_run'):
  source=P/'experiments/checkpoints'/cfg['initialize_from_run']/f'{cfg.get("initialize_from_label","data")}_seed{seed}_best.pt';net.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['model'])
 return net
def predict(net,split):
 a=tensors[split];return net(a['coarse'],a['globals'],a['edges'],topology)
def evaluate(net,region='all',trusted=False):
 with torch.no_grad():
  pred=predict(net,'validation');physical=pred.cpu().numpy().astype('float64');mse=float(np.mean((physical-data['validation']['targets'])**2));full=[];chosen=[]
  for i in range(len(pred)):
   g=int(data['validation']['geometry_ids'][i]);p=float(data['validation']['pressures'][i])
   if trusted:
    path=raw/f'geometry_{g:03d}.npz';r=float(bridge.evaluate(physical[i],p,False,geometry_path=path)['loss']);full.append(r)
    chosen.append(r if region=='all' else float(bridge.evaluate(physical[i],p,False,geometry_path=path,region=region)['loss']))
   else:
    r=float(force_loss(pred[i],g,p));full.append(r);chosen.append(r if region=='all' else float(force_loss(pred[i],g,p,region)))
 return {'component_mse_m2':mse,'component_rmse_m':mse**.5,'full_residual_loss':float(np.mean(full)),'region_residual_loss':float(np.mean(chosen))}
def train(seed,lam,region='all',label=None):
 name=label or ('data' if not lam else f'physics_lambda{lam:g}')
 stem=f'{name}_seed{seed}';record_path=metrics/f'{stem}.json';bestpath=out/f'{stem}_best.pt';latest=out/f'{stem}_latest.pt'
 record=json.loads(record_path.read_text()) if record_path.exists() else {}
 if record.get('complete') and record.get('final_validation_backend')=='trusted float64 FE':return record
 net=model_new(seed);opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['steps'],eta_min=cfg['minimum_lr']);history=[];best=float('inf');start_step=0;prior=0.;best_val=None
 if latest.exists():
  ck=torch.load(latest,weights_only=False,map_location='cpu');net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);history=ck['history'];best=ck['best'];best_val=ck['best_validation'];start_step=ck['step']+1;prior=ck['seconds']
 start=time.perf_counter();rng=np.random.default_rng(seed)
 # A seed/step-derived sampler permits exact resumption without RNG snapshots.
 for step in range(start_step,cfg['steps']):
  net.train();opt.zero_grad();pred=predict(net,'train');loss_data=(pred-tensors['train']['targets']).square().mean()/u_scale**2;eq=pred.new_zeros(())
  if lam:
   if cfg.get('physics_batch_size'):
    ids=np.random.default_rng(np.random.SeedSequence([seed,step])).choice(len(pred),cfg['physics_batch_size'],replace=False)
    eq=torch.stack([force_loss(pred[int(i)],int(data['train']['geometry_ids'][i]),float(data['train']['pressures'][i]),region) for i in ids]).mean()
   else:
    # Original protocol: each400-update pass visits20 geometries for20 updates.
    g=(step//20)%20;j=step%4;i=g*4+j
    eq=force_loss(pred[i],g,float(data['train']['pressures'][i]),region)
  loss=loss_data+lam*eq
  if not torch.isfinite(loss):raise FloatingPointError(stem)
  loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();sched.step()
  if step%cfg['validation_every']==0 or step==cfg['steps']-1:
   net.eval();val=evaluate(net,region);score=val['component_mse_m2']/u_scale**2+lam*val['region_residual_loss'];row={'step':step,'train_scaled_data_mse':float(loss_data.detach()),'validation':val,'selection_score':score};history.append(row)
   if score<best:
    best=score;best_val=val;torch.save({'model':net.state_dict(),'step':step,'validation':val},bestpath)
   elapsed=prior+time.perf_counter()-start;record={'name':name,'seed':seed,'lambda':lam,'region':region,'complete':False,'steps_completed':step+1,'best_validation':best_val,'seconds':elapsed,'history':history,'parameters':sum(p.numel() for p in net.parameters())}
   record_path.write_text(json.dumps(record,indent=2));tmp=latest.with_suffix('.tmp');torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'step':step,'history':history,'best':best,'best_validation':best_val,'seconds':elapsed},tmp);tmp.replace(latest)
   print(stem,step,best_val,flush=True)
 net.load_state_dict(torch.load(bestpath,map_location='cpu',weights_only=False)['model']);net.eval();record['selection_validation_float32']=record['best_validation'];record['best_validation']=evaluate(net,region,trusted=True);record['final_validation_backend']='trusted float64 FE';record['complete']=True;record['seconds']=prior+time.perf_counter()-start;record_path.write_text(json.dumps(record,indent=2))
 return record
try:
 screen={};screen['data']=train(11,0.)
 for lam in cfg['physics_lambdas']:screen[f'physics_lambda{lam:g}']=train(11,float(lam))
 baseline=screen['data']['best_validation']['component_rmse_m'];eligible=[n for n in screen if n!='data' and screen[n]['best_validation']['component_rmse_m']<=1.1*baseline]
 selected=min(eligible,key=lambda n:screen[n]['best_validation']['full_residual_loss']) if eligible else min((n for n in screen if n!='data'),key=lambda n:screen[n]['best_validation']['component_rmse_m'])
 lam=screen[selected]['lambda'];selection={'selected_label':selected,'lambda':lam,'meets_10percent_constraint':bool(eligible),'screening_seed':11,'validation_only':True,'screen':{n:r['best_validation'] for n,r in screen.items()},'test_and_calibration_opened':False}
 (metrics/'selection.json').write_text(json.dumps(selection,indent=2));print('LAMBDA FROZEN',selection,flush=True)
 for seed in cfg['seeds']:
  train(seed,0.);train(seed,lam)
 for region in cfg['ablation_regions']:train(11,lam,region,label=f'{region}_lambda{lam:g}')
 manifest={'config':cfg,'complete':True,'lambda_selection':selection,'data_access':['train','validation'],'source_sha256':{str(q.relative_to(P)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),P/'src/lv_surrogate/models/geometry_pod.py',P/'scripts/fem_service.py',P/'src/lv_surrogate/physics/fem_bridge.py',P/'src/lv_surrogate/physics/reference_fem.py',P/'src/lv_surrogate/physics/cardiac_residual.py',P/'experiments/metrics/physics_backend.json']},'normalizer_sha256':hashlib.sha256((D/'normalizers.npz').read_bytes()).hexdigest(),'scope':'numerical benchmark ellipsoid sensitivity distribution; not an anatomical population; startup only'}
 target=P/f'experiments/manifests/{cfg["name"]}.json'
 if target.exists():
  old=json.loads(target.read_text())
  if old.get('complete'):
   manifest['reproduction_source_sha256']=manifest['source_sha256'];manifest['source_sha256']=old['source_sha256']
   if 'executed_source_revision' in old:manifest['executed_source_revision']=old['executed_source_revision']
 target.write_text(json.dumps(manifest,indent=2));print('GEOMETRY MODEL STUDY COMPLETE',flush=True)
finally:bridge.close()
