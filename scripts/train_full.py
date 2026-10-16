"""Resumable full-data training. Calibration/test targets are never opened here."""
from pathlib import Path
import argparse,sys,time,json,hashlib,random,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch,yaml
from lv_surrogate.models.multiscale import MultiscaleSurrogate
P=Path(__file__).resolve().parents[1];D=P/'data/processed/full'

def sync(device):
 if device.type=='mps':torch.mps.synchronize()

def atomic_save(obj,path):
 temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/gnn/full.yaml');ap.add_argument('--seed',type=int);ap.add_argument('--kind',choices=['gnn','point']);ap.add_argument('--device');ap.add_argument('--benchmark-only',action='store_true');args=ap.parse_args()
 cfg=yaml.safe_load((P/args.config).read_text());device=torch.device(args.device or cfg['device']);torch.set_num_threads(4)
 if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; request GPU-capable host execution or explicitly use --device cpu')
 names=['points','coarse','globals','edges','targets'];train={n:np.load(D/f'train_{n}.npy',mmap_mode='r') for n in names};valid={n:np.load(D/f'validation_{n}.npy',mmap_mode='r') for n in names};norm=np.load(D/'normalizers.npz')
 topology=torch.from_numpy(np.load(D/'topology.npy')).to(device);parents=np.load(D/'parents.npy');ustd=torch.from_numpy(norm['targets_std']).to(device);umean=torch.from_numpy(norm['targets_mean']).to(device);scale=float(np.mean(norm['targets_std']**2))
 out=P/'experiments/checkpoints'/cfg['run_name'];out.mkdir(exist_ok=True);metrics=P/'experiments/metrics'/cfg['run_name'];metrics.mkdir(exist_ok=True);manifests=P/'experiments/manifests'/cfg['run_name'];manifests.mkdir(exist_ok=True)
 def batch(data,ids,node_ids=None):
  ids=np.asarray(ids);values={}
  for n,a in data.items():
   b=a[ids[:,None],node_ids] if node_ids is not None and n in ('points','targets') else a[ids]
   values[n]=torch.from_numpy(np.array(b,copy=True)).to(device)
  par=parents if node_ids is None else parents[node_ids]
  values['parents']=torch.from_numpy(np.array(par,copy=True)).to(device)
  return values
 def predict(net,b):return net(b['points'],b['coarse'],b['globals'],b['edges'],topology,b['parents'])*ustd+umean
 def validation(net):
  net.eval();sums=0.;count=0
  with torch.inference_mode():
   for i in range(*cfg['validation_ids']):
    b=batch(valid,[i]);diff=predict(net,b)-b['targets'];sums+=float(diff.square().mean().cpu());count+=1
  return sums/count
 jobs=[(kind,seed) for kind,seeds in [('gnn',cfg['seeds']),('point',cfg['baseline_seeds'])] for seed in seeds if (args.seed is None or seed==args.seed) and (args.kind is None or kind==args.kind)]
 for kind,seed in jobs:
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
  model=MultiscaleSurrogate(width=cfg['width'],steps=cfg['steps'],graph=kind=='gnn').to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=cfg['minimum_lr'])
  latest=out/f'{kind}_{seed}_latest.pt';bestpath=out/f'{kind}_{seed}_best.pt';history=[];best=float('inf');stale=0;start_epoch=0;prior_seconds=0.
  # A uniformly shuffled cyclic node schedule guarantees every original node is visited.
  node_order=rng.permutation(6784);offsets=rng.integers(6784,size=len(train['points']))
  if cfg.get('initialize_best_from') and not latest.exists():
   source=P/'experiments/checkpoints'/cfg['initialize_best_from']/f'{kind}_{seed}_best.pt'
   model.load_state_dict(torch.load(source,map_location=device,weights_only=False)['model'])
  if latest.exists():
   ck=torch.load(latest,map_location='cpu',weights_only=False)
   if ck.get('complete') and not args.benchmark_only:print(kind,seed,'already complete',flush=True);continue
   model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);history=ck['history'];best=ck['best'];stale=ck['stale'];start_epoch=ck['epoch']+1;prior_seconds=ck['train_seconds'];rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng']);node_order=ck['node_order'];offsets=ck['offsets']
   if device.type=='mps' and 'mps_rng' in ck:torch.mps.set_rng_state(ck['mps_rng'])
  if args.benchmark_only:
   ids=np.arange(cfg['batch_size']);q=node_order[(offsets[ids,None]+np.arange(cfg['sample_nodes']))%6784];b=batch(train,ids,q);model.train()
   for j in range(5):opt.zero_grad();loss=(predict(model,b)-b['targets']).square().mean()/scale;loss.backward();opt.step()
   sync(device);t=time.perf_counter()
   for j in range(20):opt.zero_grad();loss=(predict(model,b)-b['targets']).square().mean()/scale;loss.backward();opt.step()
   sync(device);result={'device':str(device),'seconds_per_batch':(time.perf_counter()-t)/20,'loss':float(loss.detach().cpu()),'parameters':sum(p.numel() for p in model.parameters()),'batch_size':cfg['batch_size'],'nodes_per_geometry':cfg['sample_nodes']};print(json.dumps(result),flush=True);(metrics/'throughput.json').write_text(json.dumps(result,indent=2));return
  manifest={'config':cfg,'kind':kind,'seed':seed,'torch':torch.__version__,'device':str(device),'determinism':'fixed RNG; MPS indexed reductions may be nondeterministic','normalizer_sha256':hashlib.sha256((D/'normalizers.npz').read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'src/lv_surrogate/models/multiscale.py']},'parameters':sum(p.numel() for p in model.parameters()),'initialization':cfg.get('initialize_best_from','random; no released checkpoint or latents'),'training_geometries':len(train['points']),'validation_geometries':len(range(*cfg['validation_ids']))}
  (manifests/f'{kind}_{seed}.json').write_text(json.dumps(manifest,indent=2))
  # Continuation must retain its starting checkpoint if every update degrades it.
  if not latest.exists():
   best=validation(model)
   atomic_save({'model':model.state_dict(),'epoch':-1,'validation_mse_cm2':best,'manifest':manifest},bestpath)
   history.append({'epoch':-1,'validation_component_mse_cm2':best,'role':'initialization_candidate'})
  sync(device);start=time.perf_counter()
  for epoch in range(start_epoch,cfg['epochs']):
   model.train();order=rng.permutation(len(train['points']));loss_sum=0.;nseen=0
   for i in range(0,len(order),cfg['batch_size']):
    ids=order[i:i+cfg['batch_size']];q=node_order[(offsets[ids,None]+epoch*cfg['sample_nodes']+np.arange(cfg['sample_nodes']))%6784];b=batch(train,ids,q);opt.zero_grad(set_to_none=True);pred=predict(model,b);loss=(pred-b['targets']).square().mean()/scale
    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss {kind} seed{seed} epoch{epoch}')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();loss_sum+=float(loss.detach().cpu())*len(ids);nseen+=len(ids)
   scheduler.step();row={'epoch':epoch,'train_scaled_mse':loss_sum/nseen,'learning_rate':opt.param_groups[0]['lr']}
   if epoch%cfg['validation_every']==0 or epoch==cfg['epochs']-1:
    val=validation(model);row['validation_component_mse_cm2']=val
    if val<best:
     best=val;stale=0;atomic_save({'model':model.state_dict(),'epoch':epoch,'validation_mse_cm2':val,'manifest':manifest},bestpath)
    else:stale+=1
   history.append(row);sync(device);elapsed=prior_seconds+time.perf_counter()-start;complete=epoch==cfg['epochs']-1 or stale>=cfg['patience_checks']
   ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'history':history,'best':best,'stale':stale,'train_seconds':elapsed,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'node_order':node_order,'offsets':offsets,'complete':complete}
   if device.type=='mps':ck['mps_rng']=torch.mps.get_rng_state()
   atomic_save(ck,latest)
   record={'kind':kind,'seed':seed,'complete':complete,'epochs_completed':epoch+1,'best_validation_component_rmse_cm':best**.5,'train_seconds':elapsed,'history':history,'parameters':manifest['parameters']};(metrics/f'{kind}_{seed}.json').write_text(json.dumps(record,indent=2))
   print(json.dumps({k:v for k,v in record.items() if k!='history'}),flush=True)
   if complete:break
if __name__=='__main__':main()
