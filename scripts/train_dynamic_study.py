"""Five-seed full free-rollout training and validation-only physics screening."""
from pathlib import Path
import json,sys,time,hashlib,argparse,copy
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.dynamic_graph import DynamicGraph,newmark_operators
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.physics.fem_bridge import FEMBridge
ap=argparse.ArgumentParser();ap.add_argument('--regime',choices=['full','limited'],default='full');args=ap.parse_args();cfg=json.loads((P/'configs/physics/dynamic_training.json').read_text());name=cfg['name']+'_'+args.regime;D=P/f'data/processed/cardiac_dynamic_ml_{args.regime}';stats=dict(np.load(D/'normalizers.npz'));data={s:dict(np.load(D/f'{s}.npz')) for s in ['train','validation']};device=torch.device(cfg['device']);force_device=torch.device(cfg.get('physics_device',cfg['device']));torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
if not json.loads((P/'experiments/metrics/dynamic_backend.json').read_text())['passed']:raise RuntimeError('Dynamic mechanics gate failed')
top=torch.tensor(np.load(D/'topology.npy'),device=device);arrays={s:{k:torch.tensor(a[k],dtype=torch.float32,device=device) for k in ['coarse','globals','edges','pressure','q']} for s,a in data.items()};n=data['train']['pressure'].shape[1];dt=.005;vmap,amap=[torch.tensor(x,dtype=torch.float32,device=device) for x in newmark_operators(n,dt)];operators={};out=P/'experiments/checkpoints'/name;out.mkdir(exist_ok=True);metrics=P/'experiments/metrics'/name;metrics.mkdir(exist_ok=True);bridge=None

def predict(net,split):return net(*[arrays[split][k] for k in ['coarse','globals','edges']],top,arrays[split]['pressure'])
def histories(q):
 zero=torch.zeros_like(q[:,:1]);v=vmap@q;a=amap@q
 return torch.cat((zero,q[:,:-1]),1),torch.cat((zero,v[:,:-1]),1),torch.cat((zero,a[:,:-1]),1)
def operator(case):
 if case not in operators:
  geo=np.load(P/f'data/processed/cardiac_dynamic_v1/{case}/geometry.npz');operators[case]=(CardiacFirstStep(geo,dtype=torch.float32,dt=dt,analytic_piola=True).to(force_device),float(geo['force_scale_N']))
 return operators[case]
def force(net,q,split,items):
 old=histories(q);loss=[]
 for b,t in items:
  own,scale=operator(str(data[split]['case_ids'][b]));u=net.decode(q[b,t]).reshape(-1,3).to(force_device);h={k:net.decode(x[b,t]).reshape(-1,3).to(force_device) for k,x in zip(['u_old','v_old','a_old'],old)}
  # Endpoint failure is not a valid stress; report it, rather than fake a residual.
  R=own(u,float(data[split]['pressure'][b,t])*16070.954,create_graph=torch.is_grad_enabled(),history=h);loss.append((R/scale).square().mean())
 return torch.stack(loss).mean().to(q.device)
def loss_data(q,split):
 target=arrays[split]['q'];loss=(q-target).square().mean()
 for order,weight,key in [(1,.1,'dq_scale'),(2,.01,'ddq_scale')]:loss=loss+weight*((torch.diff(q,n=order,dim=1)-torch.diff(target,n=order,dim=1))/q.new_tensor(stats[key])).square().mean()
 return loss
valitems=[(b,t) for b in range(len(data['validation']['q'])) for t in cfg['validation_residual_frames']]
def validate(net,lam=0.,mechanics=False):
 net.eval()
 with torch.no_grad():
  q=predict(net,'validation');d=float(loss_data(q,'validation'));coeff=(q-arrays['validation']['q'])*net.coefficient_scale;rmse=float((coeff.square().sum(-1).mean()/net.basis.shape[1]).sqrt());eq=float(force(net,q,'validation',valitems)) if mechanics else None
 return {'rollout_data_objective':d,'projected_component_rmse_m':rmse,'sampled_full_residual_loss':eq,'selection_score':d+lam*(eq or 0.)}
def model(seed):torch.manual_seed(seed);return DynamicGraph(stats).to(device)
def train(seed,lam,label,initial=None,warm=False):
 path=out/f'{label}_seed{seed}.pt';recpath=metrics/f'{label}_seed{seed}.json'
 if recpath.exists():
  rec=json.loads(recpath.read_text())
  if rec.get('complete'):return rec
 net=model(seed)
 if initial:net.load_state_dict(torch.load(initial,map_location=device,weights_only=False)['model'])
 steps=cfg['warm_updates'] if warm else cfg['refinement_updates'];lr=cfg['warm_lr'] if warm else cfg['refinement_lr'];opt=torch.optim.Adam(net.parameters(),lr=lr);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=cfg['minimum_lr']);start=time.perf_counter();history=[];best=float('inf');bestval=None;latest=out/f'{label}_seed{seed}_latest.pt';start_step=0;prior=0.
 if latest.exists():
  ck=torch.load(latest,map_location=device,weights_only=False);net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);start_step=ck['step']+1;history=ck['history'];best=ck['best'];bestval=ck['best_validation'];prior=ck['seconds']
 if start_step==0:
  v=validate(net,lam,mechanics=not warm);best=v['selection_score'];bestval=v;torch.save({'model':net.state_dict(),'step':-1},path)
 for step in range(start_step,steps):
  net.train();opt.zero_grad();q=predict(net,'train');d=loss_data(q,'train');eq=q.new_zeros(())
  if lam:
   rng=np.random.default_rng(np.random.SeedSequence([seed,step]));items=[(int(rng.integers(len(q))),int(rng.integers(n))) for _ in range(cfg['physics_frames_per_update'])];eq=force(net,q,'train',items)
  loss=d+lam*eq
  if not torch.isfinite(loss):raise FloatingPointError(f'{name}/{label} step{step}')
  loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();sched.step()
  if (step+1)%cfg['validation_every']==0 or step==steps-1:
   val=validate(net,lam,mechanics=not warm);history.append({'step':step,'validation':val})
   if val['selection_score']<best:best=val['selection_score'];bestval=val;torch.save({'model':net.state_dict(),'step':step},path)
   elapsed=prior+time.perf_counter()-start;torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'step':step,'history':history,'best':best,'best_validation':bestval,'seconds':elapsed},latest);print(name,label,seed,step,val,flush=True)
 rec={'complete':True,'seed':seed,'lambda':lam,'best_validation':bestval,'history':history,'updates':steps,'seconds':prior+time.perf_counter()-start,'initialization':str(initial) if initial else 'seeded random; zero readout','checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};recpath.write_text(json.dumps(rec,indent=2));return rec
try:
 screen_seed=11;warm=train(11,0.,'warm',warm=True);initial=out/'warm_seed11.pt';balancepath=metrics/'gradient_balance.json'
 if balancepath.exists():balance=json.loads(balancepath.read_text())
 else:
  net=model(11);net.load_state_dict(torch.load(initial,map_location=device,weights_only=False)['model']);q=predict(net,'train');d=loss_data(q,'train');gd=torch.autograd.grad(d,list(net.parameters()),retain_graph=True);ratios=[]
  for t in [29,69,129,179]:
   eq=force(net,q,'train',[(0,t)]);ge=torch.autograd.grad(eq,list(net.parameters()),retain_graph=True);ratios.append(float(sum(g.square().sum() for g in gd).sqrt()/sum(g.square().sum() for g in ge).sqrt().clamp_min(1e-20)))
  center=float(10**np.round(np.log10(np.median(ratios))));balance={'training_only':True,'ratios':ratios,'lambda_grid':[center/10,center,center*10]};balancepath.write_text(json.dumps(balance,indent=2))
 records={'data':train(11,0.,'data',initial=initial)}
 for lam in balance['lambda_grid']:records[f'physics_{lam:g}']=train(11,lam,f'physics_{lam:g}',initial=initial)
 # Candidate validation uses CPU float64 inference and independent native assembly.
 bridge=FEMBridge(P,threads=2);trusted={}
 for label,r in records.items():
  net=DynamicGraph(stats).double();net.load_state_dict(torch.load(out/f'{label}_seed11.pt',map_location='cpu',weights_only=False)['model']);net.eval()
  with torch.no_grad():q=net(*[torch.tensor(data['validation'][k],dtype=torch.float64) for k in ['coarse','globals','edges']],top.cpu(),torch.tensor(data['validation']['pressure'],dtype=torch.float64));vv,aa=[torch.tensor(x)@q for x in newmark_operators(n,dt)];z=torch.zeros_like(q[:,:1]);old=[torch.cat((z,x[:,:-1]),1) for x in [q,vv,aa]];u=net.decode(q).reshape(len(q),n,-1,3).numpy();rr=[]
  for b,t in valitems:
   case=str(data['validation']['case_ids'][b]);h={k:net.decode(x[b,t]).reshape(-1,3).detach().numpy() for k,x in zip(['u_old','v_old','a_old'],old)};rr.append(float(bridge.evaluate(u[b,t],float(data['validation']['pressure'][b,t])*16070.954,False,geometry_path=P/f'data/processed/cardiac_dynamic_v1/{case}/geometry.npz',dt=dt,history=h)['loss']))
  trusted[label]={'component_rmse_m':float(np.sqrt(np.mean((u-data['validation']['u'])**2))),'sampled_full_residual_loss':float(np.mean(rr))}
 eligible=[k for k in records if k!='data' and trusted[k]['component_rmse_m']<=1.1*trusted['data']['component_rmse_m']];chosen=min(eligible,key=lambda k:trusted[k]['sampled_full_residual_loss']) if eligible else min((k for k in records if k!='data'),key=lambda k:trusted[k]['component_rmse_m']);lam=records[chosen]['lambda']
 selection={'selected':chosen,'lambda':lam,'meets_10percent_constraint':bool(eligible),'trusted_validation':trusted,'lambda_balance':balance};print('DYNAMIC SELECTION',name,selection,flush=True)
 for seed in cfg['seeds'][1:]:
  train(seed,0.,'warm',warm=True);initial=out/f'warm_seed{seed}.pt';train(seed,0.,'data',initial=initial);train(seed,lam,chosen,initial=initial)
 result={'complete':True,'config':cfg,'regime':args.regime,'selection':selection,'data_access':['train','validation'],'normalizer_sha256':hashlib.sha256((D/'normalizers.npz').read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),P/'src/lv_surrogate/models/dynamic_graph.py',P/'src/lv_surrogate/physics/cardiac_residual.py']},'checkpoints':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for label in ['data',chosen] for f in out.glob(f'{label}_seed*.pt') if 'latest' not in f.name}}
 (P/f'experiments/manifests/{name}.json').write_text(json.dumps(result,indent=2));print('DYNAMIC TRAINING COMPLETE',name,flush=True)
finally:
 if bridge:bridge.close()
