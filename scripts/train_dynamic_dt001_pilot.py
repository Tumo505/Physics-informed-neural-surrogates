"""Original1ms temporal benchmark with causal bounded recurrent whole-cycle losses."""
from pathlib import Path
import sys,json,time,hashlib,math,numpy as np,torch
from scipy.linalg import svd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.dynamic_graph import DynamicGraph,newmark_operators
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.physics.fem_bridge import FEMBridge
torch.set_num_threads(2);cfg=json.loads((P/'configs/physics/dynamic_dt001_pilot.json').read_text());name=cfg['name'];out=P/'experiments/checkpoints'/name;out.mkdir(exist_ok=True);metric=P/f'experiments/metrics/{name}.json'
if metric.exists():
 old=json.loads(metric.read_text());labels=['data',old.get('selection','')]
 if old.get('complete') and all((out/f'{label}.pt').exists() and (P/f'experiments/metrics/{name}_{label}_prediction.npz').exists() for label in labels):print('REUSED ORIGINAL1ms RECURRENT PILOT',flush=True);sys.exit(0)
values={'u':[],'pressure':[],'time':[]}
for i in range(999):
 with np.load(P/f'data/processed/cardiac_passive_trajectory_v1/step_{i:04d}.npz') as a:
  for k in values:values[k].append(a[k])
a={k:np.array(v) for k,v in values.items()};target=a['u'].reshape(999,-1);ntrain=600;nval=800;dt=.001;_,s,V=svd(target[:ntrain],full_matrices=False);basis=V[:12];coefficient=target@basis.T;scale=np.maximum(np.sqrt(np.mean(coefficient[:ntrain]**2,axis=0)),1e-8);qtrue=coefficient/scale;stats={'basis':basis,'coefficient_scale':scale,'coefficient_bound':1.5*abs(qtrue[:ntrain]).max(0)+.1};field_scale=float(np.sqrt(np.mean(target[:ntrain]**2)));diff_scales=[float(np.sqrt(np.mean(np.sum(np.diff(coefficient[:ntrain],n=o,axis=0)**2,axis=-1)/target.shape[1]))) for o in [1,2]];np.savez(out/'normalizers.npz',**stats);top=torch.tensor(np.load(P/'data/processed/cardiac_geometry_ml/topology.npy'));labels=np.load(P/'data/processed/cardiac_geometry_ml/cluster_labels.npy');g=dict(np.load(P/'data/processed/cardiac/cohort.npz'));counts=np.bincount(labels,minlength=64);features=np.c_[g['coords']/.1,g['fiber'],g['sheet']];coarse=np.zeros((64,9));np.add.at(coarse,labels,features);coarse/=counts[:,None];delta=coarse[top[:,1],:3]-coarse[top[:,0],:3];inputs=[torch.tensor(v[None],dtype=torch.float64) for v in [coarse,np.zeros(3),np.c_[delta,np.linalg.norm(delta,axis=-1)]]];pressure=torch.tensor(a['pressure'][None]/16070.954);truth=torch.tensor(qtrue[None]);own=CardiacFirstStep(g,dt=dt,analytic_piola=True);force_scale=float(g['force_scale_N']);maps={n:tuple(torch.tensor(x) for x in newmark_operators(n,dt)) for n in [600,800,999]};valframes=[600,625,650,675,700,725,750,775,799];bridge=None

def predict(net,n):return net(*inputs,top,pressure[:,:n])[0]
def histories(q):
 vmap,amap=maps[len(q)];v=vmap@q;acc=amap@q;zero=torch.zeros_like(q[:1]);return [torch.cat((zero,x[:-1]),0) for x in [q,v,acc]]
def data_loss(q,lo,hi):
 diff=(q[lo:hi]-truth[0,lo:hi])*q.new_tensor(scale);loss=(diff.square().sum(-1)/target.shape[1]).mean()/field_scale**2
 for order,weight,dscale in [(1,.1,diff_scales[0]),(2,.01,diff_scales[1])]:
  dd=(torch.diff(q[lo:hi],n=order,dim=0)-torch.diff(truth[0,lo:hi],n=order,dim=0))*q.new_tensor(scale);loss=loss+weight*(dd.square().sum(-1)/target.shape[1]).mean()/max(dscale,1e-12)**2
 return loss

def eq_loss(q,net,frames):
 old=histories(q);loss=[]
 for t in frames:
  u=net.decode(q[t]).reshape(-1,3);h={k:net.decode(x[t]).reshape(-1,3) for k,x in zip(['u_old','v_old','a_old'],old)};r=own(u,float(a['pressure'][t]),create_graph=torch.is_grad_enabled(),history=h);loss.append((r/force_scale).square().mean())
 return torch.stack(loss).mean()
def validation(net,lam,warm=False):
 with torch.no_grad():q=predict(net,800);d=float(data_loss(q,600,800));err=net.decode(q[600:]).numpy()-target[600:800];eq=0. if warm else float(eq_loss(q,net,valframes))
 with torch.no_grad():training_error=net.decode(q[:600]).numpy()-target[:600]
 relative=float(np.sqrt(np.mean(training_error**2))/field_scale);eligible=relative<=cfg.get('maximum_training_relative_RMSE',.15)
 return {'component_rmse_m':float(np.sqrt(np.mean(err**2))),'training_relative_RMSE':relative,'training_fit_gate_passed':eligible,'force_loss':eq,'score':d+lam*eq if eligible else float('inf')}
def train(label,lam,initial=None,warm=False):
 path=out/f'{label}.pt';torch.manual_seed(cfg['seed']);net=DynamicGraph(stats).double()
 if initial:net.load_state_dict(torch.load(initial,weights_only=False)['model'])
 steps=cfg['warm_updates'] if warm else cfg['refinement_updates'];opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'] if warm else cfg['refinement_lr']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=2e-5);val=validation(net,lam,warm);best=val['score'];bestval=val;torch.save({'model':net.state_dict()},path);start=time.perf_counter();rows=[];rng=np.random.default_rng(11)
 for step in range(steps):
  opt.zero_grad();q=predict(net,600);loss=data_loss(q,0,600)
  if lam:loss=loss+lam*eq_loss(q,net,[0 if step%10==0 else int(rng.integers(600))])
  if not torch.isfinite(loss):raise FloatingPointError(label)
  loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();sched.step()
  if (step+1)%50==0:
   val=validation(net,lam,warm);rows.append({'step':step,'validation':val});print('1ms',label,step,val,flush=True)
   if val['score']<best:best=val['score'];bestval=val;torch.save({'model':net.state_dict()},path)
 if not np.isfinite(best):raise RuntimeError('No nontrivial model met the training-fit acceptance gate')
 return {'lambda':lam,'best_validation':bestval,'seconds':time.perf_counter()-start,'history':rows}
records={};records['warm']=train('warm',0.,warm=True);initial=out/'warm.pt';net=DynamicGraph(stats).double();net.load_state_dict(torch.load(initial,weights_only=False)['model']);q=predict(net,600);dl=data_loss(q,0,600);gd=torch.autograd.grad(dl,list(net.parameters()),retain_graph=True);ratios=[]
for t in [99,199,299,499]:
 eq=eq_loss(q,net,[t]);ge=torch.autograd.grad(eq,list(net.parameters()),retain_graph=True);ratios.append(float(sum(x.square().sum() for x in gd).sqrt()/sum(x.square().sum() for x in ge).sqrt().clamp_min(1e-20)))
center=float(10**np.round(np.log10(np.median(ratios))));grid=[center/10,center,center*10];records['data']=train('data',0.,initial)
for lam in grid:records[f'physics_{lam:g}']=train(f'physics_{lam:g}',lam,initial)
baseline=records['data']['best_validation']['component_rmse_m'];eligible=[k for k in records if k.startswith('physics_') and records[k]['best_validation']['component_rmse_m']<=1.1*baseline];selected=min(eligible,key=lambda k:records[k]['best_validation']['force_loss']) if eligible else min((k for k in records if k.startswith('physics_')),key=lambda k:records[k]['best_validation']['component_rmse_m']);print('1ms PHYSICS SELECTED',selected,flush=True)
# Native final residuals and all-cell/all-state Jacobians, no fitted state supplied.
bridge=FEMBridge(P,threads=2);results={};X=g['coords'];C=g['cells'];inv=np.linalg.inv((X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1))
try:
 for label in ['data',selected]:
  net=DynamicGraph(stats).double();net.load_state_dict(torch.load(out/f'{label}.pt',weights_only=False)['model']);net.eval()
  with torch.inference_mode():q=predict(net,999);u=net.decode(q).reshape(999,-1,3).numpy();old=histories(q);uaf=net.decode(.4*old[0]+.6*q).reshape(999,-1,3).numpy()
  minima=[]
  for field in [u,uaf]:
   minimum=1.
   for chunk in np.array_split(field,50):F=np.eye(3)+np.einsum('beij,ejk->beik',(chunk[:,C[:,1:]]-chunk[:,C[:,:1]]).transpose(0,1,3,2),inv);minimum=min(minimum,float(np.linalg.det(F).min()))
   minima.append(minimum)
  rr=[]
  for t in np.linspace(800,998,12,dtype=int):
   h={k:net.decode(x[t]).reshape(-1,3).detach().numpy() for k,x in zip(['u_old','v_old','a_old'],old)};rr.append(float(bridge.evaluate(u[t],float(a['pressure'][t]),False,history=h)['loss'])**.5)
  results[label]={'completed_steps':999,'full_cycle_component_rmse_m':float(np.sqrt(np.mean((u-a['u'])**2))),'test_block_component_rmse_m':float(np.sqrt(np.mean((u[800:]-a['u'][800:])**2))),'minimum_endpoint_detF':minima[0],'minimum_stress_state_detF':minima[1],'test_block_sampled_normalized_force_rms':float(np.mean(rr)),'training_relative_RMSE':float(np.sqrt(np.mean((u[:600]-a['u'][:600])**2))/field_scale),'nontrivial_training_fit':float(np.sqrt(np.mean((u[:600]-a['u'][:600])**2))/field_scale)<=cfg.get('maximum_training_relative_RMSE',.15),'full_cycle_noninverted':min(minima)>0};np.savez_compressed(P/f'experiments/metrics/{name}_{label}_prediction.npz',prediction=u,truth=a['u'],time=a['time'])
 result={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'complete':True,'config':cfg,'training_gradient_balance_ratios':ratios,'lambda_grid':grid,'selection':selected,'meets_validation_10percent_constraint':bool(eligible),'training_records':records,'models':results,'scope':cfg['scope'],'evaluation':'repeated temporal evaluation; first600 frames train, next200 select, last199 test; inference over all999 starts from zero and uses no true histories','caveat':'one geometry/cycle/seed, not an independent population; no conformal claim'};
 def serializable(v):
  if isinstance(v,dict):return {k:serializable(x) for k,x in v.items()}
  if isinstance(v,list):return [serializable(x) for x in v]
  if isinstance(v,float) and not math.isfinite(v):return None
  return v
 metric.write_text(json.dumps(serializable(result),indent=2,allow_nan=False));print('ORIGINAL1ms RECURRENT PILOT COMPLETE',results,flush=True)
finally:bridge.close()
