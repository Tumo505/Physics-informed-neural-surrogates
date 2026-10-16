"""Matched history-conditioned neural/POD temporal pilot; train/validation only."""
from pathlib import Path
import json,sys,copy,time,argparse
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.temporal_pod import TemporalPOD
from lv_surrogate.physics.fem_bridge import FEMBridge
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/physics/temporal_model.json');args=ap.parse_args()
torch.set_num_threads(2);cfg=json.loads((P/args.config).read_text());D=P/'data/processed/cardiac_temporal_ml';stats=dict(np.load(D/'normalizers.npz'));dt=float(stats['dt']);scale=float(stats['delta_scale']);data={s:dict(np.load(D/f'{s}.npz')) for s in ['train','validation']}
verification=json.loads((P/'experiments/metrics/extended_mechanics_verification.json').read_text())
if not verification['passed'] or not any(r['task']=='temporal_step400' for r in verification['cases']):raise RuntimeError('Late-cycle mechanics gate missing')
tensors={s:{k:torch.tensor(a[k],dtype=torch.float64) for k in ['features','delta']} for s,a in data.items()};own=CardiacFirstStep(np.load(P/'data/processed/cardiac/cohort.npz'),analytic_piola=True);force_scale=float(np.load(P/'data/processed/cardiac/cohort.npz')['force_scale_N'])
if not json.loads((P/'experiments/metrics/temporal_backend.json').read_text())['passed']:raise RuntimeError('Float64 analytic temporal gate missing')
bridge=FEMBridge(P,threads=2);out=P/'experiments/checkpoints'/cfg['name'];out.mkdir(exist_ok=True);metrics=P/'experiments/metrics'/cfg['name'];metrics.mkdir(exist_ok=True);records={}
def displacement(da,a,indices):
 return torch.tensor(a['u_old'][indices]+dt*a['v_old'][indices]+.5*dt**2*a['a_old'][indices],dtype=torch.float64)+.36*dt**2*da.reshape(-1,*a['u_old'].shape[1:])
def history(a,i):return {k:a[k][i] for k in ['u_old','v_old','a_old']}
def validation(model):
 a=data['validation']
 with torch.no_grad():da=model(tensors['validation']['features']);mse=float((da-tensors['validation']['delta']).square().mean());pred=displacement(da,a,np.arange(len(da))).numpy()
 subset=np.linspace(0,len(da)-1,cfg['validation_residual_frames'],dtype=int)
 eq=float(np.mean([float(bridge.evaluate(pred[i],a['pressure'][i],False,history=history(a,i))['loss']) for i in subset]));return {'scaled_correction_mse':mse/scale**2,'component_rmse_m':float(np.sqrt(np.mean((pred-a['u'])**2))),'residual_loss_eight_frames':eq}
try:
 for lam in [0.]+cfg['physics_lambdas']:
  name='data' if not lam else f'physics_lambda{lam:g}';recpath=metrics/f'{name}.json';path=out/f'{name}.pt'
  if recpath.exists() and json.loads(recpath.read_text()).get('complete'):records[name]=json.loads(recpath.read_text());continue
  torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);net=TemporalPOD(stats).double();opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['training_updates'],eta_min=.0001);best=float('inf');hist=[];start=time.perf_counter()
  for step in range(cfg['training_updates']):
   ids=rng.choice(len(data['train']['features']),cfg['batch_size'],replace=False);opt.zero_grad();da=net(tensors['train']['features'][ids]);loss_data=(da-tensors['train']['delta'][ids]).square().mean()/scale**2;eq=da.new_zeros(())
   if lam:
    i=int(ids[step%len(ids)]);up=displacement(da[step%len(ids):step%len(ids)+1],data['train'],[i])[0];R=own(up,float(data['train']['pressure'][i]),create_graph=True,history=history(data['train'],i));eq=(R/force_scale).square().mean()
   loss=loss_data+lam*eq
   if not torch.isfinite(loss):raise FloatingPointError(name)
   loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();sched.step()
   if step%cfg['validation_every']==0 or step==cfg['training_updates']-1:
    val=validation(net);score=val['scaled_correction_mse']+lam*val['residual_loss_eight_frames'];hist.append({'step':step,'validation':val,'score':score})
    if score<best:best=score;bestval=val;torch.save({'model':net.state_dict(),'validation':val,'step':step},path)
    print(name,step,bestval,flush=True)
  record={'complete':True,'lambda':lam,'seed':cfg['seed'],'best_validation':bestval,'history':hist,'seconds':time.perf_counter()-start};recpath.write_text(json.dumps(record,indent=2));records[name]=record
 baseline=records['data']['best_validation']['component_rmse_m'];eligible=[n for n in records if n!='data' and records[n]['best_validation']['component_rmse_m']<=1.1*baseline]
 chosen=min(eligible,key=lambda n:records[n]['best_validation']['residual_loss_eight_frames']) if eligible else min((n for n in records if n!='data'),key=lambda n:records[n]['best_validation']['component_rmse_m'])
 (P/f'experiments/manifests/{cfg["name"]}_selection.json').write_text(json.dumps({'config':cfg,'selected':chosen,'meets_10percent_constraint':bool(eligible),'validation':{k:r['best_validation'] for k,r in records.items()},'calibration_test_read':False,'complete':True},indent=2));print('TEMPORAL TRAINING COMPLETE',chosen,flush=True)
finally:bridge.close()
