"""Prespecified small Track A experiment; train only on train IDs, select on validation."""
from pathlib import Path
import sys,json,time,copy,random,hashlib,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch,yaml
from lv_surrogate.data.dalton import Dalton
from lv_surrogate.models.message_passing import GraphSurrogate
P=Path(__file__).resolve().parents[1]

def main():
 cfg=yaml.safe_load((P/'configs/gnn/pilot.yaml').read_text());torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
 sets={s:Dalton(P/'data/raw/lvData',s) for s in ['train','validation','test']}
 rng=np.random.default_rng(cfg['seed_split']);ids=rng.choice(len(sets['train']),cfg['train_cases'],replace=False).tolist()
 split={'train':ids,'validation':list(range(*cfg['validation_ids'])),'calibration':list(range(*cfg['calibration_ids'])),'test':list(range(*cfg['test_ids']))}
 (P/'data/splits/pilot.json').write_text(json.dumps(split,indent=2))
 # All statistics computed on training cases, including coords and edge geometry.
 def raw(c):
  x=np.concatenate((c['real-node-coords'],c['real-node-features']),-1)
  g=np.concatenate((c['global-features'],c['shape-coeffs'][:32]))
  s,r=sets['train'].edges.T;delta=c['real-node-coords'][r]-c['real-node-coords'][s];e=np.concatenate((delta,np.linalg.norm(delta,axis=-1,keepdims=True)),-1)
  return x,g,e,c['real-node-displacement']
 train_raw=[raw(sets['train'][i]) for i in ids]
 stats={}
 for j,name in enumerate(['x','g','e','u']):
  a=np.stack([r[j] for r in train_raw]).astype('float64');axes=(0,) if j==1 else (0,1);stats[name+'mean']=a.mean(axis=axes).astype('float32');stats[name+'std']=np.maximum(a.std(axis=axes),1e-8).astype('float32')
 np.savez(P/'experiments/checkpoints/pilot_normalizers.npz',**stats)
 def tensorize(row):
  return tuple(torch.from_numpy(((a-stats[n+'mean'])/stats[n+'std']).astype('float32')) for a,n in zip(row,['x','g','e','u']))
 cached={'train':[tensorize(r) for r in train_raw]}
 for s,src in [('validation','validation'),('calibration','validation'),('test','test')]:cached[s]=[tensorize(raw(sets[src][i])) for i in split[s]]
 del train_raw
 topo=torch.from_numpy(sets['train'].edges);ustd=torch.from_numpy(stats['ustd']);scale=(ustd**2).sum();um=torch.from_numpy(stats['umean'])
 all_meta=[]
 for seed in cfg['seeds']:
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
  net=GraphSurrogate(cached['train'][0][0].shape[-1],cached['train'][0][1].shape[-1],cfg['width'],cfg['message_steps']);opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']);best=float('inf');history=[];start=time.perf_counter()
  for epoch in range(cfg['epochs']):
   net.train();losses=[]
   for i in np.random.permutation(len(ids)):
    x,g,e,u=cached['train'][i];opt.zero_grad();pred=net(x,g,e,topo);loss=(((pred-u)*ustd)**2).sum(-1).mean()/scale
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);opt.step();losses.append(loss.item())
   net.eval()
   with torch.no_grad():v=float(np.mean([((((net(x,g,e,topo)-u)*ustd)**2).mean()).item() for x,g,e,u in cached['validation']]))
   history.append({'epoch':epoch,'train_scaled_loss':float(np.mean(losses)),'validation_component_mse_cm2':v})
   if v<best:best=v;state=copy.deepcopy(net.state_dict());best_epoch=epoch
   print('seed',seed,'epoch',epoch,'val_component_rmse_cm',v**.5,flush=True)
  elapsed=time.perf_counter()-start;net.load_state_dict(state);torch.save(state,P/f'experiments/checkpoints/gnn_seed{seed}.pt')
  # Validation, calibration and pilot test predictions after selection is frozen.
  with torch.no_grad():
   for s in ['validation','calibration','test']:
    preds=np.stack([(net(x,g,e,topo)*ustd+um).numpy() for x,g,e,u in cached[s]])
    np.save(P/f'experiments/predictions/seed{seed}_{s}.npy',preds)
  meta={'seed':seed,'train_seconds':elapsed,'best_epoch':best_epoch,'best_validation_mse_cm2':best,'parameter_count':sum(p.numel() for p in net.parameters()),'history':history};all_meta.append(meta)
  (P/'experiments/metrics/training.json').write_text(json.dumps(all_meta,indent=2))
 manifest={'command':sys.argv,'config':cfg,'split':split,'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),'device':'cpu','mps_available':torch.backends.mps.is_available(),'source_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'src').rglob('*.py')},'normalizer_sha256':hashlib.sha256((P/'experiments/checkpoints/pilot_normalizers.npz').read_bytes()).hexdigest()}
 (P/'experiments/manifests/gnn_pilot.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':
 (P/'experiments/predictions').mkdir(exist_ok=True);main()
