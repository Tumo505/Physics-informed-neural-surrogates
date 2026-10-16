"""Batch-one full-field MPS/CPU latency, synchronizing asynchronous GPU work."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch,yaml
from lv_surrogate.models.multiscale import MultiscaleSurrogate
P=Path(__file__).resolve().parents[1];D=P/'data/processed/full';cfg=yaml.safe_load((P/'configs/gnn/full_low_lr.yaml').read_text());torch.set_num_threads(2)
base=P/'experiments/checkpoints'/cfg['run_name'];raw={n:np.load(D/f'test_{n}.npy',mmap_mode='r')[64:65].copy() for n in ['points','coarse','globals','edges']};top=np.load(D/'topology.npy');par=np.load(D/'parents.npy');norm=np.load(D/'normalizers.npz');all_results={}
for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
 t0=time.perf_counter();models=[]
 for seed in cfg['seeds']:
  m=MultiscaleSurrogate(width=cfg['width'],steps=cfg['steps']).to(device);m.load_state_dict(torch.load(base/f'gnn_{seed}_best.pt',map_location=device,weights_only=False)['model']);m.eval();models.append(m)
 load_seconds=time.perf_counter()-t0;b={n:torch.tensor(v,device=device) for n,v in raw.items()};t=torch.tensor(top,device=device);parents=torch.tensor(par,device=device);std=torch.tensor(norm['targets_std'],device=device);mu=torch.tensor(norm['targets_mean'],device=device)
 def sync():
  if device=='mps':torch.mps.synchronize()
 def inference(ms):
  return torch.stack([m(b['points'],b['coarse'],b['globals'],b['edges'],t,parents)*std+mu for m in ms]).mean(0)
 with torch.inference_mode():
  for _ in range(10):inference(models[:1])
  sync();out={}
  for label,ms in [('one_member',models[:1]),('five_member_ensemble',models)]:
   times=[]
   for _ in range(100):
    sync();t0=time.perf_counter();y=inference(ms);sync();times.append(time.perf_counter()-t0)
   out[label]={'median_ms':float(np.median(times)*1000),'p95_ms':float(np.quantile(times,.95)*1000),'raw_seconds':times}
 all_results[device]={'batch':1,'output_shape':[6784,3],'precision':'float32','cpu_threads':2,'resident_preprocessed_inputs':True,'normalizer_inverse_included':True,'repetitions':100,'warmup':10,'gpu_synchronized':device=='mps','model_load_five_seconds':load_seconds,'timing':out}
result={'hardware':'Apple M5 Pro','results':all_results,'fem_speedup':None,'limitation':'No matched Dalton FEM solver; inference-only, preprocessing and corpus generation separate'}
(P/'experiments/metrics/full_runtime.json').write_text(json.dumps(result,indent=2));print({d:{k:v['median_ms'] for k,v in x['timing'].items()} for d,x in all_results.items()})
