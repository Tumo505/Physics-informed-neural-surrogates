"""Isolated CPU batch-one timing; no matched FEM measurement, hence no speedup."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from lv_surrogate.data.dalton import Dalton
from lv_surrogate.models.message_passing import GraphSurrogate
P=Path(__file__).resolve().parents[1];torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
t0=time.perf_counter();d=Dalton(P/'data/raw/lvData','test');c=d[0];st=np.load(P/'experiments/checkpoints/pilot_normalizers.npz');io_seconds=time.perf_counter()-t0
s,r=d.edges.T

def preprocess():
 x=np.concatenate((c['real-node-coords'],c['real-node-features']),-1);g=np.concatenate((c['global-features'],c['shape-coeffs'][:32]));delta=c['real-node-coords'][r]-c['real-node-coords'][s];e=np.concatenate((delta,np.linalg.norm(delta,axis=-1,keepdims=True)),-1)
 return [torch.from_numpy(((a-st[n+'mean'])/st[n+'std']).astype('float32')) for a,n in zip([x,g,e],['x','g','e'])]
x,g,e=preprocess();topo=torch.from_numpy(d.edges);t0=time.perf_counter();net=GraphSurrogate(len(x[0]),len(g));net.load_state_dict(torch.load(P/'experiments/checkpoints/gnn_seed11.pt',weights_only=True));net.eval();load=time.perf_counter()-t0
u_std=torch.from_numpy(st['ustd']);u_mean=torch.from_numpy(st['umean'])
with torch.inference_mode():
 for _ in range(10):net(x,g,e,topo)
 forward=[];full=[];pre=[]
 for _ in range(100):
  t=time.perf_counter();out=net(x,g,e,topo)*u_std+u_mean;forward.append(time.perf_counter()-t)
 for _ in range(100):
  t=time.perf_counter();xx,gg,ee=preprocess();pre.append(time.perf_counter()-t);out=net(xx,gg,ee,topo)*u_std+u_mean;full.append(time.perf_counter()-t)
out={'hardware':'Apple M5 Pro CPU, 2 torch threads','precision':'float32','n_nodes':len(x),'n_edges':len(topo),'batch':1,'repetitions':100,'warmup':10,'model':'one modern pilot seed11; ensemble requires five passes','model_load_seconds':load,'case_load_seconds':io_seconds,'inference_ms_median':float(np.median(forward)*1000),'inference_ms_p95':float(np.quantile(forward,.95)*1000),'preprocess_ms_median':float(np.median(pre)*1000),'end_to_end_resident_case_ms_median':float(np.median(full)*1000),'end_to_end_resident_case_ms_p95':float(np.quantile(full,.95)*1000),'fem_solution_seconds':None,'speedup':None,'reason':'No matched FEM solution exists on this host/task; no speedup or break-even claim','raw_forward_seconds':forward,'raw_e2e_seconds':full}
(P/'experiments/metrics/runtime.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if not k.startswith('raw')})
