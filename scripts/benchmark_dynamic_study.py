"""Isolated matched full-cycle CPU float64 timing, including uncertainty outputs."""
from pathlib import Path
import json,sys,time,subprocess,os,platform,hashlib,argparse
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.dynamic_graph import DynamicGraph
from lv_surrogate.eval.fallback import InputDomainGate
ap=argparse.ArgumentParser();ap.add_argument('--repeat-benchmark',action='store_true');args=ap.parse_args();torch.set_num_threads(2);cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());evaluation=json.loads((P/'experiments/metrics/dynamic_final_evaluation.json').read_text());key=evaluation['preferred_from_validation_only'];regime,label=key.split('/');tcfg=json.loads((P/'configs/physics/dynamic_training.json').read_text());name=tcfg['name']+'_'+regime;stats=dict(np.load(P/f'data/processed/{tcfg.get("data_tag","cardiac_dynamic_ml")}_{regime}/normalizers.npz'));case=next(c for c in cfg['cases'] if c['split']=='test');root=P/f'data/processed/cardiac_dynamic_v1/{case["id"]}';top=torch.tensor(np.load(P/'data/processed/cardiac_geometry_ml/topology.npy'));cluster=np.load(P/'data/processed/cardiac_geometry_ml/cluster_labels.npy');counts=np.bincount(cluster,minlength=64);nets=[];t=time.perf_counter()
for seed in [11,22,33,44,55]:
 net=DynamicGraph(stats).double();net.load_state_dict(torch.load(P/f'experiments/checkpoints/{name}/{label}_seed{seed}.pt',map_location='cpu',weights_only=False)['model']);net.eval();nets.append(net)
load_seconds=time.perf_counter()-t
timing_path=P/'experiments/metrics/dynamic_runtime.json';run_manifest=json.loads((P/f'experiments/manifests/{name}.json').read_text())
if timing_path.exists() and not args.repeat_benchmark:
 old=json.loads(timing_path.read_text())
 if old.get('frozen_checkpoint_sha256')!=run_manifest['checkpoints'] or old['case']!=case:raise RuntimeError('Existing timings refer to different frozen model/task')
 print('REUSED recorded isolated dynamic timings; no fresh benchmark executed',flush=True);sys.exit(0)
with np.load(root/'trajectory.npz') as a:pressure=a['pressure'].copy()
geo=dict(np.load(root/'geometry.npz'));X=geo['coords'];C=geo['cells'];inv=np.linalg.inv((X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1))
def preprocess(geo):
 x=np.c_[geo['coords']/.1,geo['fiber'],geo['sheet']];pooled=np.zeros((64,9));np.add.at(pooled,cluster,x);pooled/=counts[:,None];delta=pooled[top[:,1],:3]-pooled[top[:,0],:3]
 return [torch.tensor(v[None],dtype=torch.float64) for v in [pooled,np.r_[(np.array(case['scales'])-1)/.05,0.],np.c_[delta,np.linalg.norm(delta,axis=-1)],pressure/16070.954]]
def predict(b):
 q=torch.stack([net(*b[:3],top,b[3])[0] for net in nets]);fields=nets[0].decode(q).reshape(5,200,-1,3);return fields.mean(0),fields.var(0,unbiased=True)
def minJ(u):
 minimum=1.
 old=np.concatenate((np.zeros_like(u[:1]),u[:-1]),axis=0)
 for chunk in np.array_split(np.concatenate((u,.4*old+.6*u),axis=0),40):
  F=np.eye(3)+np.einsum('beij,ejk->beik',(chunk[:,C[:,1:]]-chunk[:,C[:,:1]]).transpose(0,1,3,2),inv);minimum=min(minimum,float(np.linalg.det(F).min()))
 return minimum
b=preprocess(geo);resident=[];input_times=[];with_checks=[]
with torch.inference_mode():
 for _ in range(5):mean,var=predict(b)
 for _ in range(30):
  t=time.perf_counter();mean,var=predict(b);resident.append(time.perf_counter()-t)
 for _ in range(20):
  t=time.perf_counter()
  with np.load(root/'geometry.npz') as source:ready=preprocess(source)
  mean,var=predict(ready);input_times.append(time.perf_counter()-t)
 for _ in range(10):
  t=time.perf_counter();mean,var=predict(b);jac=minJ(mean.numpy());with_checks.append(time.perf_counter()-t);assert jac>0
# No FE workers should overlap this isolated benchmark (pipeline waits for completion).
env=os.environ.copy();env.update(FI_PROVIDER='tcp',FI_TCP_IFACE='en0',XDG_CACHE_HOME=str(P/'.cache'),DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'),MAMBA_ROOT_PREFIX=str(P/'.mamba'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');fe=[];walls=[];identical=[]
for repeat in range(2):
 suffix=f'_benchmark{repeat}'+(f'_{time.time_ns()}' if args.repeat_benchmark else '');folder=P/f'data/processed/cardiac_dynamic_v1/{case["id"]}{suffix}'
 if (folder/'manifest.json').exists():raise RuntimeError('Benchmark output exists; preserve prior timings rather than silently timing cache reuse')
 t=time.perf_counter();subprocess.run([str(P/'.tools/bin/micromamba'),'run','-p',str(P/'.fenics'),'python',str(P/'scripts/generate_dynamic_study.py'),'--case',case['id'],'--output-suffix',suffix],cwd=P,env=env,check=True);walls.append(time.perf_counter()-t);m=json.loads((folder/'manifest.json').read_text());assert m['complete'];fe.append(sum(s['nonlinear_solver_seconds'] for s in m['steps']));truth=np.load(root/'trajectory.npz')['u'];repeated=np.load(folder/'trajectory.npz')['u'];identical.append(float(np.max(abs(truth-repeated))))
def summary(a):return {'n':len(a),'median_s':float(np.median(a)),'p95_s':float(np.quantile(a,.95))}
result={'frozen_checkpoint_sha256':run_manifest['checkpoints'],'model':key,'case':case,'hardware':{'machine':platform.machine(),'system':platform.system(),'cpu':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()},'precision':'CPU float64 weights cast from float32 training; FE float64','threads':2,'task':'full200-step passive rollout from rest,4577 nodes; ensemble emits full displacement mean and coordinate sample variance','NN_resident_mean_and_variance':summary(resident),'NN_with_input_processing':summary(input_times),'NN_with_endpoint_and_stress_state_J_checks':summary(with_checks),'FE_sum_nonlinear_solve_seconds':summary(fe),'FE_process_setup_generation_IO_and_audit_wall':summary(walls),'median_speedup_resident':float(np.median(fe)/np.median(resident)),'median_speedup_input_processing':float(np.median(fe)/np.median(input_times)),'median_speedup_with_J_checks':float(np.median(fe)/np.median(with_checks)),'five_member_load_seconds':load_seconds,'FE_repeat_max_absolute_displacement_difference_m':identical,'scope':'isolated same host, same geometry/loading/discretization, two fresh FE full-cycle runs; n2 FE timing precision limited; no speedup claimed for rejected requests routed to FE','raw_seconds':{'resident':resident,'input_processing':input_times,'J_checks':with_checks,'FE_solver':fe,'FE_wall':walls},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(P/'experiments/metrics/dynamic_runtime.json').write_text(json.dumps(result,indent=2));print('DYNAMIC TIMING COMPLETE',result['median_speedup_input_processing'],flush=True)
