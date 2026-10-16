"""Matched batch-one CPU float64 latency and warm FE startup timings."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.geometry_pod import GeometryPOD
from lv_surrogate.physics.fem_bridge import FEMBridge
ap=argparse.ArgumentParser();ap.add_argument('--run',default='cardiac_geometry_models_v2');ap.add_argument('--data-root',default='data/processed/cardiac_geometry_ml');ap.add_argument('--cohort',default='cardiac_geometry_v1');ap.add_argument('--output',default='geometry_runtime');args=ap.parse_args()
torch.set_num_threads(2);name=args.run;manifest=json.loads((P/f'experiments/manifests/{name}.json').read_text());lam=manifest['lambda_selection']['lambda'];label=f'physics_lambda{lam:g}';D=P/args.data_root;a=dict(np.load(D/'test.npz'));norm=np.load(D/'normalizers.npz');top=torch.from_numpy(np.load(D/'topology.npy'));models=[];start=time.perf_counter()
for seed in manifest['config']['seeds']:
 net=GeometryPOD(*[norm[k] for k in ['field_mean','basis','coefficient_scale']]).double();ck=torch.load(P/f'experiments/checkpoints/{name}/{label}_seed{seed}_best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);net.eval();models.append(net)
load_seconds=time.perf_counter()-start
cluster_labels=np.load(D/'cluster_labels.npy');cluster_counts=np.bincount(cluster_labels,minlength=64);preprocessing=[];joint=[]
def preprocess(geom,p):
 features=np.c_[geom['coords'],geom['fiber'],geom['sheet']];pooled=np.zeros((64,9));np.add.at(pooled,cluster_labels,features);pooled/=cluster_counts[:,None];delta=pooled[top[:,1],:3]-pooled[top[:,0],:3];edge=np.c_[delta,np.linalg.norm(delta,axis=-1)];global_features=np.r_[geom['geometry_scales'],p]
 return [(pooled-norm['coarse_mean'])/norm['coarse_std'],(global_features-norm['globals_mean'])/norm['globals_std'],(edge-norm['edges_mean'])/norm['edges_std']]
def ensemble(b):
 samples=torch.stack([net(*b,top) for net in models]);return samples.mean(0),samples.var(0,unbiased=True)
queries=[int(i) for g in np.unique(a['geometry_ids'])[:4] for i in np.where(a['geometry_ids']==g)[0]];nn=[];single=[];fe=[];fe_wall=[];activation=[];bridge=FEMBridge(P,threads=2)
try:
 for i in queries:
  b=[torch.tensor(a[k][i:i+1],dtype=torch.float64) for k in ['coarse','globals','edges']];g=int(a['geometry_ids'][i]);p=float(a['pressures'][i]);path=P/f'data/processed/{args.cohort}/geometry_{g:03d}.npz'
  geom=np.load(path)
  for _ in range(30):
   t=time.perf_counter();normalized=preprocess(geom,p);preprocessing.append(time.perf_counter()-t)
  np.testing.assert_allclose(normalized[0],a['coarse'][i],rtol=1e-10,atol=1e-10)
  with torch.inference_mode():
   for _ in range(10):prediction,variance=ensemble(b)
   for _ in range(50):
    t=time.perf_counter();prediction=models[0](*b,top);single.append(time.perf_counter()-t)
    t=time.perf_counter();prediction,variance=ensemble(b);nn.append(time.perf_counter()-t)
   for _ in range(10):
    t=time.perf_counter()
    with np.load(path) as payload:ready=preprocess(payload,p)
    inputs=[torch.tensor(v[None],dtype=torch.float64) for v in ready];prediction,variance=ensemble(inputs);joint.append(time.perf_counter()-t)
  t=time.perf_counter();bridge.evaluate(np.zeros((4577,3)),p,False,geometry_path=path);activation.append(time.perf_counter()-t)
  bridge.solve(p,geometry_path=path)
  for _ in range(2):
   t=time.perf_counter();r=bridge.solve(p,geometry_path=path);fe_wall.append(time.perf_counter()-t);fe.append(r['seconds'])
 def summary(values):return {'n':len(values),'median_ms':float(np.median(values)*1000),'p95_ms':float(np.quantile(values,.95)*1000)}
 training=[]
 for f in (P/f'experiments/metrics/{name}').glob('*.json'):
  r=json.loads(f.read_text())
  if r.get('complete') and 'seconds' in r:training.append(r['seconds'])
 corpus=json.loads((P/f'experiments/manifests/{args.cohort}.json').read_text());corpus_solver_seconds=sum(c['seconds'] for r in corpus['records'] for c in r['solves']);delta=np.median(fe)-np.median(joint);upfront=corpus['seconds_this_invocation']+sum(training)
 result={'task':'matched passive startup pressure solves on4 held-out numerical geometries,4 loads each','precision':'CPU float64 both FE and neural inference; learned float32 weights cast to float64','threads':2,'n_geometry_load_queries':len(queries),'input_preprocessing_including_warm_file_reads':summary(preprocessing),'ensemble_outputs':['full displacement mean','coordinate sample variance'],'input_to_mean_and_variance':summary(joint),'nn_single':summary(single),'nn_five_member_ensemble':summary(nn),'warm_FE_solver':summary(fe),'warm_FE_roundtrip_including_local_payload':summary(fe_wall),'warm_median_speedup_ensemble':float(np.median(fe)/np.median(nn)),'warm_median_speedup_with_input_processing':float(np.median(fe)/np.median(joint)),'model_load_seconds_five_members':load_seconds,'geometry_activation_wall_ms':summary(activation),'new_geometry_activation_wall_ms':summary(activation[::4]),'cached_geometry_request_wall_ms':summary([v for j,v in enumerate(activation) if j%4]),'geometry_activation_scope':'first request per geometry includes setup plus zero-field residual and local IO; later loads reuse setup','corpus_solver_seconds':corpus_solver_seconds,'corpus_generation_wall_seconds':corpus['seconds_this_invocation'],'all_gpu_study_training_validation_seconds':sum(training),'measured_cost_recovery_query_count':float(upfront/delta) if delta>0 else None,'warm_start_parent_run':manifest['config'].get('initialize_from_run'),'warm_start_cost_included':False,'cost_accounting':'generation + all GPU study arms, lambda grid and ablations; excludes warm-start parent training/corpus, CPU pilot, verification, preprocessing, temporal study and software setup; partial cost accounting, not full project break-even','scope':'resident normalized mesh/fiber inputs for NN-only timing; per-query graph pooling/feature normalization with warm input-file reads measured separately; source fiber generation excluded; no full-trajectory or Dalton speedup inferred','raw_seconds':{'nn_single':single,'nn_ensemble':nn,'input_to_mean_and_variance':joint,'FE_solver':fe,'FE_roundtrip':fe_wall}}
 (P/f'experiments/metrics/{args.output}.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='raw_seconds'},flush=True)
finally:bridge.close()
