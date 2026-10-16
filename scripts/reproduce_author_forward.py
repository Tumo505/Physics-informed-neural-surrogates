"""Run UNEDITED author models.py with released parameters under current Flax/JAX."""
from pathlib import Path
import sys,json,time,importlib.util,argparse
ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=16);args=ap.parse_args()
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
import numpy as np
import jax,jax.numpy as jnp
from lv_surrogate.data.dalton import Dalton,AuthorPreprocess
from lv_surrogate.models.reference import read_weights,fixed_predict
spec=importlib.util.spec_from_file_location('author_models',P/'external/passive-lv-gnn-emul/models.py');m=importlib.util.module_from_spec(spec);sys.modules['author_models']=m;spec.loader.exec_module(m)
R=P/'external/passive-lv-gnn-emul';fixed=R/'data/lvDataFixedGeom';w=R/'emulationResults/trainedParameters/lvDataFixedGeom'
enc=read_weights(w/'theta_encode_params.pkl');dec=read_weights(w/'trainedNetworkParams.pkl');lat=np.load(w/'latent_nodal_values.npy').astype('float32')
encoder=m.FlaxMLP([128,128,40],True);net=m.DeepGraphEmulatorFixedGeom([128,128],[3],6784,jnp.asarray(lat),lambda x:encoder.apply(enc,x))
theta=np.load(fixed/'processedData/test/global-features.npy').astype('float32')[0:1]
y=np.asarray(net.apply(dec,jnp.asarray(theta)));yn=fixed_predict(theta,enc,dec,lat)
check={'numpy_vs_unedited_author_fixed_max_abs_normalized':float(np.max(np.abs(y-yn))),'jax':jax.__version__,'flax':__import__('flax').__version__,'original_cli':'not run: deprecated flax.optim; legacy jaxlib unavailable for cp312 arm64','adapter':'unedited author models.py, modern runtime, released weights'}
np.testing.assert_allclose(y,yn,rtol=1e-4,atol=1e-4)
d=Dalton(P/'data/raw/lvData','test');proc=AuthorPreprocess(fixed);s,r=proc.edges.T
net=m.DeepGraphEmulator([128,128],[40],5,jnp.asarray(r),jnp.asarray(s),len(proc.layers),[3],proc.layers==0)
p=read_weights(R/'emulationResults/trainedParameters/lvData/trainedNetworkParams.pkl')
fn=jax.jit(lambda v,e,t,z:net.apply(p,v,e,t,z))
mean=np.load(fixed/'normalisationStatistics/real-node-displacement-mean.npy').astype('float32');std=np.load(fixed/'normalisationStatistics/real-node-displacement-std.npy').astype('float32')
# Exact preprocessing agreement with included fixed geometry processed data.
c={k:np.load(fixed/f'rawData/test/{k}.npy')[0].astype('float32') for k in ['real-node-coords','real-node-features','global-features','shape-coeffs','real-node-displacement']}
v,e,t,z=proc(c)
check['preprocessing_max_abs']={name:float(np.max(np.abs(a-np.load(fixed/f'processedData/test/{name}.npy').astype('float32')[0]))) for name,a in [('augmented-node-features',v),('edge-features',e),('global-features',t[0]),('shape-coeffs',proc.norm(c['shape-coeffs'][None],'shape-coeffs')[0])]}
assert max(check['preprocessing_max_abs'].values())<1e-4,check
rows=[];times=[]
for i in range(args.limit or len(d)):
 c=d[i];inputs=proc(c);start=time.perf_counter();pred=np.asarray(fn(*map(jnp.asarray,inputs)))*std+mean;times.append(time.perf_counter()-start)
 diff=pred-c['real-node-displacement'];e=np.linalg.norm(diff,axis=-1);rows.append({'case':f'test:{i}','mean_nodal_l2_cm':float(e.mean()),'component_rmse_cm':float(np.sqrt(np.mean(diff**2))),'p95_nodal_l2_cm':float(np.quantile(e,.95)),'max_nodal_l2_cm':float(e.max())})
 if i==0:np.savez_compressed(P/'experiments/metrics/reference_varying_case0.npz',prediction=pred,truth=c['real-node-displacement'],coords=c['real-node-coords'])
 if i%50==0:print(i,rows[-1],flush=True)
check.update(n_cases=len(rows),n_nodes=6784,metrics_mean_over_cases={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='case'},per_case=rows,warm_forward_ms_median=float(np.median(times[1:])*1000),first_call_seconds=times[0])
(P/'experiments/metrics/reference_varying.json').write_text(json.dumps(check,indent=2));print({k:v for k,v in check.items() if k!='per_case'})
