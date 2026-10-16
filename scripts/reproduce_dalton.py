from pathlib import Path
import sys,time,json,hashlib,platform,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from lv_surrogate.models.reference import read_weights,fixed_predict
P=Path(__file__).resolve().parents[1]; R=P/'external/passive-lv-gnn-emul'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 b=R/'data/lvDataFixedGeom'; d=b/'processedData/test';w=R/'emulationResults/trainedParameters/lvDataFixedGeom'
 enc=read_weights(w/'theta_encode_params.pkl');dec=read_weights(w/'trainedNetworkParams.pkl');lat=np.load(w/'latent_nodal_values.npy').astype('float32')
 theta=np.load(d/'global-features.npy').astype('float32');truth=np.load(d/'real-node-displacement.npy',mmap_mode='r')
 mean=np.load(b/'normalisationStatistics/real-node-displacement-mean.npy').astype('float32');std=np.load(b/'normalisationStatistics/real-node-displacement-std.npy').astype('float32')
 n=a.limit or len(theta);records=[];preds=[];times=[]
 for i in range(n):
  start=time.perf_counter();pred=fixed_predict(theta[i:i+1],enc,dec,lat)*std+mean;times.append(time.perf_counter()-start)
  diff=pred-truth[i];e=np.linalg.norm(diff,axis=-1)
  records.append({'case':f'test:{i}','mean_nodal_l2_cm':float(e.mean()),'component_rmse_cm':float(np.sqrt(np.mean(diff**2))),'p95_nodal_l2_cm':float(np.quantile(e,.95)),'max_nodal_l2_cm':float(e.max())})
  if i==0:preds=pred
  if i%50==0:print(i,records[-1],flush=True)
 out={'task':'released fixed-geometry LV decoder; held-out parameter cases, ONE geometry','backend':'NumPy compatibility forward; original legacy CLI pending','n_cases':n,'n_independent_geometries':1,'n_nodes':len(lat),'metrics_mean_over_cases':{k:float(np.mean([r[k] for r in records])) for k in records[0] if k!='case'},'warm_latency_ms_median':float(np.median(times[1:])*1000) if n>1 else None,'per_case':records}
 (P/'experiments/metrics/reference_fixed.json').write_text(json.dumps(out,indent=2))
 np.savez_compressed(P/'experiments/metrics/reference_fixed_case0.npz',prediction=preds,truth=truth[0],coords=np.load(d/'real-node-coords.npy')[0])
 hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in w.glob('*')}
 (P/'experiments/manifests/reference_fixed.json').write_text(json.dumps({'command':sys.argv,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'weight_hashes':hashes,'units':'cm','source_commit':'449cc22459ba5ada344fe0ff238e6b586d5e9319'},indent=2))
 print(json.dumps({k:v for k,v in out.items() if k!='per_case'},indent=2))
if __name__=='__main__':main()
