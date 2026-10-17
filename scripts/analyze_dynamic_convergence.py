"""Whole-cycle mass-consistent temporal sensitivity at matched endpoint times."""
from pathlib import Path
import hashlib,json,numpy as np
P=Path(__file__).resolve().parents[1];root=P/'data/processed/cardiac_dynamic_v1';a=dict(np.load(root/'convergence_dt0.005/trajectory.npz'));b=dict(np.load(root/'convergence_dt0.0025/trajectory.npz'));geo=np.load(root/'convergence_dt0.005/geometry.npz');X=geo['coords'];C=geo['cells'];vol=abs(np.linalg.det((X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1)))/6
# Keep the archived999 frames immutable; exact native continuation supplies1s.
endpoint=np.load(P/'data/processed/cardiac_dynamic_endpoint_1s/endpoint.npz')
endpoint_metric=json.loads((P/'experiments/metrics/dynamic_time_endpoint_1s.json').read_text())
assert endpoint_metric['complete'] and float(endpoint['time'])==1.
endpoint_path=P/'data/processed/cardiac_dynamic_endpoint_1s/endpoint.npz'
assert hashlib.sha256(endpoint_path.read_bytes()).hexdigest()==endpoint_metric['endpoint_sha256']
invDm=np.linalg.inv((X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1));stress=.4*endpoint['u_old']+.6*endpoint['u'];stressJ=np.linalg.det(np.eye(3)+np.einsum('eij,ejk->eik',(stress[C[:,1:]]-stress[C[:,:1]]).transpose(0,2,1),invDm));assert np.isfinite(stressJ).all() and stressJ.min()>0
coarse=a['u'];middle=b['u'][1::2];fine=np.concatenate((np.stack([np.load(P/f'data/processed/cardiac_passive_trajectory_v1/step_{int(round(t/.001))-1:04d}.npz')['u'] for t in a['time'][:-1]]),endpoint['u'][None]))
np.testing.assert_allclose(a['time'],b['time'][1::2],rtol=0,atol=1e-14)
def norm2(fields):
 values=[]
 for f in fields:
  nodal=f[C];values.append(float((vol/20*((nodal.sum(1)**2).sum(-1)+(nodal**2).sum((1,2)))).sum()))
 return np.array(values)
reference=norm2(fine);records=[]
for name,u in [('dt005',coarse),('dt0025',middle)]:
 diff=norm2(u-fine);records.append({'resolution':name,'relative_spacetime_volume_L2_error_vs_dt001':float(np.sqrt(diff.sum()/reference.sum())),'maximum_endpoint_relative_volume_L2_error':float(np.sqrt(diff/np.maximum(reference,1e-30)).max()),'endpoint_relative_volume_L2_errors':np.sqrt(diff/np.maximum(reference,1e-30)).tolist(),'volume_time_component_rmse_m':float(np.sqrt(diff.mean()/(3*vol.sum())))})
ratio=records[1]['relative_spacetime_volume_L2_error_vs_dt001']/records[0]['relative_spacetime_volume_L2_error_vs_dt001'];result={'complete':True,'n_common_endpoints':len(coarse),'time_range_s':[float(a['time'][0]),float(a['time'][-1])],'dt_s':[.005,.0025,.001],'fixed_spatial_mesh_tetrahedra':len(C),'mass_norm':'exact P1 consistent tetrahedral mass integral; space-time endpoint Riemann sum at common5ms grid','cases':records,'refining_time_reduces_error_vs_dt001':bool(ratio<1),'error_ratio_dt0025_over_dt005':ratio,'dt005_within_1percent_spacetime_tolerance':records[0]['relative_spacetime_volume_L2_error_vs_dt001']<=.01,'endpoint_1s_source':endpoint_metric,'endpoint_1s_independent_stress_minimum_detF':float(stressJ.min()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['finest1ms is a reference, not continuum truth','unequal timestep ratios; no asymptotic rate asserted','time sensitivity on one baseline waveform/geometry only']};(P/'experiments/metrics/dynamic_time_convergence.json').write_text(json.dumps(result,indent=2));print('FULL-CYCLE TIME COMPARISON COMPLETE',len(coarse),'endpoints',flush=True)
