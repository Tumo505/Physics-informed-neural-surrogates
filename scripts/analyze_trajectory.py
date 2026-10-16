"""Jacobian audit and matched-time temporal discretization sensitivity."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];base=np.load(P/'data/processed/cardiac/cohort.npz');X=base['coords'];C=base['cells'];dm=(X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1);inv=np.linalg.inv(dm);vol=abs(np.linalg.det(dm))/6
trajectory=P/'data/processed/cardiac_passive_trajectory_v1';records=[]
for path in sorted(trajectory.glob('step_*.npz')):
 with np.load(path) as a:
  u=a['u'];gradu=np.einsum('eij,ejk->eik',(u[C[:,1:]]-u[C[:,:1]]).transpose(0,2,1),inv);J=np.linalg.det(np.eye(3)+gradu);records.append({'time_s':float(a['time']),'pressure_Pa':float(a['pressure']),'minimum_detF':float(J.min()),'inverted_element_fraction':float((J<=0).mean()),'max_nodal_displacement_m':float(np.linalg.norm(u,axis=-1).max())})
comparison=[]
def L2(u):
 ue=u[C];return float(np.sqrt(np.sum(vol/20*(np.sum(ue**2,axis=(1,2))+np.sum(ue.sum(1)**2,axis=1)))))
for t in np.arange(.002,.0401,.002):
 states={}
 for name,dt in [('cardiac_dt002',.002),('cardiac_passive_trajectory_v1',.001),('cardiac_dt0005',.0005)]:
  path=P/f'data/processed/{name}/step_{round(t/dt)-1:04d}.npz'
  if not path.exists():raise RuntimeError(f'Missing matched time state: {path}')
  with np.load(path) as a:states[dt]=a['u'].copy()
 ref=L2(states[.0005]);comparison.append({'time_s':float(t),'dt002_relative_volume_L2_difference_vs_dt0005':L2(states[.002]-states[.0005])/ref,'dt001_relative_volume_L2_difference_vs_dt0005':L2(states[.001]-states[.0005])/ref})
metadata=json.loads((P/'experiments/metrics/cardiac_passive_trajectory_v1.json').read_text());result={'trajectory_complete':metadata['complete'],'audited_states':len(records),'minimum_detF':min(r['minimum_detF'] for r in records),'maximum_inverted_element_fraction':max(r['inverted_element_fraction'] for r in records),'trajectory':records,'time_sensitivity':comparison,'scope':'one fixed geometry, passive pressure waveform; timestep sensitivity evaluated only over initial40ms, not a full-cycle convergence proof'}
(P/'experiments/metrics/trajectory_validation.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');t=[r['time_s'] for r in records]
ax[0].plot(t,[1000*r['max_nodal_displacement_m'] for r in records]);ax[0].set(xlabel='Time (s)',ylabel='Maximum displacement (mm)',title='Passive FE trajectory')
ax[1].plot(t,[r['minimum_detF'] for r in records]);ax[1].axhline(0,color='r',ls='--');ax[1].set(xlabel='Time (s)',ylabel='Minimum element detF',title='All stored endpoint states')
for dt,key in [(.002,'dt002'),(.001,'dt001')]:ax[2].plot([r['time_s']*1000 for r in comparison],[100*r[key+'_relative_volume_L2_difference_vs_dt0005'] for r in comparison],label=f'dt={dt*1000:g} ms')
ax[2].set(xlabel='Time (ms)',ylabel='Relative volume-L2 difference (%)',title='Reference dt=0.5 ms');ax[2].legend()
fig.savefig(P/'reports/figures/trajectory_validation.png',dpi=180);print('TRAJECTORY AUDIT',result['trajectory_complete'],result['minimum_detF'],flush=True)
