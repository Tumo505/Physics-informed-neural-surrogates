"""Preserve terminal failure provenance and plot the bounded recovery trace."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
protocol=P/'configs/physics/dynamic_quadratic_cauchy_failed_step_v1.json';p=json.loads(protocol.read_text());out=P/p['output'];r=json.loads(out.read_text());assert r['finished'] and not r['all_requested_steps_accepted'] and not r['records']
a=r['attempts'][-1];assert not a['accepted'] and a['SNES_reason']==0
steps=r['admissible_solves'][-1]['custom_newton_steps'];accepted=[]
for s in steps:
 trials=[z for z in s['trials'] if z['armijo_passed'] and z['whole_element_admissible']]
 assert len(trials)==1
 accepted.append(trials[-1]['native_force_N'])
cp=P/p['iterate_checkpoint'];checkpoint=np.load(cp);assert not bool(checkpoint['accepted'])
parent=json.loads((P/p['resume_probe_protocol']).read_text());seed=np.load(P/parent['checkpoint'])
histories={k:bool(np.array_equal(checkpoint[k],seed[k])) for k in ['u_old','v_old','a_old']};assert all(histories.values())
for path,h in p['frozen_source_sha256'].items():assert sha(P/path)==h
wpath=P/'experiments/metrics/dynamic_quadratic_cauchy_failed_step_v1_workflow.json';w=json.loads(wpath.read_text());assert w['complete'] and not w['local_recovery_verified']
result=dict(complete=True,recovery_passed=False,study_validation_passed=False,physical_chain_end_s=.281875,physical_chain_endpoints=451,total_required_endpoints=1600,failed_target_s=.2825,custom_iterations_accepted=len(steps),last_accepted_custom_force_N=accepted[-1],last_accepted_force_to_final_tolerance_ratio=accepted[-1]/1e-8,Cauchy_steps=sum(s['direction_used']=='Cauchy' for s in steps),Newton_steps=sum(s['direction_used']=='Newton' for s in steps),native_wall_seconds=r['wall_seconds'],workflow_wall_seconds=w['total_wall_seconds'],sampled_peak_RSS_bytes=max(z['sampled_peak_RSS_bytes'] for z in w['phases']),terminal_attempt=a,terminal_displacement_coefficients_finite=bool(np.isfinite(checkpoint['u']).all()),unchanged_physical_histories=histories,terminal_checkpoint_sha256=sha(cp),protocol_sha256=sha(protocol),output_sha256=sha(out),workflow_sha256=sha(wpath),frozen_sources_unchanged=True,scope='Last accepted custom iterate force is NOT the terminal trial force. Budget interruption left an unaccepted trial checkpoint with finite displacement but no finite recorded terminal force. No fresh SNES convergence,no physical endpoint,no final norm/positivity qualification claimed. Accepted custom trials had online proofs only. Original failed outputs and independently qualified seven-state prefix retained. Full-cycle convergence,1% mesh/time/integration qualification across geometries/loadings,fullP2 gradient/transfer,new train/validation-selected models,independent calibration,and untouched final comparison remain unpassed. More wall time alone is unsupported; a separately fixed and qualified nonlinear method is needed before further full-reference work. No root existence,condition-number or physical-instability inference.')
path=P/'experiments/metrics/dynamic_quadratic_cauchy_failure_assessment.json'
if path.exists():assert json.loads(path.read_text())==result
else:path.write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');x=np.arange(1,len(steps)+1);y=np.array(accepted)
for ax in axes:
 ax.plot(x,y,color='#374151',lw=1)
 for d,c in [('Newton','#2563eb'),('Cauchy','#dc2626')]:
  mask=np.array([s['direction_used']==d for s in steps]);ax.scatter(x[mask],y[mask],s=15,color=c,label=d)
 ax.set_xlabel('Accepted custom iteration');ax.set_ylabel('Native force residual (N)');ax.grid(alpha=.2)
axes[0].set_yscale('log');axes[0].axhline(1e-8,ls='--',color='black',label='Final physical tolerance');axes[0].set_title('No physical endpoint accepted');axes[0].legend(fontsize=8)
axes[1].set_xlim(10,len(steps)+1);axes[1].set_ylim(.03,.041);axes[1].set_title('Slow residual reduction after iteration 10')
fig.suptitle('Bounded Newton–Cauchy recovery failed at 0.2825 s')
fig.savefig(P/'reports/figures/dynamic_cauchy_recovery_failure.png',dpi=180);plt.close(fig)
print(json.dumps({k:result[k] for k in ['recovery_passed','custom_iterations_accepted','Cauchy_steps','last_accepted_custom_force_N','unchanged_physical_histories']}))
