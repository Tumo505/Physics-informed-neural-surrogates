"""Recheck bounded robustness evidence and quantify, without promising future runtime."""
import hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(P/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
stem='dynamic_quadratic_regularized_extended_window_v1';pp=P/'configs/physics'/f'{stem}.json';p=json.loads(pp.read_text());rp=P/p['output'];r=json.loads(rp.read_text());ap=P/'experiments/metrics'/f'{stem}_verification_resolved.json';a=json.loads(ap.read_text())
assert a['passed'] and not a['accepted_prefix_only'] and r['all_requested_steps_accepted'] and len(r['records'])==len(a['records'])==4
for f,h in {**p['frozen_source_sha256'],**a['audit_input_sha256']}.items():assert sha(P/f)==h
assert sha(P/p['fields'])==a['fields_sha256'] and sha(P/p['checkpoint'])==r['probe_checkpoint_sha256']
for x,y in zip(r['records'],a['records']):assert x['accepted'] and x['SNES_reason']>0 and x['nonlinear_iterations']==0 and y['native_residual_N']<1e-8 and y['endpoint_unresolved_elements']==y['stress_unresolved_elements']==0
assert all(x==0 for x in a['final_checkpoint_history_relative_errors'].values())
times=[x['seconds'] for x in r['records']];remaining=1600-458;mean=sum(times)/4
result=dict(complete=True,local_four_step_robustness_passed=True,full_cycle_passed=False,numerical_convergence_passed=False,study_validation_passed=False,physical_chain_end_s=.28625,physical_chain_endpoints=458,required_endpoints=1600,remaining_endpoints=remaining,solve_seconds=times,mean_solve_seconds=mean,native_wall_seconds=r['wall_seconds'],independent_forces_N=[x['native_residual_N'] for x in a['records']],maximum_independent_norm_relative_error=max(x['independent_L2_norm_relative_error'] for x in a['records']),minimum_endpoint_bound=min(x['endpoint_minimum_Bernstein_bound'] for x in a['records']),minimum_stress_bound=min(x['stress_minimum_Bernstein_bound'] for x in a['records']),flat_remaining_solve_only_hours=remaining*mean/3600,reference_budget_hours=72,generous_solve_only_seconds_per_remaining_step_if_all_72h_available=72*3600/remaining,protocol_sha256=sha(pp),output_sha256=sha(rp),audit_sha256=sha(ap),checkpoint_sha256=sha(P/p['checkpoint']),scope='Localfournewstepvalidationonly. Flatfuturecostis asensitivityscenario,notETA/uniformcostevidence. Evenallocatingall72hto remainingstepsignoringpriorcost/setup/finalaudits doesnotmeetmeasuredrate. No fullreference restartqualified,no blindbudgetextension. Needfasterqualifiedsolver/localresourcesor separatelydeclaredlargerresourceprotocol and full-cycle/numericalvalidation. Onefullreferencewouldnotalonepassnumericalmesh/time/integrationacrossgeometry/loading. 20neuralmodels frozen;newML/fullP2gradienttransfer/independentcalibration/untouchedcomparison gated. Passiveidealizedscope,negativecontrolsandoldtestseparationpreserved.')
out=P/'experiments/metrics/dynamic_quadratic_regularized_extended_assessment.json'
if out.exists():assert json.loads(out.read_text())==result
else:out.write_text(json.dumps(result,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.bar([f"{x['time_s']:.6f}" for x in r['records']],times,color='#2563eb');ax.axhline(72*3600/remaining,color='#b91c1c',ls='--',label='72 h / remaining steps (excludes audits)');ax.set_xlabel('Physical time (s)');ax.set_ylabel('Measured solve time (s)');ax.set_title('Four new states passed; runtime remains a limitation');ax.legend(fontsize=8);ax.set_ylim(0,430);fig.savefig(P/'reports/figures/dynamic_regularized_extended_runtime.png',dpi=180);plt.close(fig)
print('Four-step independent pass; flat remaining solve-only scenario',result['flat_remaining_solve_only_hours'],'hours, not ETA')
