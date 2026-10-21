"""Freeze the bounded numerical-validation outcome without changing models or criteria."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--observed-no-active-jobs',action='store_true',required=True);args=ap.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
evidence={}
def read(name):
 p=P/f'experiments/metrics/{name}.json';evidence[str(p.relative_to(P))]=sha(p);return json.loads(p.read_text())
controls=[]
for label,name in [
 ('Baseline mesh\nQ6; dt 0.625 ms','dynamic_quadratic_controlled_ew_mesh'),
 ('Validation g004 mesh\nQ6; dt 1.25 ms','dynamic_quadratic_validation_g004_l0'),
 ('Baseline timestep\nQ6; 1.25 to 1 ms','dynamic_quadratic_controlled_time'),
 ('Baseline Q6 to Q8\ndt 0.625 ms','dynamic_quadratic_controlled_halfstep_quadrature'),
 ('Baseline Q8 to Q10\ndt 1.25 ms','dynamic_quadratic_controlled_quadrature10')]:
 d=read(name);assert d['complete'];assert d['passes_declared_one_percent_sensitivity']==(d['relative_space_time_L2']<=.01)
 controls.append(dict(label=label,evidence=name,relative_space_time_L2=d['relative_space_time_L2'],within_one_percent=d['passes_declared_one_percent_sensitivity']))
reference_checks=[]
for tag in ['coarse','refined']:
 name=f'cardiac_quadratic_reference_halfstep_ew_{tag}_v1';v=read(name+'_verification');assert v['passed']
 root=P/'data/processed'/name;m=json.loads((root/'manifest.json').read_text());assert m['complete'] and v['output_sha256']==m['output_sha256']
 assert all(sha(root/n)==digest for n,digest in v['output_sha256'].items())
 reference_checks.append(dict(reference=name,independent_verification_passed=True,output_hashes_rechecked=True))
failure=read('cardiac_quadratic_reference_halfstep_ew_third_v1_failure')
trust=read('dynamic_quadratic_third_trust_region_recovery');initial=read('dynamic_quadratic_third_prolongation_recovery');half=read('dynamic_quadratic_third_halfstep_diagnostic')
assert all(d['finished'] and not d['all_requested_steps_accepted'] for d in [trust,initial,half])
assert half['budget_exhausted'] and not half['records']
assert not initial['records'] and initial['initial_guess_check']['prolonged_residual_N'] is None
assert failure['SNES_reason']<0 and failure['nonlinear_iterations']==100
prefix=read('cardiac_quadratic_reference_halfstep_ew_third_v1_accepted_tail_verification_resolved');assert prefix['passed'] and not prefix['full_cycle']
fp=P/'experiments/manifests/dynamic_final_freeze.json';f=json.loads(fp.read_text());assert all(sha(P/n)==h for n,h in f['checkpoint_sha256'].items());evidence[str(fp.relative_to(P))]=sha(fp)
result=dict(assessment_complete=True,study_validation_passed=False,current_reference_protocol_status='failed',
 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_no_active_jobs=args.observed_no_active_jobs,
 numerical_controls=controls,completed_reference_checks=reference_checks,
 third_mesh=dict(full_cycle_complete=False,native_accepted_steps=failure['accepted_steps_before_failure'],durable_checkpoint_steps=failure['persisted_checkpoint_steps'],planned_steps=1600,failed_time_s=failure['failed_time_s'],SNES_reason=failure['SNES_reason'],failed_force_N=failure['residual_l2_N'],qualified_recovered_tail_steps=len(prefix['records']),
 trust_region_accepted=False,prolongation_initialization_admitted=False,halfstep_diagnostic_status='budget_exhausted_without_accepted_state',halfstep_last_unaccepted_force_N=half['last_iteration_history'][-1]['residual_N']),
 neural_checkpoints=dict(count=len(f['checkpoint_sha256']),all_frozen_hashes_unchanged=True,retrained_after_numerical_failures=False),
 blocked_work=['Demonstrate spatial convergence at the original1% tolerance with complete qualified references; current baseline difference11.8666% and validation-geometry difference12.3758% fail.',
 'Qualify consistent integration and timestep settings across intended geometries and loading trajectories; isolated passing controls do not resolve Q6 sensitivity.',
 'Fix a separate model/cohort protocol before retraining; independent qualified finer-reference calibration cases and untouched final comparison are not available.',
 'Run accuracy, mechanics, recursive stability, coverage, OOD/FE routing and matched runtime against those qualified references; original-discretization results remain separately valid within their scope.'],
 next_protocol_requirements=['A separately specified reference-solver/discretization investigation with native residual, tangent, positivity and time-history checks before another multi-day cohort run.',
 'Computational resources and runtime feasibility established for that protocol. More hardware alone has not been shown to resolve the observed nonlinear failure.',
 'Preserve inspected test roles, negative physics findings, passive mechanics and idealized-geometry limitations; no continuum, active-heart or clinical validation claim.'],
 evidence_sha256=evidence,executed_source_sha256=sha(Path(__file__)),
 scope='Closure of the currently fixed bounded numerical-validation protocol with a FAILED study-validation gate. This does not claim that every possible solver is impossible or that the overall research objective has been achieved. Original coarse-reference neural experiments remain completed; qualified-reference follow-up is blocked. No model selection, training, calibration, publication or push performed.')
(P/'experiments/metrics/dynamic_quadratic_validation_assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False))
os.environ.setdefault('MPLCONFIGDIR',str(P/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,6))
y=list(range(len(controls)));values=[100*c['relative_space_time_L2'] for c in controls]
ax.barh(y,values,color=['#298557' if c['within_one_percent'] else '#b84840' for c in controls],height=.64)
ax.set_yticks(y,[c['label'] for c in controls],fontsize=9);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(.01,40)
ax.axvline(1,color='black',linestyle='--',label='Declared 1% sensitivity tolerance')
for i,value in enumerate(values):ax.text(value*1.12,i,f'{value:.4g}%',va='center',fontsize=10)
ax.set_xlabel('Measured two-level space-time L2 difference (%)')
ax.set_title('Numerical validation remains FAILED\nIndividual passing controls do not qualify the full reference protocol')
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='lower right',fontsize=9)
fig.text(.5,.045,'Third mesh: 436 native-accepted / 1600 states; nonlinear failure at 0.273125 s.',ha='center',fontsize=10)
fig.text(.5,.015,'Different controls have the explicitly labeled settings. Differences are not continuum-error bounds.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,1));fig.savefig(P/'reports/figures/dynamic_quadratic_validation_assessment.png',dpi=180);plt.close(fig)
print('CURRENT PROTOCOL ASSESSMENT COMPLETE; STUDY VALIDATION PASSED: FALSE',flush=True)
