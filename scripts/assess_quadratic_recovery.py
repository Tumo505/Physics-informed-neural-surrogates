"""Assess the completed bounded recovery separately from study validation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

P = Path(__file__).resolve().parents[1]
evidence = {}
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()
def read(relative):
    path = P/relative; evidence[relative] = sha(path)
    return json.loads(path.read_text())
watch = {'probe_quadratic_strict_stopping.py','run_bounded_quadratic_probe.py',
    'verify_quadratic_substep_probe.py','resolve_quadratic_positivity.py',
    'generate_quadratic_reference.py','run_quadratic_validation.py'}
active = []
for line in subprocess.check_output(['ps','-axo','pid=,command='], text=True).splitlines():
    pid, command = line.strip().split(None,1)
    words = command.split()
    if words and Path(words[0]).name.startswith('python') and any(Path(w).name in watch for w in words[1:]):
        active.append(dict(pid=int(pid),command=command))
assert not active, 'Do not close the bounded recovery while FE/audit workflows remain active'
stems = ['dynamic_quadratic_third_history_resume_v1',
    'dynamic_quadratic_third_stress_predictor_pilot',
    'dynamic_quadratic_third_stress_predictor_consecutive',
    'dynamic_quadratic_third_best_predictor_pilot']
states = []; direct_costs = []
for stem in stems:
    probe = read(f'experiments/metrics/{stem}.json')
    audit = read(f'experiments/metrics/{stem}_verification_resolved.json')
    assert probe['finished'] and audit['complete'] and audit['passed'] and not audit['full_cycle']
    protocol = read(f'configs/physics/{stem}.json')
    assert audit['protocol_sha256']==probe['protocol_sha256']==sha(P/f'configs/physics/{stem}.json')
    assert sha(P/protocol['fields'])==audit['fields_sha256']
    assert sha(P/protocol['checkpoint'])==probe['probe_checkpoint_sha256']
    assert sha(P/'data/processed'/protocol['reference']/'inputs.npz')==audit['inputs_sha256']
    for relative,digest in audit['audit_input_sha256'].items(): assert sha(P/relative)==digest
    assert max(audit['final_checkpoint_history_relative_errors'].values())<1e-7
    if stem.endswith('consecutive'):
        assert audit['accepted_prefix_only'] and not probe['all_requested_steps_accepted']
    else: assert probe['all_requested_steps_accepted']
    for row in audit['records']:
        native = next(r for r in probe['records'] if abs(r['time_s']-row['time_s'])<1e-12)
        assert native['accepted'] and native['SNES_reason']>0 and row['native_residual_N']<1e-8
        assert min(row['endpoint_minimum_Bernstein_bound'],row['stress_minimum_Bernstein_bound'])>1e-9
        assert row.get('independent_L2_norm_relative_error',0)<1e-11
        states.append(dict(step=native['step'],time_s=row['time_s'],
            native_residual_N=row['native_residual_N'],
            endpoint_lower_bound=row['endpoint_minimum_Bernstein_bound'],
            stress_lower_bound=row['stress_minimum_Bernstein_bound'],
            independently_verified=True,accepted_prefix_only=audit['accepted_prefix_only'],
            evidence=stem,nonlinear_seconds=native['nonlinear_seconds']))
        if stem!=stems[0]: direct_costs.append(native['nonlinear_seconds'])
assert [s['step'] for s in states]==[436,437,438,439]
controls = []
for stem in ['dynamic_quadratic_controlled_ew_mesh','dynamic_quadratic_validation_g004_l0',
             'dynamic_quadratic_controlled_halfstep_quadrature']:
    metric = read(f'experiments/metrics/{stem}.json')
    assert metric['complete'] and not metric['passes_declared_one_percent_sensitivity']
    controls.append(dict(evidence=stem,relative_space_time_L2=metric['relative_space_time_L2'],tolerance=.01,passed=False))
freeze = read('experiments/manifests/dynamic_final_freeze.json')
for relative,digest in freeze['checkpoint_sha256'].items(): assert sha(P/relative)==digest
partial = read('experiments/metrics/dynamic_quadratic_third_stress_predictor_consecutive.json')
assert not partial['all_requested_steps_accepted'] and len(partial['records'])==1
remaining = 1600-(states[-1]['step']+1)
mean = float(np.mean(direct_costs)); hours = remaining*mean/3600
result = dict(complete=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    bounded_local_recovery_passed=True,study_validation_passed=False,
    full_cycle_qualified=False,numerical_convergence_passed=False,runtime_feasibility_established=False,
    observed_active_FE_or_audit_jobs=active,verified_local_states=states,
    original_reference_manifest_unchanged=True,
    partial_pilot_remains_incomplete=True,failed_numerical_controls=controls,
    neural_checkpoints=dict(count=len(freeze['checkpoint_sha256']),all_frozen_hashes_unchanged=True),
    runtime=dict(measured_direct_loaded_step_seconds=direct_costs,mean_seconds=mean,
        remaining_steps=remaining,flat_cost_scenario_hours=hours,per_reference_budget_hours=72,
        required_mean_seconds_to_fit_remaining_steps_in_72h=72*3600/remaining,
        idealized_throughput_improvement_vs_measured_mean=hours/72,
        scope='Flat-rate sensitivity scenario, not ETA. Excludes setup, predictor screening, audits and already expended prefix/recovery cost; later-step costs are unknown.'),
    remaining_requirements=[
        'Profile and independently qualify materially faster FE throughput before a full-cycle R2 restart; repeated isolated-step solves do not establish the72h budget.',
        'Complete the third-mesh cycle and audit its entire prefix as well as new states; these four local proofs do not qualify the original whole prefix.',
        'Pass full-cycle spatial, timestep and quadrature sensitivity gates at unchanged1% across baseline and validation geometry/load designs; report rates only if assumptions hold.',
        'Verify actual P2 neural-network directional gradients, then execute the separately fixed training/validation-only follow-up with qualified labels.',
        'Independently recalibrate finer-reference uncertainty and perform untouched frozen accuracy/mechanics/rollout/coverage/OOD/FE-routing/runtime comparison.'
    ],evidence_sha256=evidence,source_sha256=sha(Path(__file__)),
    monitor_action='Bounded pilot and justified audits finished with no active jobs; stop this recurring monitor.',
    scope='Local nonlinear failures repaired through0.275s on the idealized passive R2 discrete problem. No full-cycle, continuum, active-heart, clinical, population coverage or completed-study validation claim. Negative results and initial failed audits retained; no model training, test selection, compute purchase, push or publication.')
out=P/'experiments/metrics/dynamic_quadratic_recovery_assessment.json'
assert not out.exists(), 'Preserve an existing assessment; use a new version for changed evidence'
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
os.environ.setdefault('MPLCONFIGDIR',str(P/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,4.4),constrained_layout=True)
labels=[f'{s["time_s"]:.6f}' for s in states];x=np.arange(4)
axes[0].scatter(x,[s['native_residual_N'] for s in states],color='#22806b',s=40)
axes[0].axhline(1e-8,color='#ad493b',ls='--',label='Strict force threshold')
axes[0].set(yscale='log',title='Four verified local states',ylabel='Native force norm (N)')
axes[0].set_xticks(x,labels,rotation=45);axes[0].legend(fontsize=8)
axes[1].bar(x-.18,[s['endpoint_lower_bound'] for s in states],width=.36,label='Endpoint')
axes[1].bar(x+.18,[s['stress_lower_bound'] for s in states],width=.36,label='Stress stage')
axes[1].set(yscale='log',title='Whole-element positivity',ylabel='Certified float64 detF lower bound')
axes[1].set_xticks(x,labels,rotation=45);axes[1].legend(fontsize=8)
axes[2].bar(np.arange(3),np.array(direct_costs)/60,color='#426983')
axes[2].set_xticks(np.arange(3),labels[1:],rotation=45)
axes[2].set(title='Loaded Newton/GAMG cost',ylabel='Native nonlinear solve (min)')
for i,cost in enumerate(direct_costs):axes[2].text(i,cost/60+.3,f'{cost/60:.1f}',ha='center',fontsize=9)
axes[2].set_ylim(0,max(direct_costs)/60*1.18)
for ax in axes:
    ax.set_xlabel('Physical endpoint time (s)');ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Bounded R2 recovery passed locally; full study validation remains incomplete',fontsize=12)
fig.text(.5,-.06,f'Four local state audits do not qualify the full-cycle reference. Flat remaining-cost scenario: {hours:.1f} h vs72 h budget; not an ETA.\nSolve bars exclude setup, predictor screening and audits; algebraic-homotopy recovery cost is excluded.',ha='center',fontsize=8)
fig.savefig(P/'reports/figures/dynamic_quadratic_recovery_assessment.png',dpi=180,bbox_inches='tight')
print(json.dumps(dict(local_recovery_passed=True,study_validation_passed=False,runtime=result['runtime']),indent=2))
