"""Evidence-only manuscript draft; update marked blocks and preserve author edits."""
import argparse,hashlib,json,re,sys
from pathlib import Path
from registry import read,atomic
TITLE='Physics-Informed Neural Surrogates for Rapid and Uncertainty-Aware Prediction of Left Ventricular Deformation'
TEMPLATE='''# '''+TITLE+'''

**Draft status:** computational evidence in progress; not submission-ready.

Authors and affiliations: [AUTHOR INPUT REQUIRED]

## Abstract
<!-- AUTO:abstract:BEGIN -->
<!-- AUTO:abstract:END -->

## 1. Introduction
Rapid prediction of passive left-ventricular deformation requires validation of both displacement accuracy and governing mechanics. This study separates external GNN emulation from mechanically complete finite-element experiments. Geometry-level uncertainty calibration and rejection under distribution shift are evaluated separately from raw ensemble variance. Hypotheses concern mechanical consistency at comparable displacement error, coverage on exchangeable held-out geometries, and matched end-to-end runtime. No hypothesis is treated as established before its declared validation gate passes.

## 2. Materials and methods
### 2.1 Data, provenance and independent experimental units
Track A uses the Dalton synthetic passive-diastolic LV dataset and released implementation [1,2]. Track B uses the cardiac benchmark mechanics [3] with complete volumetric elements, quadrature, constitutive data, loads and boundary conditions. The tracks are not pooled as one physical distribution. Each geometry is an independent unit; time frames and nodes are not independent samples. Dataset licenses, source revisions and checksums are retained in the run artifacts.

### 2.2 Governing mechanics and finite-element references
The implementation evaluates finite deformation F = I + Grad(u), fiber-dependent passive hyperelastic stress, follower pressure, inertia, viscosity and the benchmark's natural boundary terms. It does not replace the dynamic residual with static force balance. Native P2 fields and geometry-specific maps are used. Reference admission requires the declared force tolerance, a genuine positive native nonlinear convergence reason, finite values and whole-element positivity at endpoint and stress stage. Bernstein/subdivision tests are float64 sufficient conditions, not interval-arithmetic or continuum-truth proofs.

### 2.3 Numerical and cross-platform qualification
Mesh, timestep and quadrature sensitivity are controlled separately with an unchanged 1% threshold. A completed solve is not a convergence-study pass. Imported laptop fields and histories retain their original provenance. Any platform-native restart representation is separately declared, preserves original checkpoints and must pass independent force/history/hash/norm and positivity checks. Known-field reproduction precedes cluster continuation.

### 2.4 Neural surrogates and matched-budget comparisons
The preregistered qualified follow-up uses a geometry graph encoder, recurrent coefficient predictor and training-only POD preprocessing. Data-only and physics-guided arms use paired budgets and five seeds (11,22,33,44,55), full and limited-data regimes, predicted-history rollouts and validation-only selection. Actual full-P2 network-gradient and canonical-transfer checks are mandatory before qualified physics training. Earlier inspected tests are excluded from new selection.

### 2.5 Uncertainty, shift, rejection and FE fallback
Independent geometry-level calibration targets 90% and 95% coverage with finite-sample order statistics. Raw ensemble variance is not described as a calibrated interval. Coverage under exchangeability is distinguished from distribution shift. Rejected cases require a real qualified FE fallback; unqualified fallback solves do not count as successful routing.

### 2.6 Evaluation and statistical analysis
Endpoints include volume-weighted displacement error, strain error, native force, boundary violations, minimum deformation determinant, full-cycle rollout stability, interval size/coverage, OOD rejection and matched runtime. Geometry-level paired analyses and clustered confidence intervals are required. Training and reference-generation costs are reported separately from inference. No p-values or scientific scores are synthesized from job completion.

## 3. Results
<!-- AUTO:results:BEGIN -->
<!-- AUTO:results:END -->

## 4. Discussion
[AUTHOR REVIEW REQUIRED] Interpret supported and unsupported hypotheses using the registered results, including negative findings. Distinguish numerical sensitivity from continuum truth and local solver recovery from full-cycle validation. Do not generalize passive idealized geometries to patient-specific or active electromechanical prediction.

## 5. Limitations
The current scope is passive, idealized geometry and computational validation. Existing mesh and quadrature failures remain findings, not omitted controls. Cross-platform numerical qualification, complete reference validation, independent calibration and untouched final evaluation are separate requirements. Runtime ratios from unmatched hardware or queued elapsed time are not surrogate speedups.

## 6. Conclusion
<!-- AUTO:conclusion:BEGIN -->
<!-- AUTO:conclusion:END -->

## Reproducibility and computational resources
<!-- AUTO:reproducibility:BEGIN -->
<!-- AUTO:reproducibility:END -->

## Data and code availability
Code: https://github.com/Tumo505/Physics-informed-neural-surrogates . Large data and checkpoints are transferred separately and checked against the registered manifests. The exact cluster environment and executed commands are recorded in the run ledger. This manuscript is excluded from automatic GitHub publication.

## Ethics, funding, competing interests and author contributions
[AUTHOR INPUT REQUIRED] No patient-specific clinical validation or clinical benefit is claimed. Confirm dataset terms, acknowledgments, funding, author roles and competing-interest declarations before submission.

## References
1. Dalton LV dataset. Zenodo, DOI: 10.5281/zenodo.7075055. https://zenodo.org/records/7075055
2. Released passive LV GNN implementation. https://github.com/dodaltuin/passive-lv-gnn-emul
3. Cardiac benchmark implementation and documentation. https://github.com/finsberg/cardiac_benchmark

[Full bibliographic literature review and conference-format verification remain required before submission.]
'''
def block(text,name,value):
 pattern=r'(<!-- AUTO:'+name+r':BEGIN -->).*?(<!-- AUTO:'+name+r':END -->)'
 assert len(re.findall(pattern,text,re.S))==1,'Missing or duplicate manuscript marker: '+name
 return re.sub(pattern,lambda m:m[1]+'\n'+value+'\n'+m[2],text,flags=re.S)
def render(root,run):
 root=Path(root);run=Path(run);events=read(run);statepath=run/'state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {};config=json.loads((run/'plan.json').read_text()) if (run/'plan.json').exists() else {};status=state.get('status','not started');phase=state.get('phase','not started')
 path=root/'reports/manuscript.md';path.parent.mkdir(exist_ok=True);text=path.read_text() if path.exists() else TEMPLATE
 abstracts=f'**Provisional evidence status:** {status}; current stage: {phase}. This is an automatically maintained computational-study draft. Full-cycle reference completion, numerical convergence, qualified neural training, independent calibration and untouched final comparison are distinct gates. No final hypothesis, clinical conclusion or validated speedup is inferred from this status.'
 text=block(text,'abstract',abstracts)
 lines=['### 3.1 Registered cluster execution',f'Run: `{run.relative_to(root)}`. Status: **{status}**. Stage: `{phase}`.','', '| Event | PBS job | Stage/status | UTC |','|---|---|---|---|']
 jobs=[x for x in events if x['kind'] in ['job_started','stage_finished','stage_blocked','job_finished','job_paused','stage_failed']]
 for x in jobs[-40:]:lines.append(f"| {x['kind']} | {x.get('pbs_job_id',x.get('PBS_JOBID','—'))} | {x.get('phase',x.get('status','—'))} | {x['time_utc']} |")
 progress=[x for x in events if x['kind']=='reference_progress']
 if progress:
  p=progress[-1];lines+=['',f"Reference online admission: **{p['completed_steps']}/{p['n_steps']}** states through **{p['time_s']:.6f} s**. Latest native residual: **{p['residual_N']:.6g} N**. Final independent full-cycle validation remains separate."]
 coverage=root/'hpc/study_stages.json'
 if coverage.exists():
  lines+=['','**Execution coverage:**','| Required stage | State |','|---|---|']
  for stage in json.loads(coverage.read_text())['stages']:lines.append(f"| {stage['stage']} | {stage['status']} |")
 lines+=['','### 3.2 Scientific gates and quantitative evidence','| Artifact | Role | SHA-256 |','|---|---|---|']
 artifacts=[x for x in events if x['kind']=='artifact']
 for x in artifacts[-45:]:lines.append(f"| `{x['path']}` | {x['role']} | `{x['sha256']}` |")
 if state.get('blockers'):lines+=['','**Unresolved requirements:**']+['- '+x for x in state['blockers']]
 lines+=['','### 3.3 Historical evidence imported from the laptop','Historical local results are not described as new cluster reproductions. The imported failed 1% controls were approximately 11.8666% baseline mesh, 12.3758% validation mesh and 1.5715% Q6 integration. Those failures remain unresolved unless replaced by separately registered passing controls. Original neural checkpoints remain frozen.','', '### 3.4 Accuracy, mechanics, coverage, shift and matched inference runtime','Qualified final tables remain pending until their reference/model/calibration/test freezes pass. Missing metrics are not filled with historical test scores or invented estimates.']
 # Numerical results are populated from registered JSON, with scope labels preserved.
 lines+=['','### 3.5 Cluster quantitative measurements','| Artifact | Recorded quantity | Value |','|---|---|---|']
 for x in artifacts:
  f=root/x['path']
  if f.suffix!='.json' or not f.exists():continue
  try:d=json.loads(f.read_text())
  except (ValueError,OSError):continue
  for k in ['accepted_steps','checked_steps','relative_space_time_L2','passes_declared_one_percent_sensitivity','wall_seconds','ready_to_train']:
   if k in d:lines.append(f"| `{x['path']}` | `{k}` | {d[k]} |")
  rs=d.get('records',[])
  if rs and all(isinstance(z,dict) for z in rs):
   forces=[z.get('native_residual_N',z.get('native_force_N')) for z in rs];forces=[v for v in forces if isinstance(v,(int,float))]
   if forces:lines.append(f"| `{x['path']}` | Maximum independently recorded native force (N) | {max(forces):.6g} |")
 hist=root/'experiments/metrics/dynamic_final_evaluation.json'
 if hist.exists():
  d=json.loads(hist.read_text());lines+=['','### 3.6 Historical coarse-reference comparison — not new cluster/finer-reference validation','These previously inspected outcomes are disclosed as historical evidence only. They do not select new models or satisfy finer-reference uncertainty requirements.','| Historical model | Test geometries | Component RMSE (m) | Sampled normalized force RMS | 95% geometry coverage |','|---|---:|---:|---:|---|']
  for label,m in d.get('models',{}).items():
   t=m.get('splits',{}).get('test',{});cov=[v for v in t.get('coverage',[]) if abs(v.get('nominal',0)-.95)<1e-8];coverage=f"{cov[0]['covered']}/{cov[0]['n']}" if cov else 'not available'
   if t:lines.append(f"| {label} | {t['n_geometries']} | {t['mean_component_rmse_m']:.6g} | {t['mean_sampled_normalized_force_rms']:.6g} | {coverage} |")
  lines+=['','Source: `experiments/metrics/dynamic_final_evaluation.json`. FE discretization uncertainty is excluded from those historical errors/intervals. Negative physics-loss and shifted-coverage findings must remain in the interpretation.']
 figures=['dynamic_regularized_extended_runtime.png','dynamic_cauchy_recovery_failure.png']
 if any((root/'reports/figures'/f).exists() for f in figures):
  lines+=['','### 3.7 Registered historical scientific figures']
  for f in figures:
   if (root/'reports/figures'/f).exists():lines+=['',f'![Historical computational evidence: {f}](figures/{f})']
 text=block(text,'results','\n'.join(lines))
 text=block(text,'conclusion','The validated study is **not complete**. '+('Execution is blocked by: '+'; '.join(state.get('blockers',[])) if state.get('blockers') else 'Interpretation awaits the remaining independent scientific gates.'))
 repro=[f"Source revision at run creation: `{config.get('source_commit','not registered')}`.",f"Registry: `{run.relative_to(root)}/events.jsonl` (append-only SHA-256 chain).",f"Numerical force admission remains **1e-8 N**; reported history checks retain their declared tolerance.", 'Single-node CPU/MPI FE execution is distinct from GPU neural training. Environment locks, native library versions, resource requests, allocated hosts, commands, outputs and failures are registered. Queue waiting is excluded from solver runtime.']
 jobrows=[x for x in events if x['kind']=='job_started']
 if jobrows:
  j=jobrows[-1];repro+=['CPU: '+str(j.get('cpu_model','not recorded'))+'. Allocated node(s): '+', '.join(j.get('allocated_hosts',[]))+'. Host RAM bytes: '+str(j.get('host_physical_RAM_bytes','not recorded'))+'.', 'Requested resources: '+str({k:config.get(k) for k in ['queue','cpus','ranks','memory_gb','job_hours']})+'.']
 text=block(text,'reproducibility','\n\n'.join(repro));newhash=hashlib.sha256(text.encode()).hexdigest()
 if not path.exists() or path.read_text()!=text:
  temp=path.with_suffix('.pending');temp.write_text(text);temp.replace(path)
 atomic(run/'manuscript-state.json',dict(path='reports/manuscript.md',sha256=newhash,ledger_tail=events[-1]['event_sha256'] if events else None,submission_ready=False))
 return path
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[1];print(render(root,root/a.run))
