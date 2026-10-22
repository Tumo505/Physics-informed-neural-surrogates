"""Sequential known-coarse/known-loaded qualification gates before a failed-step probe."""
import fcntl,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];plan_path=P/'configs/physics/dynamic_quadratic_mpi_admissible_study_v1.json';plan=json.loads(plan_path.read_text())
lock=(P/'experiments/logs/dynamic_quadratic_mpi_admissible_study_v1.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=P/'experiments/metrics/dynamic_quadratic_mpi_admissible_study_v1.json';assert not out.exists(),'Never duplicate or retry a study output'
state=dict(complete=False,study_validation_passed=False,phases=[],scope=plan['scope']);tick=time.perf_counter()
def persist():
 state['wall_seconds']=time.perf_counter()-tick;tmp=out.with_suffix('.pending.json');tmp.write_text(json.dumps(state,indent=2));tmp.replace(out)
persist()
try:
 for i,protocol in enumerate(plan['protocols']):
  state['active_protocol']=protocol;persist();p=json.loads((P/protocol).read_text());stem=Path(p['output']).stem
  assert not (P/p['output']).exists() and not (P/f'experiments/metrics/{stem}_workflow.json').exists()
  with (P/f'experiments/logs/{stem}_runner.log').open('w') as log:
   result=subprocess.run([str(P/'.venv/bin/python'),'scripts/run_bounded_admissible_newton.py','--protocol',protocol,'--max-phase-seconds',str(plan['phase_budgets_seconds'][i]),'--max-workflow-seconds',str(plan['workflow_budgets_seconds'][i])],cwd=P,stdout=log,stderr=subprocess.STDOUT)
  w=json.loads((P/f'experiments/metrics/{stem}_workflow.json').read_text());passed=bool(result.returncode==0 and w.get('local_recovery_verified'))
  state['phases'].append(dict(protocol=protocol,independent_local_audit_passed=passed));persist()
  if not passed:raise RuntimeError('Qualification/recovery gate failed: '+protocol)
 state['all_local_gates_passed']=True
except Exception as exc:state['failure']=str(exc)
finally:
 state['complete']=True;state['active_protocol']='finished';persist();subprocess.run([str(P/'.venv/bin/python'),'scripts/write_reports.py'],cwd=P)
