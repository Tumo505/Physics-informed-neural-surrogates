"""Guarded staged qualification and longer local full-reference continuation; no blind retries."""
import argparse,fcntl,json,os,shutil,signal,subprocess,time,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.durable_reference import atomic_json,sha
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args();plan=json.loads((P/args.protocol).read_text());out=P/plan['output'];lock=(P/'experiments/logs'/f'{out.stem}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for f,h in plan['frozen_source_sha256'].items():assert sha(P/f)==h
prior=0.;old_active=0.;finished=[]
if out.exists():
 state=json.loads(out.read_text());assert args.resume and not state['complete'] and state['protocol_sha256']==sha(P/args.protocol)
 assert state['active_phase'] in ['coarse_first_step','coarse_restart_step','full_continuation'],'Only durable generation can resume; preserve incomplete audit/seed evidence'
 prior=state['total_wall_seconds'];old_active=state.get('active_phase_wall_seconds',0.);finished=[x['phase'] for x in state['phases'] if x['return_code']==0 and not x['stop_reason']]
 state.setdefault('resume_history',[]).append(dict(prior_wall_seconds=prior,active_phase=state['active_phase'],active_phase_seconds=old_active))
else:
 assert not args.resume;state=dict(complete=False,study_validation_passed=False,protocol=args.protocol,protocol_sha256=sha(P/args.protocol),phases=[],scope=plan['scope'])
tick=time.perf_counter();env=os.environ.copy();env.update(FI_PROVIDER='tcp',FI_TCP_IFACE='en0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAMBA_ROOT_PREFIX=str(P/'.mamba'),XDG_CACHE_HOME=str(P/'.cache'),DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'))
modern=[str(P/'.venv/bin/python')];native=[str(P/'.tools/bin/micromamba'),'run','-p',str(P/'.fenics'),'python'];mpi=native[:-1]+['mpiexec','-n','12','python']
def persist():
 state['total_wall_seconds']=prior+time.perf_counter()-tick;atomic_json(out,state)
def run(name,command,hours):
 global old_active
 if name in finished:return
 used=old_active if args.resume and name==state.get('active_phase') else 0.;old_active=0.
 assert used<hours*3600;state['active_phase']=name;start=time.perf_counter();peak=0;stop=None;persist()
 log=P/'experiments/logs'/f'{out.stem}_{name}.log'
 with log.open('a' if used else 'w') as f:
  child=subprocess.Popen(command,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while child.poll() is None:
    rows=[tuple(map(int,row.split())) for row in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True).splitlines() if row.strip()];tree={child.pid}
    while True:
     found={pid for pid,parent,rss in rows if parent in tree}
     if found<=tree:break
     tree.update(found)
    current=sum(rss*1024 for pid,parent,rss in rows if pid in tree);peak=max(peak,current)
    state['active_phase_wall_seconds']=used+time.perf_counter()-start;state['sampled_active_RSS_bytes']=current
    if state['active_phase_wall_seconds']>hours*3600:stop='declared_phase_budget'
    if prior+time.perf_counter()-tick>plan['cumulative_pipeline_hours']*3600:stop='declared_cumulative_pipeline_budget'
    if peak>30*2**30:stop='process_tree_RSS_30GiB'
    if shutil.disk_usage(P).free<100*2**30:stop='free_disk_reserve_100GiB'
    persist()
    if stop:break
    time.sleep(2)
  finally:
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  code=child.wait()
 state['phases'].append(dict(phase=name,return_code=code,stop_reason=stop,wall_seconds=used+time.perf_counter()-start,sampled_peak_RSS_bytes=peak));persist()
 if code or stop:raise RuntimeError(name+': '+str(stop or code))
 finished.append(name)
try:
 c=plan['coarse_stream_protocol'];full=plan['full_stream_protocol']
 run('coarse_seed',modern+['scripts/seed_regularized_reference.py','--protocol',c],.5)
 run('coarse_seed_all_native_audit',native+['scripts/audit_regularized_reference.py','--protocol',c,'--output',plan['coarse_seed_audit']],1.)
 run('coarse_first_step',mpi+['scripts/generate_regularized_reference.py','--protocol',c,'--qualification-stop','401'],.5)
 run('coarse_restart_step',mpi+['scripts/generate_regularized_reference.py','--protocol',c,'--qualification-stop','402'],.5)
 run('coarse_export',modern+['scripts/export_parallel_stream_probe.py','--protocol',plan['coarse_probe_protocol']],.25)
 run('coarse_independent_probe_audit',native+['scripts/verify_quadratic_substep_probe.py','--protocol',plan['coarse_probe_protocol']],.5)
 a=json.loads((P/plan['coarse_probe_audit']).read_text());assert a['passed'] and all(x['native_residual_N']<1e-8 for x in a['records'])
 run('full_seed',modern+['scripts/seed_regularized_reference.py','--protocol',full],2.)
 run('full_seed_all_native_audit',native+['scripts/audit_regularized_reference.py','--protocol',full,'--output',plan['full_seed_audit']],6.)
 run('full_continuation',mpi+['scripts/generate_regularized_reference.py','--protocol',full],168.)
 run('final_whole_element_summary',modern+['scripts/finalize_parallel_reference.py','--protocol',full],.5)
 run('final_every_state_native_audit',native+['scripts/audit_regularized_reference.py','--protocol',full,'--output',plan['final_audit'],'--require-complete'],18.)
 a=json.loads((P/plan['final_audit']).read_text());assert a['passed'] and a['full_cycle']
 stream=json.loads((P/full).read_text())
 run('R1_R2_fullcycle_mesh_comparison',native+['scripts/compare_quadratic_references.py','--config-a',plan['mesh_baseline_config'],'--config-b',stream['config'],'--output',plan['mesh_comparison_output']],8.)
 state.update(complete=True,reference_native_validated=True,active_phase='finished',scope='Full native reference/audits and controlled mesh comparison finished. Numerical1% gates and furthergeometry/load/ML/calibration/finaltest validation remain separate; no automatic neural retraining or publication.')
except BaseException as exc:state.update(complete=True,reference_native_validated=False,failure=str(exc),active_phase='stopped')
finally:
 persist();subprocess.run(modern+['scripts/write_reports.py'],cwd=P,env=env)
