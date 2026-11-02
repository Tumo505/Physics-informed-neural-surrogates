"""PBS-native registered driver. Single-node FE, bounded slices, no gate-bypassing ML."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil,re
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'hpc'))
from registry import event,artifact,atomic,sha,job_metadata,environment_fingerprint,read
from manuscript import render
from policy import segment_size,memory_reason
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--slice-seconds',required=True,type=int);a=ap.parse_args();assert os.environ.get('PBS_JOBID'),'Use qsub; never run this on the gateway';run=P/a.run;plan=json.loads((run/'plan.json').read_text());state=json.loads((run/'state.json').read_text());lock=(run/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);meta=job_metadata();assert len(meta.get('allocated_hosts',[]))==1,'Only single-node bundled MPICH is qualified';job_start=time.monotonic();prior=state['cumulative_seconds'];deadline=job_start+a.slice_seconds-300;interrupted=False;last_progress=None
registered_plans=[x for x in read(run) if x['kind']=='artifact' and x['path']==str((run/'plan.json').relative_to(P))];assert registered_plans and sha(run/'plan.json')==registered_plans[-1]['sha256'],'Immutable study plan changed'
for f,h in plan['scientific_sources'].items():assert sha(P/f)==h,'Source changed after run freeze: '+f
for prefix,digest in plan['environment_sha256'].items():assert environment_fingerprint(P/prefix)==digest,'Runtime environment changed after freeze: '+prefix
assert sha(P/plan['snapshot'])==plan['snapshot_sha256'];snapshot=json.loads((P/plan['snapshot']).read_text());root=P/'data/processed'/plan['target_name'];spec=str((run/'plan.json').relative_to(P));native=[str(P/'.tools/bin/micromamba'),'run','-p',str(P/plan.get('native_prefix','.fenics')),'python'];mpi=native[:-1]+['mpiexec','-n',str(plan['ranks']),'python'];modern=[str(P/'.venv/bin/python')];env=os.environ.copy();env['PATH']=str(P/'.hpc/tools/bin')+os.pathsep+env.get('PATH','');env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HYDRA_LAUNCHER='fork',FI_PROVIDER='tcp',CUDA_VISIBLE_DEVICES='',MAMBA_ROOT_PREFIX=str(P/'.hpc/mamba'),XDG_CACHE_HOME=str(P/'.hpc/cache'),DIJITSO_CACHE_DIR=str(P/'.hpc/fenics-cache'),MPLCONFIGDIR=str(P/'.hpc/cache/matplotlib'));env.pop('FI_TCP_IFACE',None)
class Paused(Exception):pass
class Rejected(Exception):pass

def signal_pause(signum,frame):
 global interrupted
 interrupted=True
for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,signal_pause)
def persist():
 state['cumulative_seconds']=prior+time.monotonic()-job_start;atomic(run/'state.json',state)
def progress():
 global last_progress
 if not (root/'manifest.json').exists():return
 m=json.loads((root/'manifest.json').read_text());key=m['completed_steps']
 if key!=last_progress:
  r=m['steps'][-1];event(run,'reference_progress',completed_steps=key,n_steps=m['n_steps'],time_s=r['time_s'],residual_N=r['residual_l2_N'],scope='Online native admission; final independent audit pending');last_progress=key;render(P,run)
def host_memory():
 values={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  k,v=line.split(':',1);values[k]=int(v.strip().split()[0])*1024
 if 'MemAvailable' not in values:values['MemAvailable']=values.get('MemFree',0)
 return values

def execute(phase,cmd,max_seconds,outputs=(),generation=False,audit_chunk=False):
 if interrupted or deadline-time.monotonic()<max_seconds+300:raise Paused('Insufficient reserved PBS time to start '+phase)
 if state['cumulative_seconds']+max_seconds>plan['cumulative_pipeline_budget_hours']*3600:raise Rejected('Declared cumulative pipeline budget cannot accommodate '+phase)
 attempt=state['attempts'].get(phase,0)+1;state['attempts'][phase]=attempt;state['phase']=phase;state['status']='running';persist();attemptdir=run/phase/f'attempt-{attempt:03d}';attemptdir.mkdir(parents=True);log=attemptdir/'command.log';event(run,'stage_started',phase=phase,argv=cmd,pbs_job_id=os.environ['PBS_JOBID'],attempt=attempt);render(P,run);start=time.monotonic();peak=0;pause_reason=None;code=None
 with log.open('w') as f:
  child=subprocess.Popen(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while child.poll() is None:
    rows=[tuple(map(int,x.split())) for x in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True).splitlines() if x.strip()];tree={child.pid}
    while True:
     found={pid for pid,parent,rss in rows if parent in tree}
     if found<=tree:break
     tree.update(found)
    rss=sum(rss*1024 for pid,parent,rss in rows if pid in tree);peak=max(peak,rss);mem=host_memory()
    if interrupted or time.monotonic()>=deadline:pause_reason='scheduler slice interruption'
    memory_pause=memory_reason(mem.get('MemAvailable',0),rss,plan['memory_gb'],plan['host_free_reserve_gib'])
    if memory_pause:pause_reason=memory_pause
    if shutil.disk_usage(P).free<100*2**30:pause_reason='100 GiB free-disk reserve'
    if time.monotonic()-start>max_seconds:pause_reason='declared stage wall budget'
    persist();progress()
    if pause_reason:break
    time.sleep(2)
  finally:
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  code=child.wait()
 elapsed=time.monotonic()-start
 if generation:state['generation_seconds']+=elapsed
 artifact(P,run,log,'executed command log');event(run,('audit_chunk_finished' if audit_chunk else 'stage_finished') if code==0 and not pause_reason else 'stage_interrupted' if pause_reason else 'stage_failed',phase=phase,attempt=attempt,return_code=code,reason=pause_reason,wall_seconds=elapsed,sampled_peak_RSS_bytes=peak,pbs_job_id=os.environ['PBS_JOBID']);persist()
 if pause_reason:
  if pause_reason in ['declared stage wall budget','100 GiB free-disk reserve'] or not (generation or audit_chunk):raise Rejected('Interrupted non-resumable stage/resource bound: '+pause_reason+'; preserved attempt requires a separately declared follow-up')
  raise Paused(pause_reason)
 if code:raise Rejected(f'{phase} exited {code}; inspect {log.relative_to(P)}')
 for path,role in outputs:
  result=json.loads((P/path).read_text()) if str(path).endswith('.json') else None
  artifact(P,run,path,role)
  if audit_chunk:
   assert result is not None and result.get('chunk_passed'),'Audit chunk did not pass'
   assert result['audit_complete']==bool(result['passed'])==bool(result['complete']),'Partial audit promoted inconsistently'
  elif result is not None and ('passed' in result) and not result['passed']:raise Rejected(phase+' scientific gate failed')
 if not audit_chunk:state['completed_phases'].append(phase)
 persist();render(P,run)

def path(name):return str((run/name).relative_to(P))
def next_output(phase,name):
 # Every interrupted audit gets a new output ID; no existing result is overwritten.
 attempt=state['attempts'].get(phase,0)+1;directory=run/phase/f'attempt-{attempt:03d}';return str((directory/name).relative_to(P))
def completed(phase):return phase in state['completed_phases']
def audit_phase(phase,mode):
 while not completed(phase):
  out=next_output(phase,'result.json')
  command=native+['hpc/platform_native_chunked.py','--spec',spec,'--mode',mode,'--output',out,'--checkpoint-root',path(phase+'/checkpoints'),'--max-steps','64']
  execute(phase,command,7200,[(out,'Checkpointed audit chunk; full-audit status explicit')],audit_chunk=True)
  result=json.loads((P/out).read_text());state['audit_progress']=dict(phase=phase,checked_steps=result['checked_steps'],total_steps=result['total_steps'],audit_complete=result['audit_complete']);event(run,'audit_progress',**state['audit_progress']);persist();render(P,run)
  if result['audit_complete']:
   state.setdefault('audit_outputs',{})[phase]=out
   if mode=='platform':
    artifact(P,run,plan['canonical_checkpoint'],'separate native-consistent platform restart');artifact(P,run,plan['known_start_checkpoint'],'native known-case starting histories');state['platform_output']=out
   state['completed_phases'].append(phase);persist();return out


event(run,'job_started',**meta);state['status']='running';persist();render(P,run)
try:
 if state.get('status_before') in ['failed','blocked'] or state.get('terminal_rejection'):raise Rejected('Previous scientific rejection remains unresolved; do not retry this run')
 if not completed('audit_resume_control'):
  out=next_output('audit_resume_control','result.json');execute('audit_resume_control',modern+['hpc/qualify_audit_resume.py','--spec',spec,'--output',out],7200,[(out,'Known-prefix native audit stop/restart qualification')])
 if not completed('platform'):
  out=audit_phase('platform','platform')
 if not completed('known_probe'):
  out=next_output('known_probe','result.json');f=out.replace('result.json','fields.npy');cp=out.replace('result.json','checkpoint.npz');execute('known_probe',mpi+['hpc/probe_native.py','--spec',spec,'--output',out,'--fields',f,'--checkpoint',cp],3600,[(out,'known-field MPI qualification'),(f,'actual known-case MPI coefficient fields'),(cp,'actual native known-case endpoint histories')]);state.update(probe_output=out,probe_fields=f,probe_checkpoint=cp);persist()
 if not completed('known_probe_independent'):
  out=next_output('known_probe_independent','result.json');execute('known_probe_independent',native+['hpc/verify_probe_native.py','--spec',spec,'--probe',state['probe_output'],'--fields',state['probe_fields'],'--checkpoint',state['probe_checkpoint'],'--output',out],1800,[(out,'independent serial known-case verification')]);state['probe_verification']=out;persist()
 if not completed('seed'):
  original=json.loads((P/snapshot['source_stream_protocol']).read_text());p=dict(original);p.update(config=plan['config'],seed_reference=snapshot['reference'],source_reference=snapshot['source_reference'],base_seed_steps=snapshot['accepted_steps'],seed_steps=snapshot['accepted_steps'],base_checkpoint=plan['canonical_checkpoint'],probe_chain=[],expected_source_prefix_sha256=snapshot['prefix_fields_sha256'],parallel_ranks=plan['ranks'],allowed_qualification_stops=list(range(snapshot['accepted_steps']+1,snapshot['n_steps']+1)),required_audits=[],required_audit_sha256={state['platform_output']:sha(P/state['platform_output']),state['probe_verification']:sha(P/state['probe_verification'])},input_sha256={**snapshot['files'],plan['canonical_checkpoint']:sha(P/plan['canonical_checkpoint'])},frozen_source_sha256=snapshot['numerical_sources'])
  p['resource_protocol']=dict(native_hours=168,cumulative_pipeline_hours=192,per_step_seconds=1800,process_tree_RSS_GiB=.8*plan['memory_gb'],free_disk_reserve_GiB=100,scope='Separate registered PBS protocol; original laptop protocol unchanged.');state['stream_protocol']=path('stream-protocol.json');atomic(P/state['stream_protocol'],p);persist();execute('seed',modern+['scripts/seed_regularized_reference.py','--protocol',state['stream_protocol']],1800)
 if not completed('seed_audit'):
  out=audit_phase('seed_audit','seed');m=json.loads((root/'manifest.json').read_text());m['prefix_independent_native_audit_passed']=True;m['prefix_native_audit']=out;atomic(root/'manifest.json',m)
 while not completed('reference'):
  m=json.loads((root/'manifest.json').read_text());assert not(root/'failure.json').exists(),'Native failure output retained; never auto-retry'
  if m['complete']:
   state['completed_phases'].append('reference');persist();break
  left=deadline-time.monotonic();steps=segment_size(left,m['n_steps']-m['completed_steps'],plan['segment_steps'],plan['per_step_seconds'])
  if steps<1:raise Paused('Clean checkpoint stop before PBS wall-time limit')
  if state['generation_seconds']+steps*plan['per_step_seconds']>plan['native_generation_budget_hours']*3600:raise Rejected('Remaining declared native-generation budget insufficient')
  i=m['completed_steps'];trace=root/f'custom_trace_{i:06d}.json'
  if trace.exists():
   archived=run/'interrupted-traces'/f'{i:06d}-{sha(trace)[:12]}.json';archived.parent.mkdir(exist_ok=True);shutil.copyfile(trace,archived);artifact(P,run,archived,'preserved interrupted custom trace')
  phase=f'reference_{i:06d}';execute(phase,mpi+['scripts/generate_regularized_reference.py','--protocol',state['stream_protocol'],'--qualification-stop',str(i+steps)],steps*plan['per_step_seconds']+300,generation=True)
 if not completed('positivity_summary'):execute('positivity_summary',modern+['scripts/finalize_parallel_reference.py','--protocol',state['stream_protocol']],1800)
 if not completed('final_audit'):
  out=audit_phase('final_audit','final')
 if not completed('mesh_comparison'):
  out=next_output('mesh_comparison','result.json');execute('mesh_comparison',native+['scripts/compare_quadratic_references.py','--config-a','configs/physics/dynamic_quadratic_reference_halfstep_ew_refined.json','--config-b',plan['config'],'--output',out],8*3600,[(out,'full-cycle R1/R2 sensitivity, not continuum truth')]);state['mesh_comparison']=out;persist()
  comparison=json.loads((P/out).read_text())
  if not comparison['passes_declared_one_percent_sensitivity']:raise Rejected('Full-cycle R1/R2 mesh sensitivity failed unchanged1% gate: '+str(comparison['relative_space_time_L2']))
 if not completed('qualified_followup_gate'):
  execute('qualified_followup_gate',modern+['scripts/assess_qualified_followup_readiness.py'],600,[('experiments/metrics/dynamic_qualified_followup_readiness.json','qualified-reference/gradient/transfer readiness')])
 gate=json.loads((P/'experiments/metrics/dynamic_qualified_followup_readiness.json').read_text());blockers=gate.get('blockers',[])
 if not gate['ready_to_train']:raise Rejected('Qualified follow-up blocked: '+'; '.join(blockers))
 raise Rejected('Qualified reference gate passed, but a registered full-P2 train/calibration/untouched-test executor must be supplied; older inspected-test/P1 training is not silently substituted.')
except Paused as exc:
 state.update(status='paused',pause_reason=str(exc));event(run,'job_paused',phase=state.get('phase'),reason=str(exc),pbs_job_id=os.environ['PBS_JOBID']);persist();render(P,run);sys.exit(0)
except BaseException as exc:
 state.update(status='blocked',terminal_rejection=True,blockers=[str(exc)]);event(run,'stage_blocked',phase=state.get('phase'),reason=str(exc),pbs_job_id=os.environ['PBS_JOBID']);persist();render(P,run);sys.exit(2)
finally:
 persist();event(run,'job_finished',status=state['status'],phase=state.get('phase'),pbs_job_id=os.environ['PBS_JOBID']);render(P,run)
