"""Cluster-independent checks for result integrity, slicing and honest reporting."""
import importlib.util,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'hpc'))
from registry import event,read,atomic
from manuscript import render
from policy import segment_size,memory_reason

def test_ledger_detects_result_tampering(tmp_path):
 event(tmp_path,'stage_started',phase='known');event(tmp_path,'stage_finished',phase='known',return_code=0)
 assert len(read(tmp_path))==2
 p=tmp_path/'events.jsonl';s=p.read_text().replace('"return_code": 0','"return_code": 1');p.write_text(s)
 with pytest.raises(AssertionError):read(tmp_path)

def test_scheduler_slice_never_overbooks_declared_step_budget():
 for seconds in range(0,100000,137):
  n=segment_size(seconds,1142,4,1800)
  assert 0<=n<=4
  if n:assert n*(1800+120)+600<=seconds
 assert segment_size(1000,1142,4,1800)==0
 assert memory_reason(10*2**30,1,96,12)=='host-wide free-memory reserve'
 assert memory_reason(20*2**30,80*2**30,96,12)=='80 percent of allocated memory'
 assert memory_reason(20*2**30,20*2**30,96,12) is None

def test_manuscript_keeps_author_edits_and_pending_gates(tmp_path):
 run=tmp_path/'experiments/hpc/test';run.mkdir(parents=True);atomic(run/'state.json',dict(status='prepared',phase='platform',blockers=[]));atomic(run/'plan.json',dict(source_commit='fixture'))
 event(run,'run_prepared',scope='Test fixture, not HPC results')
 p=render(tmp_path,run);assert 'not complete' in p.read_text()
 p.write_text(p.read_text().replace('## 1. Introduction','## 1. Introduction\nAUTHOR EDIT TO PRESERVE'))
 event(run,'stage_blocked',phase='readiness',reason='no qualified reference')
 atomic(run/'state.json',dict(status='blocked',phase='readiness',blockers=['No qualified reference']))
 render(tmp_path,run);text=p.read_text();assert 'AUTHOR EDIT TO PRESERVE' in text and 'No qualified reference' in text
 assert 'submission-ready' in text and 'not submission-ready' in text

def test_independent_canonical_norm_constant_and_affine_fields():
 import itertools,numpy as np
 from quadrature import squared_norm
 vertices=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[0,0,1]])
 nodes=np.concatenate([vertices,np.array([(vertices[i]+vertices[j])/2 for i,j in itertools.combinations(range(4),2)])])
 b=(5-np.sqrt(5))/20;a=1-3*b;bary=np.array([[a,b,b,b],[b,a,b,b],[b,b,a,b],[b,b,b,a]])
 data=dict(coords=vertices,cells=np.array([[0,1,2,3]]),qpoints=bary[:,1:],qweights=np.ones(4)/24,canonical_cell_dofs=np.arange(30).reshape(1,10,3))
 constant=np.tile([1.,2.,3.],10);assert squared_norm(data,constant)==pytest.approx(14/6)
 assert squared_norm(data,nodes.reshape(-1))==pytest.approx(1/20)
 data['coords']=vertices+np.array([7.,-2.,3.]);assert squared_norm(data,constant)==pytest.approx(14/6)

def test_environment_freeze_changes_when_package_metadata_changes(tmp_path):
 from registry import environment_fingerprint
 p=tmp_path/'conda-meta';p.mkdir();f=p/'fixture.json';f.write_text('{"version":"1"}')
 before=environment_fingerprint(tmp_path);f.write_text('{"version":"2"}')
 assert environment_fingerprint(tmp_path)!=before

def test_explicit_environment_package_comparison(tmp_path):
 from check_environment_locks import check
 (tmp_path/'hpc').mkdir();(tmp_path/'.hpc/setup-records').mkdir(parents=True)
 for name in ['native','modern','tools']:
  (tmp_path/f'hpc/{name}-linux64-glibc217.lock').write_text('@EXPLICIT\nhttps://example.test/a.conda#digest\n')
  (tmp_path/f'.hpc/setup-records/{name}-installed.lock').write_text('# installed prefix\n@EXPLICIT\nhttps://example.test/a.conda\n')
 (tmp_path/'.hpc/setup-records/native-v2-installed.lock').write_text((tmp_path/'.hpc/setup-records/native-installed.lock').read_text())
 check(tmp_path)
 (tmp_path/'.hpc/setup-records/native-installed.lock').write_text('https://example.test/unexpected.conda\n')
 with pytest.raises(AssertionError):check(tmp_path)

def test_pbs_submissions_include_site_software_and_dependencies(tmp_path):
 import os,subprocess
 run=tmp_path/'run';run.mkdir();atomic(run/'plan.json',dict(run_id='test',job_hours=3,queue='fat',cpus=16,memory_gb=96,software='PBSJobApp',placement='excl'))
 bindir=tmp_path/'bin';bindir.mkdir();qsub=bindir/'qsub';qsub.write_text('#!/bin/sh\necho 123.server\n');qsub.chmod(0o755)
 env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'])
 subprocess.run([sys.executable,str(ROOT/'hpc/submit.py'),'--run',str(run),'--jobs','2'],env=env,check=True,capture_output=True,text=True)
 rows=[r for r in read(run) if r['kind']=='job_submission'];assert len(rows)==2
 for row in rows:
  assert 'software=PBSJobApp' in row['argv'] and 'place=excl' in row['argv'] and row['return_code']==0
 assert 'depend=afterok:123.server' in rows[1]['argv']

@pytest.mark.parametrize('with_prior',[False,True])
def test_single_launch_queues_setup_and_slices_once(tmp_path,monkeypatch,with_prior):
 from launch import launch
 (tmp_path/'experiments/manifests').mkdir(parents=True)
 (tmp_path/'data/processed/source').mkdir(parents=True)
 for name in ['restart.npz','displacement_coefficients.npy']:(tmp_path/'data/processed/source'/name).write_bytes(b'fixture')
 atomic(tmp_path/'experiments/manifests/hpc_migration_snapshot.json',dict(complete=True,reference='source',files={'data/processed/source/restart.npz':'fixture'}))
 if with_prior:
  from registry import artifact,sha
  parent=tmp_path/'experiments/hpc/prior';parent.mkdir(parents=True)
  atomic(parent/'plan.json',dict(snapshot_sha256=sha(tmp_path/'experiments/manifests/hpc_migration_snapshot.json'),cumulative_pipeline_budget_hours=192))
  artifact(tmp_path,parent,parent/'plan.json','immutable study plan')
  atomic(parent/'state.json',dict(status='blocked',terminal_rejection=True,phase='platform',generation_seconds=0.,completed_phases=[],cumulative_seconds=7201.44,blockers=['declared stage wall budget']))
 calls=[]
 import subprocess
 def fake_run(argv,**kwargs):
  calls.append(argv)
  return subprocess.CompletedProcess(argv,0,str(len(calls))+'.pbs\n','')
 monkeypatch.setattr('launch.subprocess.run',fake_run)
 monkeypatch.setattr('launch.subprocess.check_output',lambda *a,**k:'frozencommit\n')
 directory=launch(tmp_path,'fixture',jobs=3,prior_run='prior' if with_prior else None)
 assert len(calls)==4
 for i,command in enumerate(calls[1:],1):
  assert 'depend=afterok:'+str(i)+'.pbs' in command
  assert 'select=1:ncpus=28:mem=240gb' in command
 first=(directory/'slice-001.pbs').read_text();assert 'hpc/prepare.py' in first and 'hpc/run.py' in first
 assert 'hpc/prepare.py' not in (directory/'slice-002.pbs').read_text()
 assert '.hpc/tools/bin' in first and 'frozencommit' in first
 if with_prior:
  assert '--prior-run prior' in first
  assert json.loads((directory/'launch.json').read_text())['prior_resource_evidence']['cumulative_seconds']==7201.44
 with pytest.raises(FileExistsError):launch(tmp_path,'fixture',jobs=3)
 assert len(calls)==4

def test_single_launch_requires_transferred_data(tmp_path):
 from launch import launch
 (tmp_path/'experiments/manifests').mkdir(parents=True)
 atomic(tmp_path/'experiments/manifests/hpc_migration_snapshot.json',dict(complete=True,reference='missing',files={}))
 with pytest.raises(SystemExit,match='Transfer the preserved experiment data'):launch(tmp_path,'fixture')
 assert not(tmp_path/'.hpc/launches/fixture').exists()

@pytest.mark.parametrize('partial',[False,True])
def test_driver_output_planning_and_execution_share_one_directory(tmp_path,monkeypatch,partial):
 """Exercise actual driver functions together, replacing only the native backend."""
 import ast,time,types
 monkeypatch.setenv('PBS_JOBID','fixture.pbs')
 source=ast.parse((ROOT/'hpc/run.py').read_text())
 selected=[n for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['execute','next_output','Paused','Rejected']]
 run=tmp_path/'experiments/hpc/fixture';run.mkdir(parents=True)
 state=dict(attempts={},completed_phases=[],cumulative_seconds=0)
 class NativeFixture:
  def __init__(self,cmd,**kw):
   # A backend fixture supplies a result; scientific qualification is not asserted.
   Path(cmd[0]).write_text(json.dumps(dict(passed=not partial,complete=not partial,audit_complete=not partial,chunk_passed=True,scope='test fixture')))
  def poll(self):return 0
  def wait(self):return 0
 ns=dict(P=tmp_path,run=run,state=state,plan=dict(cumulative_pipeline_budget_hours=1),interrupted=False,deadline=time.monotonic()+10000,time=time,env={},os=__import__('os'),subprocess=types.SimpleNamespace(Popen=NativeFixture,STDOUT=-2),event=lambda *a,**k:None,render=lambda *a,**k:None,persist=lambda:None,artifact=lambda *a,**k:None,json=json)
 exec(compile(ast.Module(body=selected,type_ignores=[]),'driver fixture','exec'),ns)
 out=ns['next_output']('platform','result.json')
 assert not(tmp_path/out).parent.exists() # Planning must have no filesystem side effects.
 assert ns['next_output']('platform','result.json')==out
 ns['execute']('platform',[str(tmp_path/out)],10,[(out,'test result')],audit_chunk=partial)
 assert state['completed_phases']==([] if partial else ['platform']) and state['attempts']['platform']==1
 assert (tmp_path/out).is_file() and (tmp_path/out).with_name('command.log').is_file()
 previous=(tmp_path/out).read_bytes()
 next_out=ns['next_output']('platform','result.json')
 assert 'attempt-002' in next_out and not(tmp_path/next_out).parent.exists()
 assert (tmp_path/out).read_bytes()==previous

def test_status_before_first_reference_slice_is_not_failure(tmp_path):
 from status import read_status
 launch=tmp_path/'.hpc/launches/test';launch.mkdir(parents=True)
 event(launch,'job_submission',role='setup',return_code=0,stdout='123.pbs',stderr='')
 event(launch,'job_submission',role='reference_slice_1',return_code=0,stdout='124.pbs',stderr='')
 result,events,initialized=read_status(tmp_path,'experiments/hpc/test')
 assert result['status']=='awaiting_initialization' and not initialized
 assert result['scientific_progress'] is None and len(result['submitted_jobs'])==2
 assert not(tmp_path/'experiments/hpc/test').exists()
 atomic(tmp_path/'.hpc/setup-result-123.pbs.json',dict(exit_code=0))
 result,_,_=read_status(tmp_path,'experiments/hpc/test');assert result['status']=='awaiting_initialization'
 atomic(tmp_path/'.hpc/setup-result-123.pbs.json',dict(exit_code=127))
 result,_,_=read_status(tmp_path,'experiments/hpc/test');assert result['status']=='setup_failed'

def test_status_preserves_initialized_scientific_blocker(tmp_path):
 from status import read_status
 run=tmp_path/'experiments/hpc/test';run.mkdir(parents=True)
 state=dict(status='blocked',blockers=['fixture negative scientific result'])
 atomic(run/'state.json',state)
 result,_,initialized=read_status(tmp_path,'experiments/hpc/test')
 assert initialized and result==state

def test_native_python_addons_close_dependencies_without_replacing_fe_packages():
 from packaging.requirements import Requirement
 from packaging.utils import canonicalize_name
 original=json.loads((ROOT/'hpc/native-linux64-glibc217-resolution.json').read_text())['packages']
 additions=json.loads((ROOT/'hpc/native-python-addons-resolution.json').read_text())['packages']
 old={canonicalize_name(x['name']):x['version'] for x in original}
 new={canonicalize_name(x['name']):x['version'] for x in additions}
 assert not(old.keys() & new.keys()) and new['meshio']=='5.3.5'
 versions={**old,**new}
 for package in additions:
  assert package['wheel'].endswith('-py3-none-any.whl')
  assert ('--hash=sha256:'+package['sha256']) in (ROOT/'hpc/native-python-addons.lock').read_text()
  for item in package['requires_dist']:
   requirement=Requirement(item)
   if requirement.marker and not requirement.marker.evaluate({'python_version':'3.11','python_full_version':'3.11.18','extra':''}):continue
   assert versions[canonicalize_name(requirement.name)] in requirement.specifier

def test_native_addon_installed_file_tamper_detection(tmp_path,monkeypatch):
 import base64,hashlib,types
 import check_native_addons as checker
 (tmp_path/'hpc').mkdir();payload=tmp_path/'fixture.py';payload.write_bytes(b'original')
 atomic(tmp_path/'hpc/native-python-addons-resolution.json',dict(packages=[dict(name='fixture',version='1.0',sha256='wheel-fixture')]))
 recorded=base64.urlsafe_b64encode(hashlib.sha256(payload.read_bytes()).digest()).rstrip(b'=').decode()
 file=types.SimpleNamespace(hash=types.SimpleNamespace(mode='sha256',value=recorded))
 distribution=types.SimpleNamespace(version='1.0',files=[file],locate_file=lambda _:payload)
 monkeypatch.setattr(checker,'distribution',lambda _:distribution)
 assert checker.check(tmp_path)[0]['verified_files']==1
 payload.write_bytes(b'changed')
 with pytest.raises(AssertionError,match='hash mismatch'):checker.check(tmp_path)

def test_mpi_probe_help_exits_cleanly_but_bad_cli_still_aborts():
 import ast,types
 source=ast.parse((ROOT/'hpc/probe_native.py').read_text())
 guard=next(n for n in reversed(source.body) if isinstance(n,ast.If))
 for exit_code in [0,2]:
  calls=[]
  def main():raise SystemExit(exit_code)
  def abort(code):calls.append(code);raise RuntimeError('fixture MPI abort')
  ns=dict(__name__='__main__',main=main,MPI=types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(Abort=abort)))
  if exit_code==0:
   with pytest.raises(SystemExit) as exc:exec(compile(ast.Module(body=[guard],type_ignores=[]),'probe CLI fixture','exec'),ns)
   assert exc.value.code==0 and not calls
  else:
   with pytest.raises(RuntimeError,match='fixture MPI abort'):exec(compile(ast.Module(body=[guard],type_ignores=[]),'probe CLI fixture','exec'),ns)
   assert calls==[1]

def audit_fixture(count):
 import numpy as np
 records=[dict(step=i,native_residual_N=1e-10,field_sha256='fixture-'+str(i)) for i in range(count)]
 histories={k:np.arange(6,dtype=float)+count for k in ['u','u_old','v_old','a_old']}
 return records,histories

def test_audit_checkpoint_chain_restores_exact_histories(tmp_path):
 from audit_checkpoint import commit,load_latest
 import numpy as np
 binding=dict(source='fixed-fixture',mode='qualification')
 first,histories=audit_fixture(2);parent=commit(tmp_path,binding,first,histories)
 saved=load_latest(tmp_path,binding)
 for k in histories:np.testing.assert_array_equal(saved['histories'][k],histories[k])
 second,following=audit_fixture(4);commit(tmp_path,binding,second,following,parent)
 latest=load_latest(tmp_path,binding)
 assert latest['records']==second and latest['meta']['checked_steps']==4
 for k in following:np.testing.assert_array_equal(latest['histories'][k],following[k])
 with pytest.raises(AssertionError,match='Never overwrite'):commit(tmp_path,binding,second,following,parent)
 with pytest.raises(AssertionError,match='binding changed'):load_latest(tmp_path,dict(source='other'))

@pytest.mark.parametrize('point',['native_saved','before_publish','after_publish'])
def test_audit_crash_boundaries_never_promote_partial_bundle(tmp_path,point):
 from audit_checkpoint import commit,load_latest
 binding=dict(source='fixture');records,histories=audit_fixture(2)
 def fault(where):
  if where==point:raise RuntimeError('simulated audit interruption')
 with pytest.raises(RuntimeError,match='simulated audit interruption'):commit(tmp_path,binding,records,histories,fault_hook=fault)
 saved=load_latest(tmp_path,binding)
 assert (saved is not None)==(point=='after_publish')
 if saved:assert saved['meta']['checked_steps']==2
 else:
  commit(tmp_path,binding,records,histories)
  assert load_latest(tmp_path,binding)['meta']['checked_steps']==2

def test_audit_bundle_rejects_modified_native_histories(tmp_path):
 from audit_checkpoint import commit,load_latest
 binding=dict(source='fixture');records,histories=audit_fixture(2);commit(tmp_path,binding,records,histories)
 with (tmp_path/'checked-000002/native.npz').open('ab') as f:f.write(b'changed')
 with pytest.raises(AssertionError,match='native checkpoint changed'):load_latest(tmp_path,binding)

def test_audit_bundle_rejects_rewritten_earlier_records(tmp_path):
 from audit_checkpoint import commit,load_latest
 from registry import sha
 binding=dict(source='fixture');records,histories=audit_fixture(2);parent=commit(tmp_path,binding,records,histories)
 more,following=audit_fixture(4);commit(tmp_path,binding,more,following,parent)
 folder=tmp_path/'checked-000004';more[0]['native_residual_N']=2e-10
 (folder/'records.json').write_text(json.dumps(more))
 meta=json.loads((folder/'manifest.json').read_text());meta['records_sha256']=sha(folder/'records.json');(folder/'manifest.json').write_text(json.dumps(meta))
 with pytest.raises(AssertionError,match='Previously accepted audit records changed'):load_latest(tmp_path,binding)

def chunked_native_fixture(tmp_path,monkeypatch,total=4):
 """Small deterministic backend fixture, not FE or scientific qualification."""
 import types,hashlib,numpy as np
 (tmp_path/'hpc').mkdir();(tmp_path/'hpc/audit_checkpoint.py').write_bytes((ROOT/'hpc/audit_checkpoint.py').read_bytes())
 from registry import sha
 class Problem:
  def __init__(self):
   self.values={k:np.zeros(6) for k in ['u','u_old','v_old','a_old']};self.pressure=0.
   self.parameters={'p':types.SimpleNamespace(assign=lambda x:setattr(self,'pressure',x))}
   for name in self.values:setattr(self,name,types.SimpleNamespace(vector=lambda name=name:types.SimpleNamespace(zero=lambda:self.values[name].fill(0))))
  def _update_fields(self):
   v=self.values;velocity=(v['u']-v['u_old'])+.5*v['v_old'];acceleration=velocity-v['v_old'];v['u_old']=v['u'].copy();v['v_old']=velocity;v['a_old']=acceleration
 class Adapter:
  def __init__(self,*_):self.problem=Problem();self.comm=types.SimpleNamespace(size=1)
  def assign(self,k,v):self.problem.values[k]=np.array(v,copy=True)
  def gather(self,k):return self.problem.values[k].copy()
 def residual(problem):
  v=problem.values;expected=v['u_old']+.25*v['v_old']+.125*v['a_old']+problem.pressure
  return types.SimpleNamespace(norm=lambda _:float(np.linalg.norm(v['u']-expected))+1e-12)
 monkeypatch.setitem(sys.modules,'lv_surrogate.physics.parallel_fem',types.SimpleNamespace(ParallelP2=Adapter))
 monkeypatch.setitem(sys.modules,'lv_surrogate.physics.reference_fem',types.SimpleNamespace(df=None,residual=residual))
 monkeypatch.setitem(sys.modules,'platform_native',types.SimpleNamespace(norm_check=lambda *args:0.))
 spec=importlib.util.spec_from_file_location('checkpointed_native_fixture',ROOT/'hpc/platform_native_chunked.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.P=tmp_path
 source=tmp_path/'data/processed/source';source.mkdir(parents=True);problem=Problem();fields=[]
 for pressure in [.1*(i+1) for i in range(total)]:
  v=problem.values;v['u']=v['u_old']+.25*v['v_old']+.125*v['a_old']+pressure;fields.append(v['u'].copy());problem._update_fields()
 np.save(source/'displacement_coefficients.npy',np.array(fields));np.savez(source/'inputs.npz',pressure=[.1*(i+1) for i in range(total)],time=[.1*(i+1) for i in range(total)]);np.savez(source/'restart.npz',completed_steps=total,**problem.values)
 atomic(source/'manifest.json',dict(config={},completed_steps=total,steps=[dict(step=i,residual_l2_N=1e-12,SNES_reason=2) for i in range(total)]))
 atomic(source/'positivity.json',[dict(step=i,endpoint=dict(passed=True,minimum_Bernstein_bound=.1),stress=dict(passed=True,minimum_Bernstein_bound=.1)) for i in range(total)])
 atomic(tmp_path/'snapshot.json',dict(reference='source',source_reference='fixture',accepted_steps=total,numerical_sources={},files={str((source/f).relative_to(tmp_path)):sha(source/f) for f in ['manifest.json','restart.npz','inputs.npz','positivity.json']},prefix_fields_sha256=hashlib.sha256(np.array(fields).tobytes()).hexdigest()))
 atomic(tmp_path/'plan.json',dict(snapshot='snapshot.json',native_prefix='.hpc/native-v2',canonical_checkpoint='canonical.npz',known_start_checkpoint='known-start.npz'))
 return module

def test_checkpointed_native_audit_resumes_and_checks_final_histories(tmp_path,monkeypatch):
 import numpy as np
 audit=chunked_native_fixture(tmp_path,monkeypatch)
 first=audit.run('plan.json','first.json','platform','split',2)
 assert first['checked_steps']==2 and first['chunk_passed'] and not first['passed'] and not first['complete']
 second=audit.run('plan.json','second.json','platform','split',2)
 fresh=audit.run('plan.json','fresh.json','platform','fresh',4)
 assert second['complete'] and second['passed'] and second['audit_complete']
 assert second['prefix_sha256']==fresh['prefix_sha256']
 for history in second['history_differences'].values():assert history['max_abs']==0.
 from audit_checkpoint import load_latest
 a=load_latest(tmp_path/'split',json.loads((tmp_path/'split/checked-000004/manifest.json').read_text())['binding'])
 b=load_latest(tmp_path/'fresh',json.loads((tmp_path/'fresh/checked-000004/manifest.json').read_text())['binding'])
 for k in a['histories']:np.testing.assert_array_equal(a['histories'][k],b['histories'][k])

def test_checkpointed_audit_rejects_changed_previously_checked_field(tmp_path,monkeypatch):
 import numpy as np
 audit=chunked_native_fixture(tmp_path,monkeypatch);audit.run('plan.json','first.json','platform','split',2)
 path=tmp_path/'data/processed/source/displacement_coefficients.npy';fields=np.load(path);fields[0,0]+=1;np.save(path,fields)
 with pytest.raises(AssertionError,match='Previously audited field changed'):audit.run('plan.json','second.json','platform','split',2)

def test_native_resume_control_schedules_split_and_fresh_comparison(tmp_path,monkeypatch):
 import types
 audit=chunked_native_fixture(tmp_path,monkeypatch,total=6)
 import qualify_audit_resume as qualifier
 monkeypatch.setattr(qualifier,'P',tmp_path);(tmp_path/'control').mkdir();calls=[]
 def native(argv,**kwargs):
  def arg(name):return argv[argv.index(name)+1]
  calls.append(argv);audit.run(arg('--spec'),arg('--output'),arg('--mode'),arg('--checkpoint-root'),int(arg('--max-steps')),int(arg('--qualification-steps')))
  return types.SimpleNamespace(returncode=0)
 monkeypatch.setattr(qualifier.subprocess,'run',native)
 monkeypatch.setattr(sys,'argv',['qualifier','--spec','plan.json','--output','control/result.json']);qualifier.main()
 result=json.loads((tmp_path/'control/result.json').read_text());assert result['passed'] and result['exact_history_equality'] and len(calls)==3
 assert result['source_steps']==[2,3,4,5] and result['nonzero_native_histories']

def test_followup_preserves_prior_spent_budget_and_rejects_generating_parent(tmp_path):
 from followup_budget import prior_budget
 from registry import artifact
 parent=tmp_path/'experiments/hpc/prior';parent.mkdir(parents=True)
 atomic(parent/'plan.json',dict(snapshot_sha256='fixed',cumulative_pipeline_budget_hours=192))
 artifact(tmp_path,parent,parent/'plan.json','immutable study plan')
 state=dict(status='blocked',terminal_rejection=True,phase='platform',generation_seconds=0.,completed_phases=[],cumulative_seconds=7201.44,blockers=['declared stage wall budget'])
 atomic(parent/'state.json',state)
 budget=prior_budget(tmp_path,'prior','fixed');assert budget['cumulative_seconds']==7201.44 and budget['generation_seconds']==0.
 state['generation_seconds']=1.;atomic(parent/'state.json',state)
 with pytest.raises(AssertionError,match='does not consume prior generated states'):prior_budget(tmp_path,'prior','fixed')
