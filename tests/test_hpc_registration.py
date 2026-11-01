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
 for name in ['native','modern']:
  (tmp_path/f'hpc/{name}-linux64-glibc217.lock').write_text('@EXPLICIT\nhttps://example.test/a.conda#digest\n')
  (tmp_path/f'.hpc/setup-records/{name}-installed.lock').write_text('# installed prefix\n@EXPLICIT\nhttps://example.test/a.conda\n')
 check(tmp_path)
 (tmp_path/'.hpc/setup-records/native-installed.lock').write_text('https://example.test/unexpected.conda\n')
 with pytest.raises(AssertionError):check(tmp_path)
