"""Wait for the existing diagnostic, then qualify a separate native-consistent restart."""
from pathlib import Path
import fcntl,json,subprocess,sys,time,hashlib
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.durable_reference import atomic_json,sha
out=P/'experiments/metrics/dynamic_native_history_followup_20261101.json';lock=(P/'experiments/logs/dynamic_native_history_followup_20261101.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not out.exists()
state=dict(complete=False,history_issue_revalidated=False,study_validation_passed=False,active_phase='wait_existing_readonly_reconciliation');atomic_json(out,state);started=time.perf_counter()
try:
 while True:
  w=json.loads((P/'experiments/metrics/dynamic_native_history_reconciliation_20261101_workflow.json').read_text())
  if w['complete']:break
  assert time.perf_counter()-started<2100,'Existing diagnostic did not complete in bounded wait';time.sleep(5)
 qpath=P/'experiments/metrics/dynamic_native_history_reconciliation_20261101.json';assert qpath.exists();q=json.loads(qpath.read_text());assert q['complete'] and q['passed'] and not q['original_audit_passed']
 subprocess.run([str(P/'.venv/bin/python'),'scripts/verify_native_history_reconciliation.py'],cwd=P,check=True)
 diag=json.loads((P/'configs/physics/dynamic_native_history_reconciliation_20261101.json').read_text());canonical=diag['canonical_checkpoint'];source=diag['reference']
 new_source_files=['scripts/seed_regularized_reference.py','scripts/verify_native_history_reconciliation.py']
 for f in new_source_files:
  h=sha(P/f);snap=P/'experiments/manifests/source_snapshots'/f'{h}_{Path(f).name}'
  if not snap.exists():snap.write_bytes((P/f).read_bytes())
 old_paths=[*list((P/'configs/physics').glob('dynamic_regularized_stream_*_20261101_v2*.json')),P/'configs/physics/dynamic_regularized_full_reference_20261101_v2.json']
 for old in old_paths:
  new=Path(str(old).replace('20261101_v2','20261101_v3'));assert not new.exists();p=json.loads(json.dumps(json.loads(old.read_text())).replace('20261101_v2','20261101_v3'))
  if 'frozen_source_sha256' in p:
   for f in new_source_files:p['frozen_source_sha256'][f]=sha(P/f)
  if 'scope' in p:p['scope']+=' v3 separately qualifies canonicalnativehistorycandidate fromread-onlyreplay;originalv2pointwisehistoryaudit/checkpoint remainsfailedandpreserved. Originalhistorytolerance unchanged.'
  if 'base_seed_steps' in p and 'third' in new.name:
   p.update(seed_reference=source,base_seed_steps=458,seed_steps=458,base_checkpoint=canonical,probe_chain=[],expected_source_prefix_sha256=q['source_prefix_fields_sha256'],native_history_reconciliation=dict(protocol='configs/physics/dynamic_native_history_reconciliation_20261101.json',qualification='experiments/metrics/dynamic_native_history_reconciliation_20261101.json',qualification_sha256=sha(qpath),original_saved_history_retained=True,new_history_representation_requires_independent_allstate_audit=True))
   p['required_audit_sha256']['experiments/metrics/dynamic_native_history_reconciliation_20261101.json']=sha(qpath)
   p['input_sha256']={f:sha(P/f) for f in [f'data/processed/{source}/manifest.json',f'data/processed/{source}/positivity.json',canonical]}
  new.write_text(json.dumps(p,indent=2)+'\n')
 state.update(active_phase='new_protocol_full_revalidation_then_continuation',candidate_checkpoint_sha256=q['native_replay_checkpoint_sha256'],pipeline='experiments/metrics/dynamic_regularized_full_reference_20261101_v3_workflow.json',original_failure_preserved=True);atomic_json(out,state)
 result=subprocess.run([str(P/'.venv/bin/python'),'scripts/run_regularized_full_reference.py','--protocol','configs/physics/dynamic_regularized_full_reference_20261101_v3.json'],cwd=P)
 w=json.loads((P/state['pipeline']).read_text());state.update(complete=True,pipeline_finished=True,pipeline_failure=w.get('failure'),return_code=result.returncode)
 audit=P/'experiments/metrics/dynamic_regularized_stream_third_20261101_v3_seed_audit.json'
 if audit.exists():state['history_issue_revalidated']=bool(json.loads(audit.read_text()).get('passed'))
except BaseException as exc:state.update(complete=True,failure=str(exc),active_phase='stopped')
finally:atomic_json(out,state)
