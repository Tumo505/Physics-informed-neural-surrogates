"""Fault-inject acceptance transaction crash points using real qualified native fields."""
from pathlib import Path
import json,sys,tempfile,shutil,hashlib
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.durable_reference import atomic_json,recover,sha
p=json.loads((P/'configs/physics/dynamic_quadratic_regularized_forcing_coarse_v1.json').read_text());r=json.loads((P/p['output']).read_text());a=json.loads((P/'experiments/metrics/dynamic_quadratic_regularized_forcing_coarse_v1_verification.json').read_text());assert a['passed']
f=np.load(P/p['fields']);native=np.load(P/p['checkpoint']);states=[]
for stage in ['journal_before_rotation','old_rotated','checkpoint_rotated','positivity_written']:
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);(root/'step_journal').mkdir();c=np.array([f[-1]])
  m=dict(completed_steps=0,config_sha256='test-qualified-transaction',steps=[])
  np.savez_compressed(root/'restart.npz',completed_steps=0,u=np.zeros(f.shape[1]));prior=sha(root/'restart.npz')
  np.savez_compressed(root/'restart.pending.npz',completed_steps=1,**{k:native[k] for k in ['u','u_old','v_old','a_old']});digest=sha(root/'restart.pending.npz')
  rec=dict(step=0,SNES_reason=r['records'][-1]['SNES_reason'],nonlinear_iterations=0,residual_l2_N=a['records'][-1]['native_residual_N']);proof={s:dict(passed=True,minimum_Bernstein_bound=a['records'][-1][s+'_minimum_Bernstein_bound']) for s in ['endpoint','stress']}
  atomic_json(root/'positivity.json',[]);atomic_json(root/'manifest.json',m);atomic_json(root/'step_journal/000000.json',dict(record=rec,positivity=proof,config_sha256=m['config_sha256'],coefficient_sha256=hashlib.sha256(c[0].tobytes()).hexdigest(),input_checkpoint_sha256=prior,accepted_checkpoint_sha256=digest))
  if stage!='journal_before_rotation':(root/'restart.npz').replace(root/'restart.previous.npz')
  if stage in ['checkpoint_rotated','positivity_written']:(root/'restart.pending.npz').replace(root/'restart.npz')
  if stage=='positivity_written':atomic_json(root/'positivity.json',[proof])
  assert recover(root,m,c) and m['completed_steps']==1 and sha(root/'restart.npz')==digest
  assert not recover(root,m,c)
  states.append(dict(crash_point=stage,passed=True,exact_native_checkpoint_preserved=True))
out=P/'experiments/metrics/dynamic_regularized_stream_journal_qualification.json';assert not out.exists();out.write_text(json.dumps(dict(complete=True,passed=True,records=states,source_sha256=sha(P/'src/lv_surrogate/physics/durable_reference.py'),scope='Transaction fault injection with real qualified coarse endpoint/native histories. No physical solve,synthetic SNES reason or regenerated history. Does not qualify full reference mechanics.'),indent=2)+'\n');print('Native-history durable acceptance qualification passed')
