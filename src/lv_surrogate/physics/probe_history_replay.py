"""Hash-linked causal replay of every accepted parent-probe field before restart."""
from pathlib import Path
import json,hashlib
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def load_parent(project,protocol_path,audit_path):
 P=Path(project);pp=P/protocol_path;p=json.loads(pp.read_text());rp=P/p['output'];r=json.loads(rp.read_text());a=json.loads((P/audit_path).read_text())
 assert r['finished'] and r['all_requested_steps_accepted'] and a['complete'] and a['passed']
 assert sha(pp)==r['protocol_sha256']==a['protocol_sha256'] and sha(P/p['fields'])==a['fields_sha256']
 assert sha(P/p['checkpoint'])==r['probe_checkpoint_sha256']
 root=P/'data/processed'/p['reference'];m=json.loads((root/'manifest.json').read_text());dt=float(m['config']['dt_s']);assert dt==r['dt_s']
 assert sha(root/'restart.npz')==r['checkpoint_sha256']
 fields=np.load(P/p['fields'],mmap_mode='r');saved=np.load(P/p['checkpoint']);count=int(saved['completed_probe_steps']);assert count==len(r['records'])==len(fields)>0
 assert all(z['accepted'] and z['SNES_reason']>0 and z['residual_N']<1e-8 for z in r['records'])
 def advance(history,u):
  old,v,acc=[history[k] for k in ['u_old','v_old','a_old']]
  anew=(u-old-dt*v-(.5-.36)*dt**2*acc)/(.36*dt**2)
  return dict(u_old=np.array(u,copy=True),v_old=v+dt*(.3*acc+.7*anew),a_old=anew)
 if p.get('resume_probe_protocol'):
  seedp=json.loads((P/p['resume_probe_protocol']).read_text());seedr=json.loads((P/seedp['output']).read_text());seedpath=P/seedp['checkpoint']
  assert seedp['reference']==p['reference'] and sha(seedpath)==r['resumed_checkpoint_sha256']==seedr['probe_checkpoint_sha256']
  seed=np.load(seedpath);history={k:seed[k].copy() for k in ['u_old','v_old','a_old']};seed_hash=sha(seedpath)
 else:
  assert 'qualification_step' in p and m['complete'];original=root/'displacement_coefficients.npy'
  assert sha(original)==m['output_sha256']['displacement_coefficients.npy']
  archived=np.load(original,mmap_mode='r');z=np.zeros(fields.shape[1]);history=dict(u_old=z.copy(),v_old=z.copy(),a_old=z.copy())
  for u in archived[:r['start_step']]:history=advance(history,u)
  seed_hash=sha(original)
 previous=None
 for row,u in zip(r['records'],fields):
  previous={k:x.copy() for k,x in history.items()};history=advance(history,u)
 for i,row in enumerate(r['records']):assert row['step']==r['start_step']+i
 errors={}
 for k in history:
  np.testing.assert_allclose(history[k],saved[k],rtol=1e-7,atol=1e-10)
  errors[k]=float(np.linalg.norm(history[k]-saved[k])/max(np.linalg.norm(saved[k]),1e-30))
 np.testing.assert_array_equal(saved['u'],fields[-1]);np.testing.assert_array_equal(saved['u_old'],fields[-1])
 metadata=dict(parent_protocol_sha256=sha(pp),parent_output_sha256=sha(rp),parent_audit_sha256=sha(P/audit_path),parent_fields_sha256=sha(P/p['fields']),parent_checkpoint_sha256=sha(P/p['checkpoint']),seed_history_sha256=seed_hash,parent_steps_replayed=count,relative_history_errors=errors,all_fields_causally_replayed=True)
 return {k:saved[k].copy() for k in history},previous,r['start_step']+count,metadata
