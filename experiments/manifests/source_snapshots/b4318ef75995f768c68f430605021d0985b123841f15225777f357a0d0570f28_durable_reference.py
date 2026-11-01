"""Durable acceptance transaction: recover saved native histories, never regenerate them."""
import hashlib,json,os
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,value):
 path=Path(path);temp=path.with_suffix('.pending.json')
 with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 temp.replace(path)
 # Persist the rename itself as well as file contents.
 fd=os.open(str(path.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def recover(root,m,c):
 root=Path(root);index=m['completed_steps'];jp=root/'step_journal'/f'{index:06d}.json';cp=root/'restart.npz'
 if not jp.exists():
  assert cp.exists() and int(np.load(cp)['completed_steps'])==index
  return False
 j=json.loads(jp.read_text());r=j['record'];assert r['step']==index and r['SNES_reason']>0 and r['nonlinear_iterations']==0 and r['residual_l2_N']<1e-8
 assert j['config_sha256']==m['config_sha256'] and hashlib.sha256(np.asarray(c[index]).tobytes()).hexdigest()==j['coefficient_sha256']
 assert all(j['positivity'][s]['passed'] and j['positivity'][s]['minimum_Bernstein_bound']>1e-9 for s in ['endpoint','stress'])
 done=cp.exists() and sha(cp)==j['accepted_checkpoint_sha256']
 prior=root/'restart.previous.npz' if done or not cp.exists() else cp
 assert prior.exists() and sha(prior)==j['input_checkpoint_sha256']
 if not done:
  pending=root/'restart.pending.npz';assert pending.exists() and sha(pending)==j['accepted_checkpoint_sha256']
  if cp.exists():cp.replace(root/'restart.previous.npz')
  pending.replace(cp)
 saved=np.load(cp);assert int(saved['completed_steps'])==index+1;np.testing.assert_array_equal(saved['u'],c[index]);np.testing.assert_array_equal(saved['u_old'],c[index])
 pos=json.loads((root/'positivity.json').read_text());assert len(pos) in [index,index+1]
 if len(pos)==index:pos.append(j['positivity'])
 else:assert pos[-1]==j['positivity']
 atomic_json(root/'positivity.json',pos);m['steps'].append(r);m['completed_steps']=index+1
 m.setdefault('journal_recoveries',[]).append(dict(step=index,native_saved_histories_preserved=True,physical_equation_resolved_again=False))
 atomic_json(root/'manifest.json',m);return True
