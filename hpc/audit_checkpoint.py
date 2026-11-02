"""Immutable, hash-linked audit bundles containing actual saved native histories."""
import hashlib,json,os,uuid
from pathlib import Path
import numpy as np
from registry import sha

def sync_dir(path):
 fd=os.open(str(path),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def write_json(path,value):
 with Path(path).open('x') as f:
  json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def load_latest(root,binding):
 root=Path(root);previous=None;latest=None;last_count=0;prior_records=[]
 for directory in sorted(root.glob('checked-*')):
  manifest=directory/'manifest.json';meta=json.loads(manifest.read_text())
  assert meta['binding']==binding,'Audit input/source/runtime binding changed'
  assert meta['parent_manifest_sha256']==previous,'Audit checkpoint chain changed'
  count=meta['checked_steps'];assert count>last_count and directory.name==f'checked-{count:06d}'
  assert sha(directory/'records.json')==meta['records_sha256'],'Audit records changed'
  assert sha(directory/'native.npz')==meta['native_sha256'],'Audit native checkpoint changed'
  records=json.loads((directory/'records.json').read_text());assert len(records)==count
  assert [r['step'] for r in records]==list(range(count))
  assert records[:last_count]==prior_records,'Previously accepted audit records changed'
  assert all(np.isfinite(r['native_residual_N']) and r['native_residual_N']<1e-8 for r in records)
  previous=sha(manifest);last_count=count;prior_records=records;latest=dict(directory=directory,manifest_sha256=previous,meta=meta,records=records)
 if latest:
  with np.load(latest['directory']/'native.npz') as cp:
   assert int(cp['completed_steps'])==last_count
   latest['histories']={k:cp[k].copy() for k in ['u','u_old','v_old','a_old']}
 return latest

def commit(root,binding,records,histories,parent=None,fault_hook=lambda _:None):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);count=len(records)
 assert count>0 and [r['step'] for r in records]==list(range(count))
 assert all(np.isfinite(r['native_residual_N']) and r['native_residual_N']<1e-8 for r in records)
 assert set(histories)=={'u','u_old','v_old','a_old'} and all(np.isfinite(v).all() for v in histories.values())
 final=root/f'checked-{count:06d}';assert not final.exists(),'Never overwrite an accepted audit bundle'
 pending=root/('.pending-'+uuid.uuid4().hex);pending.mkdir()
 with (pending/'native.npz').open('xb') as f:
  np.savez_compressed(f,completed_steps=count,**histories);f.flush();os.fsync(f.fileno())
 fault_hook('native_saved')
 write_json(pending/'records.json',records)
 write_json(pending/'manifest.json',dict(checked_steps=count,binding=binding,parent_manifest_sha256=parent,records_sha256=sha(pending/'records.json'),native_sha256=sha(pending/'native.npz'),scope='Verified audit prefix and actual native histories only; no new physical states or full-audit pass implied.'))
 sync_dir(pending);fault_hook('before_publish');pending.rename(final);sync_dir(root);fault_hook('after_publish')
 return sha(final/'manifest.json')

def save_or_verify(path,count,histories):
 """Preserve previously published canonical histories if finalization was interrupted."""
 path=Path(path)
 if path.exists():
  with np.load(path) as cp:
   assert int(cp['completed_steps'])==count
   for k,v in histories.items():np.testing.assert_array_equal(cp[k],v)
  return
 temporary=path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
 with temporary.open('xb') as f:
  np.savez_compressed(f,completed_steps=count,**histories);f.flush();os.fsync(f.fileno())
 temporary.rename(path);sync_dir(path.parent)
