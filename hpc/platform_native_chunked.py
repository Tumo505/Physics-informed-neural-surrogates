"""Checkpointed serial audit; unchanged force, norm, field, history and proof criteria."""
import argparse,hashlib,json,mmap,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'src'),str(P/'hpc')]
from lv_surrogate.physics.parallel_fem import ParallelP2
from lv_surrogate.physics.reference_fem import df,residual
from registry import sha,atomic
from platform_native import norm_check
from audit_checkpoint import load_latest,commit,save_or_verify

def run(spec,output,mode,checkpoint_root,max_steps=64,qualification_steps=None):
 tick=time.perf_counter();out=P/output;assert not out.exists();s=json.loads((P/spec).read_text());snapshot=json.loads((P/s['snapshot']).read_text())
 source=P/'data/processed'/snapshot['reference'];root=source if mode in ['platform','qualification'] else P/'data/processed'/s['target_name']
 m=json.loads((root/'manifest.json').read_text());cfg=m['config'];available=m['completed_steps'];count=available
 if mode=='qualification':assert qualification_steps==4;count=4
 else:assert qualification_steps is None
 offset=available-count if mode=='qualification' else 0
 binding=dict(mode=mode,source_start=offset,count=count,plan_sha256=sha(P/spec),snapshot_sha256=sha(P/s['snapshot']),manifest_sha256=sha(root/'manifest.json'),inputs_sha256=sha(root/'inputs.npz'),source_checkpoint_sha256=sha(root/'restart.npz'),proof_sha256=sha(root/'positivity.json'),audit_source_sha256=sha(Path(__file__)),checkpoint_source_sha256=sha(P/'hpc/audit_checkpoint.py'))
 store=P/checkpoint_root;latest=load_latest(store,binding);start=len(latest['records']) if latest else 0
 print('AUDIT INITIALIZE',mode,'checked',start,'of',count,flush=True)
 fields=np.load(root/'displacement_coefficients.npy',mmap_mode='r');data=np.load(root/'inputs.npz');saved=np.load(root/'restart.npz')
 if mode=='qualification':assert np.max(np.abs(data['pressure'][offset:available]))>0,'Known audit control must include nonzero loading'
 for f,h in snapshot['numerical_sources'].items():assert sha(P/f)==h
 if mode in ['platform','qualification']:
  for f,h in snapshot['files'].items():assert sha(P/f)==h
  assert available==snapshot['accepted_steps']
 proof=json.loads((root/'positivity.json').read_text());assert len(proof)==available
 for i,r in enumerate(proof):assert r['step']==i and all(r[k]['passed'] and r[k]['minimum_Bernstein_bound']>1e-9 for k in ['endpoint','stress'])
 if mode=='final':
  assert m['complete'] and available==m['n_steps']
  if latest is None:
   for name,digest in m['output_sha256'].items():assert sha(root/name)==digest
 adapter=ParallelP2(P,snapshot['source_reference'],cfg);assert adapter.comm.size==1
 records=latest['records'] if latest else [];parent=latest['manifest_sha256'] if latest else None
 if latest:
  for k,v in latest['histories'].items():adapter.assign(k,v)
  np.testing.assert_array_equal(latest['histories']['u'],fields[offset+start-1]);np.testing.assert_array_equal(latest['histories']['u_old'],fields[offset+start-1])
 else:
  for k in ['u','u_old','v_old','a_old']:getattr(adapter.problem,k).vector().zero()
  # Control starts at four known loaded states; replay earlier native updates only.
  for j in range(offset):
   adapter.assign('u',fields[j]);adapter.problem._update_fields()
   if (j+1)%64==0:print('CONTROL NATIVE HISTORY REPLAY',j+1,'/',offset,flush=True)
 # Revalidate the link between every already checked field and saved audit records.
 for r in records:
  assert hashlib.sha256(np.asarray(fields[offset+r['step']]).tobytes()).hexdigest()==r['field_sha256'],'Previously audited field changed'
 selected=set(np.linspace(0,count-1,min(6,count),dtype=int));stop=min(start+max_steps,count);history=None
 initialized=time.perf_counter()-tick
 for i in range(start,stop):
  began=time.perf_counter();source_step=offset+i;u=fields[source_step];field_sha=hashlib.sha256(np.asarray(u).tobytes()).hexdigest();adapter.assign('u',u);adapter.problem.parameters['p'].assign(float(data['pressure'][source_step]))
  force=float(residual(adapter.problem).norm('l2'));assert np.isfinite(force) and force<1e-8,f'Native force failed at{i}: {force}'
  original=m['steps'][source_step];assert original['step']==source_step and original['residual_l2_N']<1e-8
  if original['SNES_reason'] is None:
   provenance=original['native_acceptance_provenance'];assert original['native_reason_record_lost']
   for key,digest_key in [('failure_path','failure_sha256'),('independent_audit','independent_audit_sha256'),('accepted_fields','accepted_fields_sha256')]:assert sha(P/provenance[key])==provenance[digest_key]
   failure=json.loads((P/provenance['failure_path']).read_text());assert failure['persisted_checkpoint_steps']<=source_step<failure['accepted_steps_before_failure']
   np.testing.assert_array_equal(u,np.load(P/provenance['accepted_fields'],mmap_mode='r')[provenance['audit_record']])
  else:assert original['SNES_reason']>0
  row=dict(step=i,source_step=source_step,time_s=float(data['time'][source_step]),pressure_Pa=float(data['pressure'][source_step]),native_residual_N=force,field_sha256=field_sha)
  if i in selected:row['independent_norm_relative_error']=norm_check(adapter,data,u)
  adapter.problem._update_fields();row['audit_seconds']=time.perf_counter()-began;records.append(row)
  if mode=='platform' and i+1==count-2:
   save_or_verify(P/s['known_start_checkpoint'],i+1,{k:adapter.gather(k) for k in ['u','u_old','v_old','a_old']})
  if (i+1)%16==0 or i+1==stop:print('CLUSTER CHECKPOINTED INDEPENDENT FORCE',i+1,'/',count,flush=True)
  if (i+1)%16==0 and hasattr(fields._mmap,'madvise'):fields._mmap.madvise(mmap.MADV_DONTNEED)
 if stop>start:
  history={k:adapter.gather(k) for k in ['u','u_old','v_old','a_old']}
  commit(store,binding,records,history,parent)
 else:history=latest['histories']
 result=dict(complete=False,passed=False,audit_complete=False,chunk_passed=True,checked_steps=stop,total_steps=count,mode=mode,records=records,initialization_seconds=initialized,chunk_wall_seconds=time.perf_counter()-tick,physical_step_generated=False,checkpoint_manifest_sha256=sha(store/f'checked-{stop:06d}'/'manifest.json'),scope='Verified prefix chunk only. Full force/norm/history/field/hash/proof audit pending until audit_complete=True.')
 if stop==count:
  digest=hashlib.sha256()
  for i in range(count):
   raw=np.asarray(fields[offset+i]).tobytes();assert hashlib.sha256(raw).hexdigest()==records[i]['field_sha256'],'Audited field changed before finalization';digest.update(raw)
  if mode=='final':
   for name,expected in m['output_sha256'].items():assert sha(root/name)==expected
  differences={}
  if mode=='qualification':
   full=hashlib.sha256()
   for j in range(available):full.update(np.asarray(fields[j]).tobytes())
   assert full.hexdigest()==snapshot['prefix_fields_sha256'],'Control replay fields changed'
  if mode in ['platform','qualification','seed','final']:
   for k,v in history.items():
    diff=v-saved[k];differences[k]=dict(pointwise_match=bool(np.allclose(v,saved[k],rtol=1e-7,atol=1e-10)),max_abs=float(np.max(abs(diff))),relative_L2=float(np.linalg.norm(diff)/max(np.linalg.norm(saved[k]),1e-30)))
    if mode in ['platform','qualification']:assert differences[k]['relative_L2']<1e-9 and differences[k]['max_abs']<1e-7,'Platform history disagreement exceeds separately declared representation bound'
    else:np.testing.assert_allclose(v,saved[k],rtol=1e-7,atol=1e-10)
  if mode=='platform':
   assert digest.hexdigest()==snapshot['prefix_fields_sha256'];np.testing.assert_array_equal(history['u'],fields[count-1]);np.testing.assert_array_equal(history['u_old'],fields[count-1]);save_or_verify(P/s['canonical_checkpoint'],count,history)
  result.update(complete=True,passed=True,audit_complete=True,history_differences=differences,prefix_sha256=digest.hexdigest(),scope='All requested states independently checked at unchanged force/norm/history/hash/proof criteria; qualification mode covers only the fixed four-state control. No new physical states or continuum/numerical/study validation implied.')
 atomic(out,result);print('AUDIT CHUNK SAVED',stop,'/',count,'full_audit_passed',result['audit_complete'],flush=True);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['platform','seed','final','qualification'],required=True);p.add_argument('--checkpoint-root',required=True);p.add_argument('--max-steps',type=int,default=64);p.add_argument('--qualification-steps',type=int);a=p.parse_args();assert 1<=a.max_steps<=64
 run(a.spec,a.output,a.mode,a.checkpoint_root,a.max_steps,a.qualification_steps)
