"""Cross-platform force/history qualification using exact cached mesh and canonical maps."""
import argparse,json,sys,time,hashlib,mmap
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'src'),str(P/'hpc')]
from lv_surrogate.physics.parallel_fem import ParallelP2
from lv_surrogate.physics.reference_fem import df,residual
from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit
from registry import sha,atomic

def norm_check(adapter,data,u):
 from quadrature import squared_norm
 independent=squared_norm(data,u)
 native=float(df.assemble(df.inner(adapter.problem.u,adapter.problem.u)*df.dx(metadata={'quadrature_degree':6})));np.testing.assert_allclose(independent,native,rtol=1e-11,atol=1e-24);return abs(independent-native)/max(native,1e-30)

def run(spec,output,mode):
 out=P/output;assert not out.exists();s=json.loads((P/spec).read_text());snapshot=json.loads((P/s['snapshot']).read_text());source=P/'data/processed'/snapshot['reference'];root=source if mode=='platform' else P/'data/processed'/s['target_name'];m=json.loads((root/'manifest.json').read_text());cfg=m['config'];count=m['completed_steps'];fields=np.load(root/'displacement_coefficients.npy',mmap_mode='r');data=np.load(root/'inputs.npz');saved=np.load(root/'restart.npz');adapter=ParallelP2(P,snapshot['source_reference'],cfg);assert adapter.comm.size==1
 for f,h in snapshot['numerical_sources'].items():assert sha(P/f)==h
 if mode=='platform':
  for f,h in snapshot['files'].items():assert sha(P/f)==h
  assert count==snapshot['accepted_steps']
 proof=json.loads((root/'positivity.json').read_text());assert len(proof)==count
 for i,r in enumerate(proof):assert r['step']==i and all(r[k]['passed'] and r[k]['minimum_Bernstein_bound']>1e-9 for k in ['endpoint','stress'])
 if mode=='final':
  assert m['complete'] and count==m['n_steps']
  for name,digest in m['output_sha256'].items():assert sha(root/name)==digest
 for k in ['u','u_old','v_old','a_old']:getattr(adapter.problem,k).vector().zero()
 records=[];digest=hashlib.sha256();started=time.perf_counter();selected=set(np.linspace(0,count-1,min(6,count),dtype=int));result=dict(complete=False,passed=False,mode=mode,records=records,physical_step_generated=False,checked_steps=0)
 try:
  for i in range(count):
   u=fields[i];digest.update(np.asarray(u).tobytes());adapter.assign('u',u);adapter.problem.parameters['p'].assign(float(data['pressure'][i]));force=float(residual(adapter.problem).norm('l2'));assert np.isfinite(force) and force<1e-8,f'Native force failed at{i}: {force}'
   original_record=m['steps'][i];assert original_record['step']==i and original_record['residual_l2_N']<1e-8
   if original_record['SNES_reason'] is None:
    provenance=original_record['native_acceptance_provenance'];assert original_record['native_reason_record_lost']
    for key,digest_key in [('failure_path','failure_sha256'),('independent_audit','independent_audit_sha256'),('accepted_fields','accepted_fields_sha256')]:assert sha(P/provenance[key])==provenance[digest_key]
    failure=json.loads((P/provenance['failure_path']).read_text());assert failure['persisted_checkpoint_steps']<=i<failure['accepted_steps_before_failure']
    np.testing.assert_array_equal(u,np.load(P/provenance['accepted_fields'],mmap_mode='r')[provenance['audit_record']])
   else:assert original_record['SNES_reason']>0
   row=dict(step=i,time_s=float(data['time'][i]),native_residual_N=force)
   if i in selected:row['independent_norm_relative_error']=norm_check(adapter,data,u)
   adapter.problem._update_fields();records.append(row)
   if mode=='platform' and i+1==count-2:
    np.savez_compressed(P/s['known_start_checkpoint'],completed_steps=i+1,**{k:adapter.gather(k) for k in ['u','u_old','v_old','a_old']})
   if (i+1)%16==0 or i+1==count:
    result['checked_steps']=i+1;atomic(out,result);print('CLUSTER INDEPENDENT FORCE',i+1,'/',count,flush=True)
    if hasattr(fields._mmap,'madvise'):fields._mmap.madvise(mmap.MADV_DONTNEED)
  history={k:adapter.gather(k) for k in ['u','u_old','v_old','a_old']};differences={}
  for k,v in history.items():
   diff=v-saved[k];differences[k]=dict(pointwise_match=bool(np.allclose(v,saved[k],rtol=1e-7,atol=1e-10)),max_abs=float(np.max(abs(diff))),relative_L2=float(np.linalg.norm(diff)/max(np.linalg.norm(saved[k]),1e-30)))
   if mode=='platform':assert differences[k]['relative_L2']<1e-9 and differences[k]['max_abs']<1e-7,'Platform history disagreement exceeds separately declared representation bound'
   else:np.testing.assert_allclose(v,saved[k],rtol=1e-7,atol=1e-10)
  if mode=='platform':
   assert digest.hexdigest()==snapshot['prefix_fields_sha256'];np.testing.assert_array_equal(history['u'],fields[count-1]);np.testing.assert_array_equal(history['u_old'],fields[count-1]);np.savez_compressed(P/s['canonical_checkpoint'],completed_steps=count,**history)
  result.update(complete=True,passed=True,checked_steps=count,history_differences=differences,prefix_sha256=digest.hexdigest(),checkpoint_sha256=sha(root/'restart.npz'),canonical_checkpoint_sha256=sha(P/s['canonical_checkpoint']) if mode=='platform' else None,wall_seconds=time.perf_counter()-started,scope='All accepted canonical fields independently reassembled atstrict1e-8N usingcachedphysicalmesh/fibers andverifiedone-to-one maps;selectedindependentquadraturenormchecks. Platformmode records importedhistorydifferences and serializesaseparate nativeconsistentrepresentation;originalcheckpoint/tolerance unchanged. Seed/finalmode requireoriginalpointwise1e-7/1e-10. Wholeelementprooflinkage retained;not continuum/numerical/studyvalidation.')
 except Exception as exc:result.update(complete=True,passed=False,failure=str(exc),wall_seconds=time.perf_counter()-started)
 finally:atomic(out,result)
 assert result['passed'],result.get('failure')
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--output',required=True);ap.add_argument('--mode',choices=['platform','seed','final'],required=True);a=ap.parse_args();run(a.spec,a.output,a.mode)
