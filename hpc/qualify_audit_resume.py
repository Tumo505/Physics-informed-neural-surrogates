"""Known loaded native audit: two-plus-two resumed states versus four fresh states."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from registry import sha,atomic
from audit_checkpoint import load_latest
P=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=P/a.output;assert not out.exists();plan=json.loads((P/a.spec).read_text());folder=out.parent
 native=[str(P/'.tools/bin/micromamba'),'run','-p',str(P/plan['native_prefix']),'python'];reports=[]
 for name,store,steps in [('split-first','split',2),('split-second','split',2),('fresh-four','fresh',4)]:
  report=folder/(name+'.json');log=folder/(name+'.log');assert not report.exists() and not log.exists()
  argv=native+['hpc/platform_native_chunked.py','--spec',a.spec,'--output',str(report.relative_to(P)),'--mode','qualification','--qualification-steps','4','--max-steps',str(steps),'--checkpoint-root',str((folder/store).relative_to(P))]
  with log.open('x') as stream:result=subprocess.run(argv,cwd=P,stdout=stream,stderr=subprocess.STDOUT)
  assert result.returncode==0,'Known-prefix native audit control failed; inspect '+str(log)
  r=json.loads(report.read_text());assert r['chunk_passed'];reports.append(dict(report=str(report.relative_to(P)),sha256=sha(report),log_sha256=sha(log),checked_steps=r['checked_steps'],audit_complete=r['audit_complete']))
 assert reports[0]['checked_steps']==2 and not reports[0]['audit_complete']
 assert all(r['checked_steps']==4 and r['audit_complete'] for r in reports[1:])
 checkpoints=[]
 for name in ['split','fresh']:
  directory=folder/name;meta=json.loads((directory/'checked-000004/manifest.json').read_text());checkpoints.append(load_latest(directory,meta['binding']))
 assert checkpoints[0]['meta']['binding']==checkpoints[1]['meta']['binding']
 assert any(np.any(checkpoints[0]['histories'][k]!=0) for k in ['v_old','a_old']),'Known tail control must exercise nonzero native histories'
 for k in ['u','u_old','v_old','a_old']:np.testing.assert_array_equal(checkpoints[0]['histories'][k],checkpoints[1]['histories'][k])
 forces=[]
 for left,right in zip(checkpoints[0]['records'],checkpoints[1]['records']):
  assert left['field_sha256']==right['field_sha256']
  assert left['native_residual_N']<1e-8 and right['native_residual_N']<1e-8
  np.testing.assert_allclose(left['native_residual_N'],right['native_residual_N'],rtol=1e-10,atol=1e-12)
  forces.append(abs(left['native_residual_N']-right['native_residual_N']))
 atomic(out,dict(complete=True,passed=True,reports=reports,known_steps=4,source_steps=[r['source_step'] for r in checkpoints[0]['records']],nonzero_native_histories=True,exact_history_equality=True,maximum_native_force_difference_N=max(forces),physical_new_states_generated=False,scope='Fixed four-state loaded-tail stop/restart control after causal native replay; control replay is not a full force audit. Strict original native force/norm/proof/field criteria checked. No full reference or numerical/study validation implied.'))
if __name__=='__main__':main()
