"""Gate continuation on every inherited state, not just the repaired endpoint."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.bernstein import BernsteinAudit
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
p=json.loads((P/args.protocol).read_text());cfg=json.loads((P/p['config']).read_text());root=P/'data/processed'/cfg['name']
mp=root/'manifest.json';m=json.loads(mp.read_text());out=root/'positivity.json'
records=json.loads(out.read_text()) if out.exists() else []
if m['prefix_positivity_passed']:assert len(records)>=m['seed_steps'];raise SystemExit(0)
if p.get('inherit_verified_prefix_positivity') and not records:
 source=P/'data/processed'/p['source_reference'];sm=json.loads((source/'manifest.json').read_text())
 b=json.loads((P/p['inherit_verified_prefix_positivity']).read_text())
 assert sm['complete'] and b['complete'] and b['whole_element_positivity_sufficient_condition_passed'] and b['output_sha256']==sm['output_sha256']
 def sha(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
  return h.hexdigest()
 assert all(sha(source/n)==h for n,h in sm['output_sha256'].items())
 for r in b['records'][:m['seed_steps']]:
  records.append(dict(step=r['step'],time_s=r['time_s'],**{s:dict(passed=True,minimum_Bernstein_bound=r[s+'_minimum_Bernstein_lower_bound'],inherited_whole_element_proof=True) for s in ['endpoint','stress']}))
else:
 audit=BernsteinAudit(np.load(root/'inputs.npz'));fields=np.load(root/'displacement_coefficients.npy',mmap_mode='r')
 for i in range(len(records),m['seed_steps']):
  old=fields[i-1] if i else np.zeros(fields.shape[1]);tick=time.perf_counter();proof=audit.pair(fields[i],old)
  records.append(dict(step=i,time_s=m['steps'][i]['time_s'],**proof))
  if (i+1)%5==0 or i+1==m['seed_steps']:
   temporary=out.with_suffix('.pending.json');temporary.write_text(json.dumps(records,indent=2));temporary.replace(out)
   print('WHOLE INHERITED PREFIX',i+1,'/',m['seed_steps'],'last_seconds',time.perf_counter()-tick,flush=True)
assert len(records)==m['seed_steps'];out.write_text(json.dumps(records,indent=2))
m['prefix_positivity_passed']=True;m['prefix_whole_element_checked_steps']=len(records)
temporary=mp.with_suffix('.pending.json');temporary.write_text(json.dumps(m,indent=2));temporary.replace(mp)
print('ENTIRE INHERITED PREFIX POSITIVE',len(records),flush=True)
