"""Compose independent native checks with supplemental whole-element bounds."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
pp=P/args.protocol;p=json.loads(pp.read_text());stem=Path(p['output']).stem
native_path=P/f'experiments/metrics/{stem}_verification.json';refined_path=P/f'experiments/metrics/{stem}_bernstein_refinement.json'
n=json.loads(native_path.read_text());r=json.loads(refined_path.read_text())
assert n['complete'] and r['complete'] and r['passed']
assert r['positive_margin']==1e-9
assert n['protocol_sha256']==hashlib.sha256(pp.read_bytes()).hexdigest()
assert r['original_audit_sha256']==hashlib.sha256(native_path.read_bytes()).hexdigest()
for key,path in [('fields_sha256',P/p['fields']),('inputs_sha256',P/'data/processed'/p['reference']/'inputs.npz')]:
 assert n[key]==r['output_sha256'][key]==hashlib.sha256(path.read_bytes()).hexdigest()
if p.get('resume_probe_protocol'):
 parent=json.loads((P/p['resume_probe_protocol']).read_text())
 probe=json.loads((P/p['output']).read_text())
 assert r['output_sha256']['resumed_checkpoint_sha256']==probe['resumed_checkpoint_sha256']==hashlib.sha256((P/parent['checkpoint']).read_bytes()).hexdigest()
assert all(np.isfinite(row['native_residual_N']) and row['native_residual_N']<1e-6 for row in n['records'])
assert all(row.get('independent_L2_norm_relative_error',0)<1e-11 for row in n['records'])
assert max(n['final_checkpoint_history_relative_errors'].values())<1e-7
for i,row in enumerate(n['records']):
 for state in ['endpoint','stress']:
  if not row[state+'_unresolved_elements']:continue
  proof=[x for x in r['records'] if x['step']==i and x['state']==state];assert len(proof)==1
  proof=proof[0];assert proof['original_unresolved_elements']==row[state+'_unresolved_elements']==len(proof['elements'])
  assert all(e['classification']=='positive_bound' and e['lower_bound']>r['positive_margin'] for e in proof['elements'])
  row['original_'+state+'_unresolved_elements']=row[state+'_unresolved_elements']
  row['original_'+state+'_minimum_Bernstein_bound']=row[state+'_minimum_Bernstein_bound']
  row[state+'_unresolved_elements']=0;row[state+'_minimum_Bernstein_bound']=proof['minimum_positive_lower_bound']
n.update(passed=True,full_cycle=False,supplemental_positivity=True,
 composed_audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 audit_input_sha256={str(q.relative_to(P)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [native_path,refined_path]},
 scope=n['scope']+' Original inconclusive parent bounds retained in the native audit; same-margin subdivision bounds composed here. No continuum, full-cycle or mesh-convergence claim.')
out=P/f'experiments/metrics/{stem}_verification_resolved.json';assert not out.exists();out.write_text(json.dumps(n,indent=2,allow_nan=False));print('COMPOSED LOCAL AUDIT PASSED',True,flush=True)
