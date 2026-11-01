"""Recheck immutable diagnostic provenance and independently tabulate its evidence."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
p=json.loads((P/'configs/physics/dynamic_quadratic_regularization_diagnostic_v1.json').read_text());r=json.loads((P/p['output']).read_text());q=json.loads((P/'experiments/metrics/dynamic_quadratic_regularization_diagnostic_qualification.json').read_text())
assert r['complete'] and r['no_physical_state_generated'] and not r['physical_step_recovered']
assert sha(P/p['output'])==q['diagnostic_output_sha256']
for f,h in {**p['source_sha256'],**r['input_sha256']}.items():assert sha(P/f)==h
assert [z['diagonal_regularization_factor'] for z in r['directions']]==p['diagonal_regularization_factors']
for x in r['directions']:
 assert x['linear_qualified'] and x['KSP_reason']>0 and x['true_shifted_linear_residual_N']<=max(1.1e-12,1e-7*r['native_force_N'])
 d=x['first_admissible_descent'];assert d and d['proof']
 matching=[t for t in x['trials'] if t['alpha']==d['alpha']];assert len(matching)==1 and matching[0]['armijo_passed'] and matching[0]['whole_element_passed']
 assert d['native_force_N']<=(1-1e-4*d['alpha'])*r['native_force_N']
 print(x['diagonal_regularization_factor'],d['alpha'],d['native_force_N'])
print('Recorded diagnostic provenance/linear checks/descent/proof linkage passed; no physical recovery claimed.')
