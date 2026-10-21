"""Compose complete per-state whole-element proofs without claiming convergence."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
p=json.loads((P/args.protocol).read_text());cfg=json.loads((P/p['config']).read_text());root=P/'data/processed'/cfg['name']
m=json.loads((root/'manifest.json').read_text());assert m['complete'] and m['completed_steps']==m['n_steps']
proofs=json.loads((root/'positivity.json').read_text());assert len(proofs)==m['n_steps']
for i,r in enumerate(proofs):
 assert r['step']==i and all(r[s]['passed'] and r[s]['minimum_Bernstein_bound']>1e-9 for s in ['endpoint','stress'])
records=[]
for r in proofs:
 row=dict(step=r['step'],time_s=r['time_s'])
 for s in ['endpoint','stress']:
  row[s+'_minimum_Bernstein_lower_bound']=r[s]['minimum_Bernstein_bound']
  row[s+'_unresolved_elements']=0
  row[s+'_original_parent_unresolved_elements']=r[s].get('original_unresolved_elements',0)
 records.append(row)
out=dict(complete=True,reference_name=cfg['name'],config_sha256=m['config_sha256'],output_sha256=m['output_sha256'],
 checked_steps=m['n_steps'],tetrahedra=m['tetrahedra'],positive_margin=1e-9,
 minimum_endpoint_lower_bound=min(r['endpoint']['minimum_Bernstein_bound'] for r in proofs),
 minimum_stress_lower_bound=min(r['stress']['minimum_Bernstein_bound'] for r in proofs),
 unresolved_endpoint_element_states=0,unresolved_stress_element_states=0,records=records,
 whole_element_positivity_sufficient_condition_passed=True,
 detailed_proof_sha256=hashlib.sha256((root/'positivity.json').read_bytes()).hexdigest(),
 executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 scope='Every inherited/new endpoint and stress state satisfies the same1e-9 cubic Bernstein criterion, with deterministic tetrahedral subdivision for inconclusive parent bounds. Detailed original parent counts/bounds, child bounds and polynomial checks are retained in positivity.json. Chunked implementation independently matched four legacy/native-map audits. Float64,not interval arithmetic; no numerical-convergence or continuum claim.')
(P/f'experiments/metrics/{cfg["name"]}_bernstein.json').write_text(json.dumps(out,indent=2)+'\n')
print('COMPLETE PARALLEL WHOLE-ELEMENT PROOFS',m['n_steps'],flush=True)
