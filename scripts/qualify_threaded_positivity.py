"""Compare serial and threaded proof semantics on two audited loaded states."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.bernstein import BernsteinAudit
from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit
p=json.loads((P/'configs/physics/dynamic_quadratic_mpi_admissible_consecutive_v1.json').read_text());a=json.loads((P/'experiments/metrics/dynamic_quadratic_mpi_admissible_consecutive_v1_verification_resolved.json').read_text());assert a['passed']
u=np.load(P/p['fields']);parent=json.loads((P/p['resume_probe_protocol']).read_text());old=np.load(P/parent['checkpoint'])['u_old'];data=np.load(P/'data/processed'/p['reference']/'inputs.npz');serial=BernsteinAudit(data);threaded=ThreadedBernsteinAudit(data);records=[]
for i,field in enumerate(u):
 t=time.perf_counter();baseline=serial.pair(field,old);base_seconds=time.perf_counter()-t;t=time.perf_counter();parallel=threaded.pair(field,old);parallel_seconds=time.perf_counter()-t
 for state in ['endpoint','stress']:
  b=baseline[state];r=parallel[state]
  assert b['passed'] and r['passed'] and r['disjoint_cells_total']==len(serial.maps)
  for k in ['minimum_Bernstein_bound','minimum_parent_Bernstein_bound']:np.testing.assert_allclose(b[k],r[k],rtol=0,atol=1e-12)
  assert b['original_unresolved_elements']==r['original_unresolved_elements']
  assert len(b['refined_elements'])==len(r['refined_elements'])
  for one,two in zip(b['refined_elements'],r['refined_elements']):
   for k in ['element','classification','leaf_tetrahedra','maximum_depth']:assert one[k]==two[k]
   np.testing.assert_allclose(one['lower_bound'],two['lower_bound'],rtol=0,atol=1e-12)
 records.append(dict(time_s=a['records'][i]['time_s'],serial_seconds=base_seconds,threaded_seconds=parallel_seconds,speedup=base_seconds/parallel_seconds,semantic_checks_passed=True,serial_proof=baseline,threaded_proof=parallel));old=field
 print('THREADED PROOF EQUIVALENCE',i,base_seconds,parallel_seconds,flush=True)
threaded.close();paths=['scripts/qualify_threaded_positivity.py','src/lv_surrogate/physics/threaded_bernstein.py','src/lv_surrogate/physics/bernstein.py'];sha=lambda p:hashlib.sha256((P/p).read_bytes()).hexdigest()
out=P/'experiments/metrics/dynamic_quadratic_threaded_positivity_qualification.json';assert not out.exists()
out.write_text(json.dumps(dict(complete=True,passed=True,records=records,source_sha256={p:sha(p) for p in paths},scope='Same disjoint union of all tetrahedra,same float64 cubic Bernstein polynomial,margin1e-9,maxdepth12,longest-edge subdivision. Global cell IDs,unresolved parent counts,positive bounds and refinement topology match serial known-state proofs. Parallelism changes evaluation schedule only. Not interval arithmetic or a full-cycle/numerical convergence claim.'),indent=2)+'\n')
