"""Compare paired recovered states only after both bounded probes accepted."""
import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1]
protocols = [json.loads((P/f'configs/physics/dynamic_quadratic_halfstep_recovery_{m}.json').read_text()) for m in ['mumps','gamg']]
results = [json.loads((P/p['output']).read_text()) for p in protocols]
assert all(r.get('finished') and r.get('all_requested_steps_accepted') for r in results)
assert results[0]['checkpoint_sha256'] == results[1]['checkpoint_sha256']
assert results[0]['reference_config_sha256'] == results[1]['reference_config_sha256']
assert results[0]['dt_s'] == results[1]['dt_s']
arrays = [np.load(P/p['fields'],mmap_mode='r') for p in protocols]
assert arrays[0].shape == arrays[1].shape
records = []
for i,(a,b) in enumerate(zip(*arrays)):
    assert results[0]['records'][i]['time_s'] == results[1]['records'][i]['time_s']
    records.append(dict(time_s=results[0]['records'][i]['time_s'],
        relative_coefficient_difference=float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-30)),
        maximum_absolute_difference_m=float(abs(b-a).max())))
passed = all(r['relative_coefficient_difference']<1e-6 and r['maximum_absolute_difference_m']<1e-9 for r in records)
result = dict(complete=True,passed=passed,full_cycle=False,records=records,
    declared_tolerances=dict(relative=1e-6,absolute_m=1e-9),
    field_sha256=[hashlib.sha256((P/p['fields']).read_bytes()).hexdigest() for p in protocols],
    original_checkpoint_sha256=results[0]['checkpoint_sha256'],
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Two recursive steps from the same saved accepted history at identical mesh, timestep, quadrature, loading and strict nonlinear stopping. Same-root engineering control only; no full-cycle equivalence, numerical convergence or continuum claim. Independently verify native residuals, hashes, whole-element positivity and norms separately before accepting recovered data.')
(P/'experiments/metrics/dynamic_quadratic_halfstep_recovery_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2))
