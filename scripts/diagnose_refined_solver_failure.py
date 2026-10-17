"""Describe a rejected iterative run without restarting or changing references."""
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
names = ['cardiac_quadratic_reference_refined_v1',
         'cardiac_quadratic_reference_refined_gamg_v1']
roots = [P/'data/processed'/n for n in names]
manifests = [json.loads((r/'manifest.json').read_text()) for r in roots]
failure_path = roots[1]/'failure.json'
failure = json.loads(failure_path.read_text())
assert failure['SNES_reason'] <= 0
for key in ['dt_s', 'duration_s', 'displacement_degree', 'refinement_level',
            'geometry_scales', 'amplitude', 'timing_scale', 'quadrature_degree']:
    assert manifests[0]['config'][key] == manifests[1]['config'][key]
inputs = [np.load(r/'inputs.npz') for r in roots]
for key in ['time', 'pressure', 'coords', 'cells', 'dof_coordinates', 'canonical_cell_dofs']:
    np.testing.assert_array_equal(inputs[0][key], inputs[1][key])
n = min(m['completed_steps'] for m in manifests)
fields = [np.load(r/'displacement_coefficients.npy', mmap_mode='r') for r in roots]
hashes = [hashlib.sha256(), hashlib.sha256()]
records = []
for i in range(n):
    a, b = [np.array(f[i], copy=True) for f in fields]
    for h, value in zip(hashes, [a, b]):
        h.update(value.tobytes())
    difference = b-a
    records.append(dict(step=i, time_s=float(inputs[0]['time'][i]),
        relative_coefficient_difference=float(np.linalg.norm(difference)/max(np.linalg.norm(a), 1e-30)),
        maximum_absolute_coefficient_difference_m=float(abs(difference).max())))
log = P/'experiments/logs/dynamic_quadratic_controlled_mesh_gamg_generation.log'
linear_failure = 'DIVERGED_LINEAR_SOLVE' in log.read_text()
result = dict(diagnostic_complete=True, full_cycle_equivalence_established=False,
    failed_time_s=failure['failed_time_s'], accepted_before_failure=failure['accepted_steps_before_failure'],
    persisted_steps=failure['persisted_checkpoint_steps'], SNES_reason=failure['SNES_reason'],
    native_log_reports_linear_solve_divergence=linear_failure,
    shared_persisted_prefix_steps=n, shared_prefix_end_time_s=records[-1]['time_s'],
    maximum_prefix_relative_coefficient_difference=max(r['relative_coefficient_difference'] for r in records),
    maximum_prefix_absolute_coefficient_difference_m=max(r['maximum_absolute_coefficient_difference_m'] for r in records),
    accepted_prefix_coefficient_sha256=[h.hexdigest() for h in hashes],
    input_sha256=[hashlib.sha256((r/'inputs.npz').read_bytes()).hexdigest() for r in roots],
    failure_sha256=hashlib.sha256(failure_path.read_bytes()).hexdigest(),
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records,
    scope='Partial engineering diagnostic only. Compare solely persisted accepted states, hashing those coefficient bytes while the direct reference continues. Unpersisted accepted states and failed iterates are excluded. Native log identifies a linear-solve failure; its underlying preconditioner/Krylov cause was not recorded and is unresolved. Prefix agreement does not establish full-cycle solver equivalence, positivity, numerical convergence or continuum accuracy. Original failure is retained and no trajectory is restarted.')
(P/'experiments/metrics/dynamic_quadratic_refined_solver_failure_diagnosis.json').write_text(
    json.dumps(result, indent=2, allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k!='records'}, indent=2))
