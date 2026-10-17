"""Full-cycle same-equation solver agreement, not a discretization accuracy test."""
import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1]
roots = [P/'data/processed'/name for name in
         ['cardiac_quadratic_reference_dt00125_v1', 'cardiac_quadratic_reference_gamg_v1']]
manifests = [json.loads((r/'manifest.json').read_text()) for r in roots]
for root, manifest in zip(roots, manifests):
    assert manifest['complete'] and not (root/'failure.json').exists()
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
for key in ['dt_s','duration_s','displacement_degree','refinement_level','geometry_scales',
            'amplitude','timing_scale','quadrature_degree']:
    assert manifests[0]['config'][key] == manifests[1]['config'][key]
inputs = [np.load(r/'inputs.npz') for r in roots]
for key in ['time','pressure','coords','cells','dof_coordinates','canonical_cell_dofs']:
    np.testing.assert_array_equal(inputs[0][key], inputs[1][key])
fields = [np.load(r/'displacement_coefficients.npy', mmap_mode='r') for r in roots]
assert fields[0].shape == fields[1].shape
records = []; error2 = reference2 = 0.
for i, (a, b) in enumerate(zip(*fields)):
    difference = b-a
    e2 = float(difference@difference); r2 = float(a@a)
    error2 += e2; reference2 += r2
    records.append(dict(step=i, time_s=float(inputs[0]['time'][i]),
        relative_coefficient_difference=float(np.sqrt(e2/max(r2, 1e-60))),
        maximum_absolute_coefficient_difference_m=float(abs(difference).max())))
maximum_relative = max(r['relative_coefficient_difference'] for r in records)
maximum_absolute = max(r['maximum_absolute_coefficient_difference_m'] for r in records)
result = dict(complete=True, engineering_only=True,
    passed=bool(maximum_relative < 1e-6 and maximum_absolute < 1e-9),
    n_steps=len(records), maximum_relative_coefficient_difference=maximum_relative,
    maximum_absolute_coefficient_difference_m=maximum_absolute,
    global_relative_coefficient_difference=float(np.sqrt(error2/max(reference2, 1e-60))),
    declared_tolerances=dict(maximum_frame_relative=1e-6, maximum_absolute_m=1e-9),
    reference_output_sha256=[m['output_sha256'] for m in manifests],
    source_segments=[m['source_segments'] for m in manifests], records=records,
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Both native trajectories start from rest and recursively update their own histories. All800 states have identical forcing, space and discretization. Coefficient agreement at all displacement DOFs checks same-root accumulation. Source pressure/geometry/DOF maps are identical and both files checksum-verified. This is not a continuum accuracy, temporal convergence, mesh convergence, population generalization or isolated runtime result. Whole-element positivity is audited separately.')
(P/'experiments/metrics/dynamic_quadratic_solver_equivalence.json').write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps({k:result[k] for k in ['complete','passed','maximum_relative_coefficient_difference','maximum_absolute_coefficient_difference_m']}, indent=2), flush=True)
